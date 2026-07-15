#!/usr/bin/env python3
"""Train R1/R2 event-camera association embeddings on ordered DSEC-MOT clips."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.dense_targets import IDENTITY_IGNORE_INDEX
from src.data.representations import (
    REPRESENTATION_CHOICES,
    representation_channel_splits,
    representation_channels,
)
from src.data.sequence_dataset import DSECClipDataset, collate_clip_batch
from src.models.simple_detector import (
    SimpleDenseDetector,
    SimpleDetectorConfig,
    normalise_representation_tensor,
    simple_detector_loss,
)
from src.training.simple_detector import (
    DEFAULT_VAL_SEQUENCE,
    LimitedDataset,
    choose_train_val_sequences,
    parse_int_list,
)


def identity_loss(
    embeddings: torch.Tensor,
    identity_targets: torch.Tensor,
    classifier: nn.Linear,
) -> tuple[torch.Tensor, int]:
    """Plain identity cross-entropy over known training identities."""

    known = identity_targets != IDENTITY_IGNORE_INDEX
    if not known.any():
        return embeddings.sum() * 0.0, 0
    logits = classifier(embeddings[known])
    return F.cross_entropy(logits, identity_targets[known]), int(known.sum())


def batch_hard_cosine_triplet_loss(
    embeddings: torch.Tensor,
    identity_targets: torch.Tensor,
    class_targets: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, int]:
    """Class-aware batch-hard triplet loss with cosine distance."""

    if embeddings.shape[0] == 0:
        return embeddings.sum() * 0.0, 0
    normalised = F.normalize(embeddings, dim=1)
    distances = 1.0 - normalised @ normalised.t()
    indices = torch.arange(embeddings.shape[0], device=embeddings.device)
    losses: list[torch.Tensor] = []
    for anchor in range(embeddings.shape[0]):
        identity = identity_targets[anchor]
        if int(identity) == IDENTITY_IGNORE_INDEX:
            continue
        positives = (identity_targets == identity) & (indices != anchor)
        negatives = (
            (identity_targets != identity)
            & (identity_targets != IDENTITY_IGNORE_INDEX)
            & (class_targets == class_targets[anchor])
        )
        if not positives.any() or not negatives.any():
            continue
        hardest_positive = distances[anchor][positives].max()
        hardest_negative = distances[anchor][negatives].min()
        losses.append(F.relu(hardest_positive - hardest_negative + margin))
    if not losses:
        return embeddings.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def class_aware_retrieval_metrics(
    embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    sequences: Sequence[str],
    track_ids: torch.Tensor,
) -> dict[str, float | int]:
    """Compute cosine retrieval mAP and Rank-1 with same-class galleries."""

    count = embeddings.shape[0]
    if class_ids.shape[0] != count or track_ids.shape[0] != count or len(sequences) != count:
        raise ValueError("Retrieval labels must align one-to-one with embeddings.")
    if count == 0:
        return {"retrieval_map": 0.0, "retrieval_rank1": 0.0, "valid_queries": 0}

    vectors = F.normalize(embeddings.float(), dim=1)
    distances = 1.0 - vectors @ vectors.t()
    average_precisions: list[float] = []
    rank1_hits: list[float] = []
    for query in range(count):
        gallery = (class_ids == class_ids[query]).clone()
        gallery[query] = False
        positives = torch.tensor(
            [
                bool(gallery[index])
                and sequences[index] == sequences[query]
                and int(track_ids[index]) == int(track_ids[query])
                for index in range(count)
            ],
            dtype=torch.bool,
            device=gallery.device,
        )
        if not positives.any():
            continue
        gallery_indices = torch.nonzero(gallery, as_tuple=False).flatten()
        order = gallery_indices[distances[query, gallery_indices].argsort()]
        ranked_positive = positives[order]
        positive_ranks = torch.nonzero(ranked_positive, as_tuple=False).flatten() + 1
        precisions = (
            torch.arange(
                1,
                positive_ranks.numel() + 1,
                dtype=torch.float32,
                device=positive_ranks.device,
            )
            / positive_ranks.float()
        )
        average_precisions.append(float(precisions.mean()))
        rank1_hits.append(float(ranked_positive[0]))

    valid_queries = len(average_precisions)
    if valid_queries == 0:
        return {"retrieval_map": 0.0, "retrieval_rank1": 0.0, "valid_queries": 0}
    return {
        "retrieval_map": float(np.mean(average_precisions)),
        "retrieval_rank1": float(np.mean(rank1_hits)),
        "valid_queries": valid_queries,
    }


def checkpoint_selection_key(stats: dict[str, float | int]) -> tuple[float, float, float]:
    """Order checkpoints by mAP, Rank-1, then lower detection loss."""

    return (
        float(stats["retrieval_map"]),
        float(stats["retrieval_rank1"]),
        -float(stats["detection_loss"]),
    )


def is_better_checkpoint(
    candidate: dict[str, float | int],
    incumbent: dict[str, float | int] | None,
) -> bool:
    return incumbent is None or checkpoint_selection_key(candidate) > checkpoint_selection_key(
        incumbent
    )


def _frame_tensors(
    nested: list[list[torch.Tensor]],
    time_index: int,
    device: torch.device,
) -> list[torch.Tensor]:
    return [clip[time_index].to(device, non_blocking=True) for clip in nested]


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.long, device=device)


def run_clip_epoch(
    model: SimpleDenseDetector,
    classifier: nn.Linear,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    background_weight: float,
    bbox_weight: float,
    identity_ce_weight: float,
    triplet_weight: float,
    triplet_margin: float,
    grad_clip_norm: float,
    grad_accum_steps: int,
    log_every: int,
    epoch: int,
    phase: str,
    component_splits: tuple[int, ...] = (),
    compute_retrieval: bool | None = None,
) -> dict[str, float | int]:
    """Run one epoch, resetting recurrent state at every clip microbatch."""

    is_train = optimizer is not None
    if compute_retrieval is None:
        compute_retrieval = not is_train
    model.train(is_train)
    classifier.train(is_train)
    totals: defaultdict[str, float] = defaultdict(float)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and is_train)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    retrieval_embeddings: list[torch.Tensor] = []
    retrieval_classes: list[torch.Tensor] = []
    retrieval_tracks: list[torch.Tensor] = []
    retrieval_sequences: list[str] = []
    started = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        events = batch["events"].to(device, non_blocking=True)
        cls_targets = batch["cls_targets"].to(device, non_blocking=True)
        bbox_targets = batch["bbox_targets"].to(device, non_blocking=True)
        pos_mask = batch["pos_mask"].to(device, non_blocking=True)
        clip_length = events.shape[1]
        batch_size = events.shape[0]

        with torch.autocast(device_type=device.type, enabled=use_amp):
            embedding_state = None
            detection_loss_sum = events.new_zeros(())
            positive_cells = 0
            descriptors_by_frame: list[torch.Tensor] = []
            identities_by_frame: list[torch.Tensor] = []
            classes_by_frame: list[torch.Tensor] = []
            raw_tracks_by_frame: list[torch.Tensor] = []
            sequences_by_frame: list[str] = []

            for time_index in range(clip_length):
                frame = normalise_representation_tensor(events[:, time_index], component_splits)
                outputs = model(frame, embedding_state)
                embedding_state = outputs.get("embedding_state")
                detection_loss, detection_stats = simple_detector_loss(
                    outputs=outputs,
                    cls_targets=cls_targets[:, time_index],
                    bbox_targets=bbox_targets[:, time_index],
                    pos_mask=pos_mask[:, time_index],
                    background_weight=background_weight,
                    bbox_weight=bbox_weight,
                )
                detection_loss_sum = detection_loss_sum + detection_loss
                positive_cells += detection_stats["positive_cells"]

                feature_map = outputs.get("embedding_feature_map")
                if not isinstance(feature_map, torch.Tensor):
                    raise RuntimeError("Embedding training requires embedding_feature_map.")
                boxes = _frame_tensors(batch["roi_boxes"], time_index, device)
                descriptors = model.extract_roi_embeddings(feature_map, boxes)
                descriptors_by_frame.append(descriptors)
                identities_by_frame.append(
                    torch.cat(_frame_tensors(batch["roi_identity_targets"], time_index, device))
                    if sum(int(box.shape[0]) for box in boxes)
                    else _empty_long(device)
                )
                classes_by_frame.append(
                    torch.cat(_frame_tensors(batch["roi_class_ids"], time_index, device))
                    if sum(int(box.shape[0]) for box in boxes)
                    else _empty_long(device)
                )
                raw_tracks_by_frame.append(
                    torch.cat(_frame_tensors(batch["roi_track_ids"], time_index, device))
                    if sum(int(box.shape[0]) for box in boxes)
                    else _empty_long(device)
                )
                for batch_index in range(batch_size):
                    count = int(boxes[batch_index].shape[0])
                    sequence = str(batch["meta"][batch_index][time_index]["sequence"])
                    sequences_by_frame.extend([sequence] * count)

            descriptors = torch.cat(descriptors_by_frame, dim=0)
            identities = torch.cat(identities_by_frame, dim=0)
            classes = torch.cat(classes_by_frame, dim=0)
            raw_tracks = torch.cat(raw_tracks_by_frame, dim=0)
            identity_ce, identity_objects = identity_loss(descriptors, identities, classifier)
            triplet, valid_triplet_anchors = batch_hard_cosine_triplet_loss(
                descriptors,
                identities,
                classes,
                margin=triplet_margin,
            )
            detection_loss = detection_loss_sum / clip_length
            total_loss = (
                detection_loss + identity_ce_weight * identity_ce + triplet_weight * triplet
            )

        if optimizer is not None:
            scaler.scale(total_loss / grad_accum_steps).backward()
            should_step = step % grad_accum_steps == 0 or step == len(loader)
            if should_step:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    parameters = list(model.parameters()) + list(classifier.parameters())
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        clips = int(events.shape[0])
        totals["total_loss"] += float(total_loss.detach()) * clips
        totals["detection_loss"] += float(detection_loss.detach()) * clips
        totals["identity_loss"] += float(identity_ce.detach()) * clips
        totals["triplet_loss"] += float(triplet.detach()) * clips
        totals["clips"] += clips
        totals["positive_cells"] += positive_cells
        totals["identity_objects"] += identity_objects
        totals["valid_triplet_anchors"] += valid_triplet_anchors

        if compute_retrieval:
            retrieval_embeddings.append(descriptors.detach().float().cpu())
            retrieval_classes.append(classes.detach().cpu())
            retrieval_tracks.append(raw_tracks.detach().cpu())
            retrieval_sequences.extend(sequences_by_frame)

        should_log = log_every > 0 and (step == 1 or step % log_every == 0 or step == len(loader))
        if should_log:
            elapsed = time.perf_counter() - started
            rate = totals["clips"] / elapsed if elapsed > 0 else 0.0
            print(
                f"epoch {epoch:03d} {phase} step {step:05d}/{len(loader):05d} "
                f"loss={float(total_loss.detach()):.4f} "
                f"det={float(detection_loss.detach()):.4f} "
                f"id={float(identity_ce.detach()):.4f} "
                f"triplet={float(triplet.detach()):.4f} "
                f"rate={rate:.2f} clips/s",
                flush=True,
            )

    clips = max(totals["clips"], 1.0)
    stats: dict[str, float | int] = {
        "loss": totals["total_loss"] / clips,
        "total_loss": totals["total_loss"] / clips,
        "detection_loss": totals["detection_loss"] / clips,
        "identity_loss": totals["identity_loss"] / clips,
        "triplet_loss": totals["triplet_loss"] / clips,
        "clips": int(totals["clips"]),
        "positive_cells": int(totals["positive_cells"]),
        "identity_objects": int(totals["identity_objects"]),
        "valid_triplet_anchors": int(totals["valid_triplet_anchors"]),
    }
    if compute_retrieval:
        embedding_tensor = (
            torch.cat(retrieval_embeddings, dim=0)
            if retrieval_embeddings
            else torch.empty((0, model.config.embedding_dim))
        )
        class_tensor = (
            torch.cat(retrieval_classes, dim=0)
            if retrieval_classes
            else torch.empty(0, dtype=torch.long)
        )
        track_tensor = (
            torch.cat(retrieval_tracks, dim=0)
            if retrieval_tracks
            else torch.empty(0, dtype=torch.long)
        )
        stats.update(
            class_aware_retrieval_metrics(
                embedding_tensor,
                class_tensor,
                retrieval_sequences,
                track_tensor,
            )
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-sequences", default=None)
    parser.add_argument(
        "--val-sequences",
        default=None,
        help=f"Comma-separated validation sequences; defaults to {DEFAULT_VAL_SEQUENCE}.",
    )
    parser.add_argument(
        "--representation", choices=REPRESENTATION_CHOICES, default="event_frame_voxel_grid"
    )
    parser.add_argument("--num-bins", type=int, default=3)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument("--eros-cache-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--feature-stride", type=int, default=8)
    parser.add_argument("--positive-radius", type=int, default=1)
    parser.add_argument("--class-ids", default=None)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument(
        "--fusion-mode",
        choices=("single", "two_branch", "three_branch", "gated_two_branch"),
        default="gated_two_branch",
    )
    parser.add_argument("--architecture", choices=("simple", "csp_pan"), default="simple")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--embedding-hidden-dim", type=int, default=128)
    parser.add_argument("--roi-size", type=int, default=7)
    recurrent = parser.add_mutually_exclusive_group()
    recurrent.add_argument(
        "--recurrent-embedding",
        dest="embedding_recurrent",
        action="store_true",
        help="Enable R2 spatial ConvGRU recurrence inside the embedding head.",
    )
    recurrent.add_argument(
        "--no-recurrent-embedding",
        dest="embedding_recurrent",
        action="store_false",
        help="Use the non-recurrent R1 embedding head.",
    )
    parser.set_defaults(embedding_recurrent=True)
    parser.add_argument("--identity-ce-weight", type=float, default=1.0)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--bbox-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-val-clips", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/event_reid_embedding/all_classes")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.feature_stride != 8:
        raise SystemExit("The event-ReID benchmark requires --feature-stride 8.")
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad-accum-steps must be positive.")
    if args.embedding_dim <= 0 or args.embedding_hidden_dim <= 0 or args.roi_size <= 0:
        raise SystemExit("Embedding dimension, hidden dimension, and RoI size must be positive.")
    if args.clip_length <= 0 or args.clip_stride <= 0:
        raise SystemExit("Clip length and stride must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")

    component_splits = representation_channel_splits(args.representation, args.num_bins)
    class_ids = parse_int_list(args.class_ids)
    if args.num_classes <= 0:
        raise SystemExit("--num-classes must be positive.")
    if class_ids is not None and args.num_classes != len(class_ids):
        raise SystemExit("--num-classes must match the selected --class-ids.")
    expected_branches = {"two_branch": 2, "three_branch": 3, "gated_two_branch": 2}.get(
        args.fusion_mode
    )
    if expected_branches is not None and len(component_splits) != expected_branches:
        raise SystemExit(
            f"--fusion-mode {args.fusion_mode} requires {expected_branches} components."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    variant = "r2_recurrent" if args.embedding_recurrent else "r1_non_recurrent"
    run_name = args.run_name or (
        f"{args.representation}_bins{args.num_bins}_w{args.model_width}_"
        f"{args.fusion_mode}_{variant}"
    )
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sequences, val_sequences = choose_train_val_sequences(
        root=args.root,
        train_split=args.train_split,
        train_sequences_arg=args.train_sequences,
        val_sequences_arg=args.val_sequences or DEFAULT_VAL_SEQUENCE,
    )
    train_dataset = DSECClipDataset(
        root=args.root,
        split=args.train_split,
        sequences=train_sequences,
        representation=args.representation,
        num_bins=args.num_bins,
        time_window_us=args.time_window_us,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        feature_stride=args.feature_stride,
        positive_radius=args.positive_radius,
        eros_cache_root=args.eros_cache_root,
        class_ids=class_ids,
    )
    val_dataset = DSECClipDataset(
        root=args.root,
        split=args.train_split,
        sequences=val_sequences,
        representation=args.representation,
        num_bins=args.num_bins,
        time_window_us=args.time_window_us,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        feature_stride=args.feature_stride,
        positive_radius=args.positive_radius,
        eros_cache_root=args.eros_cache_root,
        class_ids=class_ids,
        identity_vocabulary=train_dataset.identity_vocabulary,
    )
    if train_dataset.num_identities == 0:
        raise SystemExit("No training identities were found.")

    train_limited = LimitedDataset(train_dataset, args.max_train_clips, args.seed)
    val_limited = LimitedDataset(val_dataset, args.max_val_clips, args.seed)
    if len(train_limited) == 0 or len(val_limited) == 0:
        raise SystemExit(f"Empty clip dataset: train={len(train_limited)} val={len(val_limited)}.")

    model_config = SimpleDetectorConfig(
        in_channels=representation_channels(args.representation, args.num_bins),
        num_classes=args.num_classes,
        feature_stride=args.feature_stride,
        width=args.model_width,
        architecture=args.architecture,
        fusion_mode=args.fusion_mode,
        event_frame_channels=2,
        voxel_grid_channels=2 * args.num_bins,
        component_channels=component_splits,
        embedding_dim=args.embedding_dim,
        embedding_recurrent=args.embedding_recurrent,
        embedding_hidden_dim=args.embedding_hidden_dim,
        embedding_roi_size=args.roi_size,
    )
    model = SimpleDenseDetector(model_config).to(device)
    classifier = nn.Linear(args.embedding_dim, train_dataset.num_identities).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_loader = DataLoader(
        train_limited,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_clip_batch,
    )
    val_loader = DataLoader(
        val_limited,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_clip_batch,
    )
    use_amp = not args.no_amp and device.type == "cuda"

    config = {
        "root": str(args.root),
        "train_split": args.train_split,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "representation": args.representation,
        "num_bins": args.num_bins,
        "time_window_us": args.time_window_us,
        "feature_stride": args.feature_stride,
        "positive_radius": args.positive_radius,
        "class_ids": class_ids,
        "num_classes": args.num_classes,
        "architecture": args.architecture,
        "fusion_mode": args.fusion_mode,
        "model": model_config.to_dict(),
        "embedding_dim": args.embedding_dim,
        "embedding_hidden_dim": args.embedding_hidden_dim,
        "roi_size": args.roi_size,
        "embedding_recurrent": args.embedding_recurrent,
        "identity_ce_weight": args.identity_ce_weight,
        "triplet_weight": args.triplet_weight,
        "triplet_margin": args.triplet_margin,
        "num_identities": train_dataset.num_identities,
        "clip_length": args.clip_length,
        "clip_stride": args.clip_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "background_weight": args.background_weight,
        "bbox_weight": args.bbox_weight,
        "grad_clip_norm": args.grad_clip_norm,
        "max_train_clips": args.max_train_clips,
        "max_val_clips": args.max_val_clips,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_selection: dict[str, float | int] | None = None
    start_epoch = 1
    last_path = output_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        classifier.load_state_dict(checkpoint["identity_classifier_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history_path = output_dir / "history.json"
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.exists()
            else checkpoint.get("history", [])
        )
        start_epoch = int(checkpoint.get("epoch", len(history))) + 1
        best_selection = checkpoint.get("best_selection")
        print(f"Resuming {run_name} from epoch {start_epoch}", flush=True)

    if start_epoch > args.epochs:
        print(f"Training {run_name} is already complete.", flush=True)
        return 0

    print(
        f"Training {run_name}: train={train_sequences} ({len(train_limited)} clips), "
        f"val={val_sequences} ({len(val_limited)} clips), device={device}, AMP={use_amp}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_clip_epoch(
            model=model,
            classifier=classifier,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            use_amp=use_amp,
            background_weight=args.background_weight,
            bbox_weight=args.bbox_weight,
            identity_ce_weight=args.identity_ce_weight,
            triplet_weight=args.triplet_weight,
            triplet_margin=args.triplet_margin,
            grad_clip_norm=args.grad_clip_norm,
            grad_accum_steps=args.grad_accum_steps,
            log_every=args.log_every,
            epoch=epoch,
            phase="train",
            component_splits=component_splits,
            compute_retrieval=False,
        )
        with torch.inference_mode():
            val_stats = run_clip_epoch(
                model=model,
                classifier=classifier,
                loader=val_loader,
                device=device,
                optimizer=None,
                use_amp=use_amp,
                background_weight=args.background_weight,
                bbox_weight=args.bbox_weight,
                identity_ce_weight=args.identity_ce_weight,
                triplet_weight=args.triplet_weight,
                triplet_margin=args.triplet_margin,
                grad_clip_norm=0.0,
                grad_accum_steps=1,
                log_every=args.log_every,
                epoch=epoch,
                phase="val",
                component_splits=component_splits,
                compute_retrieval=True,
            )
        if int(val_stats["valid_queries"]) == 0:
            raise RuntimeError(
                "Validation produced no valid class-aware retrieval queries; "
                "best.pt cannot be selected."
            )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train": train_stats,
                "val": val_stats,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        candidate = {
            "retrieval_map": float(val_stats["retrieval_map"]),
            "retrieval_rank1": float(val_stats["retrieval_rank1"]),
            "detection_loss": float(val_stats["detection_loss"]),
            "epoch": epoch,
        }
        better = is_better_checkpoint(candidate, best_selection)
        if better:
            best_selection = candidate
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "benchmark_config": config,
            "identity_classifier_state": classifier.state_dict(),
            "num_identities": train_dataset.num_identities,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_selection": best_selection,
            "epoch": epoch,
            "selected_epoch": int(best_selection["epoch"]),
            "val_retrieval_map": float(val_stats["retrieval_map"]),
            "val_retrieval_rank1": float(val_stats["retrieval_rank1"]),
            "val_detection_loss": float(val_stats["detection_loss"]),
        }
        torch.save(checkpoint, last_path)
        if better:
            torch.save(checkpoint, output_dir / "best.pt")

        print(
            f"epoch {epoch:03d} train={float(train_stats['total_loss']):.4f} "
            f"val_det={float(val_stats['detection_loss']):.4f} "
            f"mAP={float(val_stats['retrieval_map']):.4f} "
            f"R1={float(val_stats['retrieval_rank1']):.4f} "
            f"queries={int(val_stats['valid_queries'])}",
            flush=True,
        )

    print(f"Saved best checkpoint to {output_dir / 'best.pt'}")
    print(f"Saved last checkpoint to {last_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
