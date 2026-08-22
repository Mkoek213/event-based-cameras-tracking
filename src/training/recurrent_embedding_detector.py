#!/usr/bin/env python3
"""Train RoI or dense event-camera association heads on ordered DSEC-MOT clips."""

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
    TEMPORAL_RECURRENCE_LOCATIONS,
    TEMPORAL_RECURRENCE_MODES,
    TEMPORAL_RECURRENCE_TYPES,
    SimpleDenseDetector,
    SimpleDetectorConfig,
    normalise_representation_tensor,
    simple_detector_loss,
)
from src.training.simple_detector import (
    DEFAULT_VAL_SEQUENCE,
    LimitedDataset,
    choose_train_val_sequences,
    load_compatible_state,
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


def detach_recurrent_state(state: object) -> object:
    """Detach tensor, tuple, or dictionary recurrent state for truncated BPTT."""

    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(detach_recurrent_state(item) for item in state)
    if isinstance(state, dict):
        return {key: detach_recurrent_state(value) for key, value in state.items()}
    if state is None:
        return None
    raise TypeError(f"Unsupported recurrent state type: {type(state).__name__}.")


def load_compatible_embedding_head(
    model: SimpleDenseDetector,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, int]:
    """Load only shape-compatible association-head tensors from a checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint.get("model_state", checkpoint)
    source_state = {
        key.removeprefix("module."): value
        for key, value in source_state.items()
        if key.removeprefix("module.").startswith("embedding_")
    }
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(source_state) - len(compatible)


def run_clip_epoch(
    model: SimpleDenseDetector,
    classifier: nn.Linear | None,
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
    carry_state_between_clips: bool = False,
    burn_in_frames: int = 0,
) -> dict[str, float | int]:
    """Run one epoch with reset or detached state carried between ordered clips."""

    is_train = optimizer is not None
    has_embedding_head = model.config.embedding_dim > 0
    if compute_retrieval is None:
        compute_retrieval = not is_train
    compute_retrieval = bool(compute_retrieval and has_embedding_head)
    model.train(is_train)
    if classifier is not None:
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
    persistent_embedding_state: object = None
    persistent_temporal_state: object = None
    previous_sequence: str | None = None
    previous_last_frame: int | None = None

    for step, batch in enumerate(loader, start=1):
        events = batch["events"].to(device, non_blocking=True)
        cls_targets = batch["cls_targets"].to(device, non_blocking=True)
        bbox_targets = batch["bbox_targets"].to(device, non_blocking=True)
        pos_mask = batch["pos_mask"].to(device, non_blocking=True)
        identity_targets = batch["identity_targets"].to(device, non_blocking=True)
        clip_length = events.shape[1]
        batch_size = events.shape[0]
        if carry_state_between_clips and batch_size != 1:
            raise ValueError("State carry requires clip batch size 1.")

        first_meta = batch["meta"][0][0]
        sequence = str(first_meta["sequence"])
        first_frame = int(first_meta["frame_index"])
        contiguous = bool(
            carry_state_between_clips
            and previous_sequence == sequence
            and previous_last_frame is not None
            and first_frame == previous_last_frame + 1
        )
        state_was_reset = not contiguous
        supervised_start = burn_in_frames if state_was_reset else 0
        if supervised_start >= clip_length:
            raise ValueError("burn_in_frames must be smaller than the clip length.")

        with torch.autocast(device_type=device.type, enabled=use_amp):
            embedding_state = persistent_embedding_state if contiguous else None
            temporal_state = persistent_temporal_state if contiguous else None
            detection_loss_sum = events.new_zeros(())
            positive_cells = 0
            training_descriptors_by_frame: list[torch.Tensor] = []
            identities_by_frame: list[torch.Tensor] = []
            classes_by_frame: list[torch.Tensor] = []
            dense_retrieval_by_frame: list[torch.Tensor] = []
            raw_tracks_by_frame: list[torch.Tensor] = []
            retrieval_classes_by_frame: list[torch.Tensor] = []
            sequences_by_frame: list[str] = []

            for time_index in range(clip_length):
                frame = normalise_representation_tensor(events[:, time_index], component_splits)
                outputs = model(
                    frame,
                    embedding_state=embedding_state,
                    temporal_state=temporal_state,
                )
                embedding_state = outputs.get("embedding_state")
                temporal_state = outputs.get("temporal_state")
                if time_index < supervised_start:
                    continue
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

                if not has_embedding_head:
                    continue
                boxes = _frame_tensors(batch["roi_boxes"], time_index, device)
                roi_count = sum(int(box.shape[0]) for box in boxes)
                if model.config.embedding_head_type == "dense":
                    embedding_map = outputs.get("embeddings")
                    if not isinstance(embedding_map, torch.Tensor):
                        raise RuntimeError("Dense embedding training requires an embedding map.")
                    frame_identities = identity_targets[:, time_index]
                    known = frame_identities != IDENTITY_IGNORE_INDEX
                    descriptors = embedding_map.permute(0, 2, 3, 1)[known]
                    training_descriptors_by_frame.append(descriptors)
                    identities_by_frame.append(frame_identities[known])
                    classes_by_frame.append(cls_targets[:, time_index][known] - 1)
                    dense_retrieval_by_frame.append(
                        model.extract_dense_embeddings_at_boxes(embedding_map, boxes)
                    )
                else:
                    feature_map = outputs.get("embedding_feature_map")
                    if not isinstance(feature_map, torch.Tensor):
                        raise RuntimeError("RoI embedding training requires a feature map.")
                    training_descriptors_by_frame.append(
                        model.project_roi_embeddings(feature_map, boxes)
                    )
                    identities_by_frame.append(
                        torch.cat(_frame_tensors(batch["roi_identity_targets"], time_index, device))
                        if roi_count
                        else _empty_long(device)
                    )
                    classes_by_frame.append(
                        torch.cat(_frame_tensors(batch["roi_class_ids"], time_index, device))
                        if roi_count
                        else _empty_long(device)
                    )

                raw_tracks_by_frame.append(
                    torch.cat(_frame_tensors(batch["roi_track_ids"], time_index, device))
                    if roi_count
                    else _empty_long(device)
                )
                retrieval_classes_by_frame.append(
                    torch.cat(_frame_tensors(batch["roi_class_ids"], time_index, device))
                    if roi_count
                    else _empty_long(device)
                )
                for batch_index in range(batch_size):
                    count = int(boxes[batch_index].shape[0])
                    sequence = str(batch["meta"][batch_index][time_index]["sequence"])
                    sequences_by_frame.extend([sequence] * count)

            supervised_frames = clip_length - supervised_start
            detection_loss = detection_loss_sum / supervised_frames
            if has_embedding_head:
                training_descriptors = torch.cat(training_descriptors_by_frame, dim=0)
                identities = torch.cat(identities_by_frame, dim=0)
                classes = torch.cat(classes_by_frame, dim=0)
                raw_tracks = torch.cat(raw_tracks_by_frame, dim=0)
                retrieval_classes_batch = torch.cat(retrieval_classes_by_frame, dim=0)
                if model.config.embedding_head_type == "dense":
                    post_neck_descriptors = training_descriptors
                    triplet_descriptors = training_descriptors
                    retrieval_descriptors = torch.cat(dense_retrieval_by_frame, dim=0)
                else:
                    post_neck_descriptors = model.apply_embedding_bn(training_descriptors)
                    triplet_descriptors = training_descriptors
                    retrieval_descriptors = F.normalize(post_neck_descriptors, dim=1)

                if classifier is None:
                    raise RuntimeError("Embedding training requires an identity classifier.")
                identity_ce, identity_objects = identity_loss(
                    post_neck_descriptors, identities, classifier
                )
                triplet, valid_triplet_anchors = batch_hard_cosine_triplet_loss(
                    triplet_descriptors,
                    identities,
                    classes,
                    margin=triplet_margin,
                )
            else:
                identity_ce = detection_loss * 0.0
                triplet = detection_loss * 0.0
                identity_objects = 0
                valid_triplet_anchors = 0
            total_loss = (
                detection_loss + identity_ce_weight * identity_ce + triplet_weight * triplet
            )

        if optimizer is not None:
            scaler.scale(total_loss / grad_accum_steps).backward()
            should_step = step % grad_accum_steps == 0 or step == len(loader)
            if should_step:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    parameters = [
                        parameter
                        for parameter in list(model.parameters())
                        + (list(classifier.parameters()) if classifier is not None else [])
                        if parameter.requires_grad
                    ]
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if carry_state_between_clips:
            persistent_embedding_state = detach_recurrent_state(embedding_state)
            persistent_temporal_state = detach_recurrent_state(temporal_state)
            previous_sequence = sequence
            previous_last_frame = int(batch["meta"][0][-1]["frame_index"])
        else:
            persistent_embedding_state = None
            persistent_temporal_state = None
            previous_sequence = None
            previous_last_frame = None

        clips = int(events.shape[0])
        totals["total_loss"] += float(total_loss.detach()) * clips
        totals["detection_loss"] += float(detection_loss.detach()) * clips
        totals["identity_loss"] += float(identity_ce.detach()) * clips
        totals["triplet_loss"] += float(triplet.detach()) * clips
        totals["clips"] += clips
        totals["positive_cells"] += positive_cells
        totals["identity_objects"] += identity_objects
        totals["valid_triplet_anchors"] += valid_triplet_anchors
        totals["microbatches"] += 1
        totals["identity_active_batches"] += int(identity_objects > 0)
        totals["triplet_active_batches"] += int(valid_triplet_anchors > 0)
        totals["supervised_frames"] += supervised_frames * clips
        totals["state_resets"] += int(state_was_reset)
        totals["state_carries"] += int(contiguous)

        if compute_retrieval:
            retrieval_embeddings.append(retrieval_descriptors.detach().float().cpu())
            retrieval_classes.append(retrieval_classes_batch.detach().cpu())
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
        "microbatches": int(totals["microbatches"]),
        "identity_active_batches": int(totals["identity_active_batches"]),
        "triplet_active_batches": int(totals["triplet_active_batches"]),
        "supervised_frames": int(totals["supervised_frames"]),
        "state_resets": int(totals["state_resets"]),
        "state_carries": int(totals["state_carries"]),
    }
    stats["identity_active_batch_fraction"] = totals["identity_active_batches"] / max(
        totals["microbatches"], 1.0
    )
    stats["triplet_active_batch_fraction"] = totals["triplet_active_batches"] / max(
        totals["microbatches"], 1.0
    )
    stats["triplet_anchor_fraction"] = totals["valid_triplet_anchors"] / max(
        totals["identity_objects"], 1.0
    )
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
    parser.add_argument(
        "--embedding-head-type",
        choices=("roi", "dense"),
        default="roi",
        help="Use object-level RoIAlign descriptors or FairMOT-style dense grid descriptors.",
    )
    parser.add_argument("--roi-size", type=int, default=7)
    recurrent = parser.add_mutually_exclusive_group()
    recurrent.add_argument(
        "--recurrent-embedding",
        dest="embedding_recurrent",
        action="store_true",
        help="Enable legacy R2 spatial ConvGRU recurrence inside the embedding head.",
    )
    recurrent.add_argument(
        "--no-recurrent-embedding",
        dest="embedding_recurrent",
        action="store_false",
        help="Disable the legacy embedding-only ConvGRU.",
    )
    parser.set_defaults(embedding_recurrent=True)
    parser.add_argument(
        "--temporal-recurrence-locations",
        default="",
        help=("Comma-separated subset of: " + ",".join(TEMPORAL_RECURRENCE_LOCATIONS) + "."),
    )
    parser.add_argument(
        "--temporal-recurrence-type",
        choices=TEMPORAL_RECURRENCE_TYPES,
        default="convgru",
    )
    parser.add_argument(
        "--temporal-recurrence-mode",
        choices=TEMPORAL_RECURRENCE_MODES,
        default="residual",
    )
    parser.add_argument("--identity-ce-weight", type=float, default=1.0)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument(
        "--ordered-clips",
        action="store_true",
        help="Keep training clips in dataset order instead of shuffling them.",
    )
    parser.add_argument(
        "--carry-state-between-clips",
        action="store_true",
        help=(
            "Carry detached recurrent state across contiguous, non-overlapping clips. "
            "Requires ordered clips, batch size 1, and clip stride equal to clip length."
        ),
    )
    parser.add_argument(
        "--burn-in-frames",
        type=int,
        default=0,
        help="Unsupervised prefix used only after a recurrent state reset.",
    )
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
    parser.add_argument(
        "--initial-model-checkpoint",
        type=Path,
        default=None,
        help="Load every shape-compatible model tensor before adding temporal adapters.",
    )
    parser.add_argument(
        "--initial-detector-checkpoint",
        type=Path,
        default=None,
        help="Load compatible detector weights before training the embedding head.",
    )
    parser.add_argument(
        "--freeze-detector",
        action="store_true",
        help=(
            "Freeze input stems, fusion, backbone, and detection heads, including "
            "their BatchNorm running statistics."
        ),
    )
    parser.add_argument(
        "--train-temporal-adapters",
        action="store_true",
        help="Keep newly inserted recurrent adapters trainable when the detector is frozen.",
    )
    parser.add_argument(
        "--initial-embedding-checkpoint",
        type=Path,
        default=None,
        help="Load only compatible embedding-head tensors before training.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save epoch_NNN.pt checkpoints for external HOTA-based model selection.",
    )
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
    if args.embedding_dim < 0:
        raise SystemExit("Embedding dimension cannot be negative.")
    if args.embedding_dim > 0 and args.embedding_hidden_dim <= 0:
        raise SystemExit("Embedding hidden dimension must be positive.")
    if args.embedding_dim > 0 and args.embedding_head_type == "roi" and args.roi_size <= 0:
        raise SystemExit("RoI size must be positive for the RoI embedding head.")
    if args.clip_length <= 0 or args.clip_stride <= 0:
        raise SystemExit("Clip length and stride must be positive.")
    if args.burn_in_frames < 0 or args.burn_in_frames >= args.clip_length:
        raise SystemExit("--burn-in-frames must be in [0, clip_length).")
    if args.carry_state_between_clips:
        if args.batch_size != 1:
            raise SystemExit("--carry-state-between-clips requires --batch-size 1.")
        if args.clip_stride != args.clip_length:
            raise SystemExit(
                "--carry-state-between-clips requires --clip-stride equal to --clip-length."
            )
        args.ordered_clips = True
    if args.embedding_dim == 0:
        if args.embedding_recurrent:
            raise SystemExit("Embedding recurrence requires --embedding-dim greater than zero.")
        if args.initial_embedding_checkpoint is not None:
            raise SystemExit("An embedding checkpoint cannot be used with --embedding-dim 0.")
        if args.identity_ce_weight != 0.0 or args.triplet_weight != 0.0:
            raise SystemExit(
                "Detector-only runs require --identity-ce-weight 0 and --triplet-weight 0."
            )
    temporal_locations = tuple(
        location.strip()
        for location in args.temporal_recurrence_locations.split(",")
        if location.strip()
    )
    unknown_locations = sorted(set(temporal_locations) - set(TEMPORAL_RECURRENCE_LOCATIONS))
    if unknown_locations:
        raise SystemExit(f"Unknown temporal recurrence locations: {unknown_locations}.")
    if len(set(temporal_locations)) != len(temporal_locations):
        raise SystemExit("Temporal recurrence locations must be unique.")
    if args.initial_model_checkpoint is not None and (
        args.initial_detector_checkpoint is not None
        or args.initial_embedding_checkpoint is not None
    ):
        raise SystemExit(
            "--initial-model-checkpoint cannot be combined with detector/head checkpoints."
        )
    initial_base_checkpoint = args.initial_model_checkpoint or args.initial_detector_checkpoint
    if args.freeze_detector and initial_base_checkpoint is None:
        raise SystemExit(
            "--freeze-detector requires --initial-model-checkpoint "
            "or --initial-detector-checkpoint."
        )
    if args.train_temporal_adapters and not args.freeze_detector:
        raise SystemExit("--train-temporal-adapters requires --freeze-detector.")
    if args.train_temporal_adapters and not temporal_locations:
        raise SystemExit("--train-temporal-adapters requires at least one temporal location.")

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

    if args.embedding_dim == 0:
        variant = "detector_only"
    else:
        family = "dense" if args.embedding_head_type == "dense" else "roi"
        variant_id = "d2" if args.embedding_recurrent else "d1"
        variant = f"{family}_{variant_id}"
        if args.embedding_recurrent:
            variant += "_recurrent"
    if temporal_locations:
        location_label = "-".join(temporal_locations)
        variant += (
            f"_{args.temporal_recurrence_type}_{args.temporal_recurrence_mode}_{location_label}"
        )
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
    if args.embedding_dim > 0 and train_dataset.num_identities == 0:
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
        # Keep legacy module names so existing two-branch R0 checkpoints load exactly.
        component_channels=() if args.fusion_mode == "two_branch" else component_splits,
        embedding_dim=args.embedding_dim,
        embedding_recurrent=args.embedding_recurrent,
        embedding_hidden_dim=args.embedding_hidden_dim,
        embedding_head_type=args.embedding_head_type,
        embedding_roi_size=args.roi_size,
        temporal_recurrence_locations=temporal_locations,
        temporal_recurrence_type=args.temporal_recurrence_type,
        temporal_recurrence_mode=args.temporal_recurrence_mode,
    )
    model = SimpleDenseDetector(model_config).to(device)
    loaded_model_tensors = 0
    skipped_model_tensors = 0
    if args.initial_model_checkpoint is not None:
        loaded_model_tensors, skipped_model_tensors = load_compatible_state(
            model, args.initial_model_checkpoint, device
        )
        if loaded_model_tensors == 0 or skipped_model_tensors != 0:
            raise SystemExit(
                "Initial model checkpoint is not fully compatible with the base model: "
                f"loaded={loaded_model_tensors}, skipped={skipped_model_tensors}."
            )
        print(
            f"Loaded {loaded_model_tensors} base-model tensors from "
            f"{args.initial_model_checkpoint}",
            flush=True,
        )

    loaded_detector_tensors = 0
    skipped_detector_tensors = 0
    if args.initial_detector_checkpoint is not None:
        loaded_detector_tensors, skipped_detector_tensors = load_compatible_state(
            model, args.initial_detector_checkpoint, device
        )
        if loaded_detector_tensors == 0 or skipped_detector_tensors != 0:
            raise SystemExit(
                "Initial detector checkpoint is not fully compatible: "
                f"loaded={loaded_detector_tensors}, skipped={skipped_detector_tensors}."
            )
        print(
            f"Loaded {loaded_detector_tensors} detector tensors from "
            f"{args.initial_detector_checkpoint}",
            flush=True,
        )
    loaded_embedding_tensors = 0
    skipped_embedding_tensors = 0
    if args.initial_embedding_checkpoint is not None:
        loaded_embedding_tensors, skipped_embedding_tensors = load_compatible_embedding_head(
            model, args.initial_embedding_checkpoint, device
        )
        if loaded_embedding_tensors == 0 or skipped_embedding_tensors != 0:
            raise SystemExit(
                "Initial embedding checkpoint is not fully compatible: "
                f"loaded={loaded_embedding_tensors}, skipped={skipped_embedding_tensors}."
            )
        print(
            f"Loaded {loaded_embedding_tensors} embedding-head tensors from "
            f"{args.initial_embedding_checkpoint}",
            flush=True,
        )
    if args.freeze_detector:
        model.set_detector_trainable(False)
    if args.train_temporal_adapters:
        model.set_temporal_trainable(True)

    classifier = (
        nn.Linear(args.embedding_dim, train_dataset.num_identities).to(device)
        if args.embedding_dim > 0
        else None
    )
    classifier_parameters = list(classifier.parameters()) if classifier is not None else []
    trainable_parameters = [
        parameter
        for parameter in list(model.parameters()) + classifier_parameters
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_loader = DataLoader(
        train_limited,
        batch_size=args.batch_size,
        shuffle=not args.ordered_clips,
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
        "embedding_head_type": args.embedding_head_type,
        "roi_size": args.roi_size,
        "embedding_recurrent": args.embedding_recurrent,
        "temporal_recurrence_locations": list(temporal_locations),
        "temporal_recurrence_type": args.temporal_recurrence_type,
        "temporal_recurrence_mode": args.temporal_recurrence_mode,
        "identity_ce_weight": args.identity_ce_weight,
        "triplet_weight": args.triplet_weight,
        "triplet_margin": args.triplet_margin,
        "num_identities": train_dataset.num_identities,
        "clip_length": args.clip_length,
        "clip_stride": args.clip_stride,
        "ordered_clips": args.ordered_clips,
        "carry_state_between_clips": args.carry_state_between_clips,
        "burn_in_frames": args.burn_in_frames,
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
        "initial_model_checkpoint": (
            str(args.initial_model_checkpoint) if args.initial_model_checkpoint else None
        ),
        "initial_detector_checkpoint": (
            str(args.initial_detector_checkpoint) if args.initial_detector_checkpoint else None
        ),
        "initial_embedding_checkpoint": (
            str(args.initial_embedding_checkpoint) if args.initial_embedding_checkpoint else None
        ),
        "detector_frozen": args.freeze_detector,
        "train_temporal_adapters": args.train_temporal_adapters,
        "loaded_model_tensors": loaded_model_tensors,
        "skipped_model_tensors": skipped_model_tensors,
        "loaded_detector_tensors": loaded_detector_tensors,
        "skipped_detector_tensors": skipped_detector_tensors,
        "loaded_embedding_tensors": loaded_embedding_tensors,
        "skipped_embedding_tensors": skipped_embedding_tensors,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "save_every_epoch": args.save_every_epoch,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_selection: dict[str, float | int] | None = None
    start_epoch = 1
    last_path = output_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if classifier is not None:
            classifier_state = checkpoint.get("identity_classifier_state")
            if classifier_state is None:
                raise RuntimeError("Checkpoint is missing the identity classifier state.")
            classifier.load_state_dict(classifier_state)
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
        f"val={val_sequences} ({len(val_limited)} clips), device={device}, AMP={use_amp}, "
        f"detector_frozen={args.freeze_detector}, temporal={temporal_locations or ('none',)}, "
        f"cell={args.temporal_recurrence_type}, mode={args.temporal_recurrence_mode}, "
        f"ordered={args.ordered_clips}, carry={args.carry_state_between_clips}, "
        f"burn_in={args.burn_in_frames}",
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
            carry_state_between_clips=args.carry_state_between_clips,
            burn_in_frames=args.burn_in_frames,
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
                compute_retrieval=args.embedding_dim > 0,
                carry_state_between_clips=args.carry_state_between_clips,
                burn_in_frames=args.burn_in_frames,
            )
        if args.embedding_dim > 0 and int(val_stats.get("valid_queries", 0)) == 0:
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
            "retrieval_map": float(val_stats.get("retrieval_map", 0.0)),
            "retrieval_rank1": float(val_stats.get("retrieval_rank1", 0.0)),
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
            "identity_classifier_state": classifier.state_dict()
            if classifier is not None
            else None,
            "num_identities": train_dataset.num_identities,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_selection": best_selection,
            "epoch": epoch,
            "selected_epoch": int(best_selection["epoch"]),
            "val_retrieval_map": float(val_stats.get("retrieval_map", 0.0)),
            "val_retrieval_rank1": float(val_stats.get("retrieval_rank1", 0.0)),
            "val_detection_loss": float(val_stats["detection_loss"]),
        }
        torch.save(checkpoint, last_path)
        if args.save_every_epoch:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if better:
            torch.save(checkpoint, output_dir / "best.pt")

        print(
            f"epoch {epoch:03d} train={float(train_stats['total_loss']):.4f} "
            f"val_det={float(val_stats['detection_loss']):.4f} "
            f"mAP={float(val_stats.get('retrieval_map', 0.0)):.4f} "
            f"R1={float(val_stats.get('retrieval_rank1', 0.0)):.4f} "
            f"queries={int(val_stats.get('valid_queries', 0))}",
            flush=True,
        )

    print(f"Saved best checkpoint to {output_dir / 'best.pt'}")
    print(f"Saved last checkpoint to {last_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
