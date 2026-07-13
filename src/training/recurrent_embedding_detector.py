#!/usr/bin/env python3
"""Train the controlled detector with a recurrent object-embedding head.

The detector heads (classification + bbox) stay identical to
``src.training.simple_detector``; this entry point adds a ConvGRU-based
embedding head trained FairMOT-style: every ``(sequence, track_id)`` in the
training split is one identity class, supervised with cross-entropy at
positive cells through a training-only linear classifier. Training runs on
ordered clips of consecutive annotated frames with truncated BPTT (hidden
state reset at clip boundaries).

Full training (sized for a 12-16 GB GPU):

    .venv/bin/python -m src.training.recurrent_embedding_detector \\
      --root data/datasets/dsec_mot \\
      --representation event_frame_voxel_grid \\
      --fusion-mode gated_two_branch \\
      --num-bins 3 \\
      --time-window-us 50000 \\
      --embedding-dim 128 \\
      --clip-length 8 \\
      --epochs 30 \\
      --batch-size 2 \\
      --grad-accum-steps 4 \\
      --model-width 32 \\
      --device cuda \\
      --output-dir runs/recurrent_embedding

Non-recurrent ablation: add ``--no-recurrent-embedding``.

CPU smoke test:

    .venv/bin/python -m src.training.recurrent_embedding_detector \\
      --epochs 1 --clip-length 8 --max-train-clips 2 --max-val-clips 2 \\
      --model-width 8 --device cpu --no-amp --output-dir /tmp/rec_embed_smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

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
    """Cross-entropy over per-cell embeddings at cells with a known identity."""

    known = identity_targets != IDENTITY_IGNORE_INDEX
    if not known.any():
        return embeddings.sum() * 0.0, 0
    vectors = embeddings.permute(0, 2, 3, 1)[known]
    logits = classifier(vectors)
    return F.cross_entropy(logits, identity_targets[known]), int(known.sum())


def run_clip_epoch(
    model: SimpleDenseDetector,
    classifier: nn.Linear,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    background_weight: float,
    bbox_weight: float,
    embedding_loss_weight: float,
    grad_clip_norm: float,
    grad_accum_steps: int,
    log_every: int,
    epoch: int,
    phase: str,
    component_splits: tuple[int, ...] = (),
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    classifier.train(is_train)
    totals: defaultdict[str, float] = defaultdict(float)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and is_train)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        events = batch["events"].to(device, non_blocking=True)
        cls_targets = batch["cls_targets"].to(device, non_blocking=True)
        bbox_targets = batch["bbox_targets"].to(device, non_blocking=True)
        pos_mask = batch["pos_mask"].to(device, non_blocking=True)
        identity_targets = batch["identity_targets"].to(device, non_blocking=True)
        clip_length = events.shape[1]

        with torch.autocast(device_type=device.type, enabled=use_amp):
            embedding_state = None
            det_loss_sum = events.new_zeros(())
            id_loss_sum = events.new_zeros(())
            positive_cells = 0
            identity_cells = 0
            for t in range(clip_length):
                frame = normalise_representation_tensor(events[:, t], component_splits)
                outputs = model(frame, embedding_state)
                embedding_state = outputs.get("embedding_state")
                det_loss, det_stats = simple_detector_loss(
                    outputs=outputs,
                    cls_targets=cls_targets[:, t],
                    bbox_targets=bbox_targets[:, t],
                    pos_mask=pos_mask[:, t],
                    background_weight=background_weight,
                    bbox_weight=bbox_weight,
                )
                det_loss_sum = det_loss_sum + det_loss
                positive_cells += det_stats["positive_cells"]
                if "embeddings" in outputs:
                    id_loss, known_cells = identity_loss(
                        outputs["embeddings"], identity_targets[:, t], classifier
                    )
                    id_loss_sum = id_loss_sum + id_loss
                    identity_cells += known_cells
            det_loss_mean = det_loss_sum / clip_length
            id_loss_mean = id_loss_sum / clip_length
            loss = det_loss_mean + embedding_loss_weight * id_loss_mean

        if optimizer is not None:
            scaler.scale(loss / grad_accum_steps).backward()
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
        stats = {
            "loss": float(loss.detach().cpu()),
            "det_loss": float(det_loss_mean.detach().cpu()),
            "id_loss": float(id_loss_mean.detach().cpu()),
            "positive_cells": float(positive_cells),
            "identity_cells": float(identity_cells),
        }
        for key, value in stats.items():
            totals[key] += value * clips
        totals["clips"] += clips

        should_log = log_every > 0 and (step == 1 or step % log_every == 0 or step == len(loader))
        if should_log:
            elapsed = time.perf_counter() - started
            seen = int(totals["clips"])
            rate = seen / elapsed if elapsed > 0 else 0.0
            print(
                f"epoch {epoch:03d} {phase} step {step:05d}/{len(loader):05d} "
                f"clips={seen}/{len(loader.dataset)} loss={stats['loss']:.4f} "
                f"det={stats['det_loss']:.4f} id={stats['id_loss']:.4f} "
                f"rate={rate:.2f} clips/s",
                flush=True,
            )

    clips = max(totals.pop("clips"), 1.0)
    epoch_stats = {key: value / clips for key, value in totals.items()}
    epoch_stats["clips"] = float(clips)
    return epoch_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--train-sequences", default=None, help="Comma-separated list. Defaults to train minus val."
    )
    parser.add_argument(
        "--val-sequences",
        default=None,
        help=f"Comma-separated list. Defaults to {DEFAULT_VAL_SEQUENCE} when available.",
    )
    parser.add_argument(
        "--representation", choices=REPRESENTATION_CHOICES, default="event_frame_voxel_grid"
    )
    parser.add_argument("--num-bins", type=int, default=3)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument("--eros-cache-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--feature-stride", type=int, default=8)
    parser.add_argument("--positive-radius", type=int, default=1)
    parser.add_argument(
        "--class-ids",
        default=None,
        help="Comma-separated original DSEC-MOT class ids to keep. Use 0 for car-only.",
    )
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument(
        "--fusion-mode",
        choices=("single", "two_branch", "three_branch", "gated_two_branch"),
        default="gated_two_branch",
    )
    parser.add_argument(
        "--architecture",
        choices=("simple", "csp_pan"),
        default="simple",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument(
        "--embedding-hidden-dim",
        type=int,
        default=0,
        help="ConvGRU hidden channels. 0 uses --embedding-dim.",
    )
    parser.add_argument(
        "--no-recurrent-embedding",
        action="store_true",
        help="Drop the ConvGRU so the embedding is computed per frame (ablation).",
    )
    parser.add_argument("--embedding-loss-weight", type=float, default=1.0)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=0,
        help="Stride between clip starts. 0 uses --clip-length (non-overlapping).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2, help="Clips per batch.")
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print batch progress every N steps. Set 0 to disable.",
    )
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
        "--resume",
        action="store_true",
        help="Resume from last.pt in the run directory when it exists.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/recurrent_embedding"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.feature_stride != 8:
        raise SystemExit("The simple detector currently supports --feature-stride 8 only.")
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad-accum-steps must be positive.")
    if args.embedding_dim <= 0:
        raise SystemExit("--embedding-dim must be positive for embedding training.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")

    component_splits = representation_channel_splits(args.representation, args.num_bins)
    class_ids = parse_int_list(args.class_ids)
    if args.num_classes <= 0:
        raise SystemExit("--num-classes must be positive.")
    if class_ids is not None and args.num_classes != len(class_ids):
        raise SystemExit(
            "--num-classes should match the number of selected --class-ids for filtered training."
        )
    expected_branches = {"two_branch": 2, "three_branch": 3, "gated_two_branch": 2}.get(
        args.fusion_mode
    )
    if expected_branches is not None and len(component_splits) != expected_branches:
        raise SystemExit(
            f"--fusion-mode {args.fusion_mode} requires a representation with "
            f"{expected_branches} components, got {args.representation}."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    embedding_recurrent = not args.no_recurrent_embedding
    run_name = f"{args.representation}_bins{args.num_bins}_w{args.model_width}"
    if args.architecture != "simple":
        run_name = f"{run_name}_{args.architecture}"
    if args.fusion_mode != "single":
        run_name = f"{run_name}_{args.fusion_mode}"
    run_name = f"{run_name}_recurrent_embed" if embedding_recurrent else f"{run_name}_embed"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sequences, val_sequences = choose_train_val_sequences(
        root=args.root,
        train_split=args.train_split,
        train_sequences_arg=args.train_sequences,
        val_sequences_arg=args.val_sequences,
    )
    clip_stride = args.clip_stride if args.clip_stride > 0 else args.clip_length
    train_dataset = DSECClipDataset(
        root=args.root,
        split=args.train_split,
        sequences=train_sequences,
        representation=args.representation,
        num_bins=args.num_bins,
        time_window_us=args.time_window_us,
        clip_length=args.clip_length,
        clip_stride=clip_stride,
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
        clip_stride=clip_stride,
        feature_stride=args.feature_stride,
        positive_radius=args.positive_radius,
        eros_cache_root=args.eros_cache_root,
        class_ids=class_ids,
        identity_vocabulary=train_dataset.identity_vocabulary,
    )
    num_identities = train_dataset.num_identities
    if num_identities == 0:
        raise SystemExit("No identities found in the training split.")

    train_dataset_limited = LimitedDataset(train_dataset, args.max_train_clips, args.seed)
    val_dataset_limited = LimitedDataset(val_dataset, args.max_val_clips, args.seed)
    if len(train_dataset_limited) == 0 or len(val_dataset_limited) == 0:
        raise SystemExit(
            f"Empty dataset: train={len(train_dataset_limited)} val={len(val_dataset_limited)} "
            "clips. Check selected sequences and --clip-length."
        )

    use_amp = (not args.no_amp) and device.type == "cuda"
    model_config = SimpleDetectorConfig(
        in_channels=representation_channels(args.representation, args.num_bins),
        num_classes=args.num_classes,
        feature_stride=args.feature_stride,
        width=args.model_width,
        architecture=args.architecture,
        fusion_mode=args.fusion_mode,
        event_frame_channels=2,
        voxel_grid_channels=2 * args.num_bins
        if args.representation == "event_frame_voxel_grid"
        else 0,
        component_channels=component_splits
        if args.fusion_mode in {"gated_two_branch", "three_branch"}
        or ("eros" in args.representation and args.fusion_mode != "single")
        else (),
        embedding_dim=args.embedding_dim,
        embedding_recurrent=embedding_recurrent,
        embedding_hidden_dim=args.embedding_hidden_dim,
    )
    model = SimpleDenseDetector(model_config).to(device)
    classifier = nn.Linear(args.embedding_dim, num_identities).to(device)

    parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_loader = DataLoader(
        train_dataset_limited,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_clip_batch,
    )
    val_loader = DataLoader(
        val_dataset_limited,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_clip_batch,
    )

    config = {
        "root": str(args.root),
        "train_split": args.train_split,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "representation": args.representation,
        "num_bins": args.num_bins,
        "time_window_us": args.time_window_us,
        "eros_cache_root": str(args.eros_cache_root),
        "feature_stride": args.feature_stride,
        "positive_radius": args.positive_radius,
        "class_ids": class_ids,
        "num_classes": args.num_classes,
        "model": model_config.to_dict(),
        "architecture": args.architecture,
        "fusion_mode": args.fusion_mode,
        "embedding_dim": args.embedding_dim,
        "embedding_hidden_dim": args.embedding_hidden_dim,
        "embedding_recurrent": embedding_recurrent,
        "embedding_loss_weight": args.embedding_loss_weight,
        "num_identities": num_identities,
        "clip_length": args.clip_length,
        "clip_stride": clip_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "background_weight": args.background_weight,
        "bbox_weight": args.bbox_weight,
        "max_train_clips": args.max_train_clips,
        "max_val_clips": args.max_val_clips,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_val = float("inf")
    start_epoch = 1
    last_checkpoint_path = output_dir / "last.pt"
    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        classifier.load_state_dict(checkpoint["identity_classifier_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        else:
            history = checkpoint.get("history", [])
        start_epoch = int(checkpoint.get("epoch", len(history))) + 1
        if "best_val" in checkpoint:
            best_val = float(checkpoint["best_val"])
        elif history:
            best_val = min(float(row["val"]["loss"]) for row in history)
        print(f"Resuming {run_name} from epoch {start_epoch}", flush=True)

    if start_epoch > args.epochs:
        print(f"Training {run_name} is already complete ({args.epochs} epochs).", flush=True)
        return 0

    print(f"Training {run_name}", flush=True)
    print(
        f"Train sequences: {train_sequences} ({len(train_dataset_limited)} clips, "
        f"{num_identities} identities)",
        flush=True,
    )
    print(f"Val sequences: {val_sequences} ({len(val_dataset_limited)} clips)", flush=True)
    print(f"Device: {device}, AMP: {use_amp}", flush=True)

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
            embedding_loss_weight=args.embedding_loss_weight,
            grad_clip_norm=args.grad_clip_norm,
            grad_accum_steps=args.grad_accum_steps,
            log_every=args.log_every,
            epoch=epoch,
            phase="train",
            component_splits=component_splits,
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
                embedding_loss_weight=args.embedding_loss_weight,
                grad_clip_norm=0.0,
                grad_accum_steps=1,
                log_every=args.log_every,
                epoch=epoch,
                phase="val",
                component_splits=component_splits,
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
        print(
            f"epoch {epoch:03d} "
            f"train loss={train_stats['loss']:.4f} "
            f"det={train_stats['det_loss']:.4f} "
            f"id={train_stats['id_loss']:.4f} "
            f"val loss={val_stats['loss']:.4f} "
            f"det={val_stats['det_loss']:.4f}",
            flush=True,
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "benchmark_config": config,
            "identity_classifier_state": classifier.state_dict(),
            "num_identities": num_identities,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_val": best_val,
            "epoch": epoch,
            "val_loss": val_stats["loss"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"Saved best checkpoint to {output_dir / 'best.pt'}")
    print(f"Saved last checkpoint to {output_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
