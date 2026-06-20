#!/usr/bin/env python3
"""Train the lightweight controlled detector on DSEC-MOT event representations."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.pretrained_detection_dataset import DSECDenseRepresentationDataset
from src.data.representations import (
    REPRESENTATION_CHOICES,
    BenchmarkRepresentation,
    representation_channel_splits,
)
from src.models.simple_detector import (
    SimpleDenseDetector,
    SimpleDetectorConfig,
    normalise_event_tensor,
    simple_detector_loss,
)

DEFAULT_VAL_SEQUENCE = "zurich_city_01_d"


def parse_sequence_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def discover_sequences(root: Path, split: str) -> list[str]:
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def choose_train_val_sequences(
    root: Path,
    train_split: str,
    train_sequences_arg: str | None,
    val_sequences_arg: str | None,
) -> tuple[list[str] | None, list[str]]:
    all_sequences = discover_sequences(root, train_split)
    requested_train = parse_sequence_list(train_sequences_arg)
    requested_val = parse_sequence_list(val_sequences_arg)
    if requested_val is not None:
        val_sequences = requested_val
    elif DEFAULT_VAL_SEQUENCE in all_sequences:
        val_sequences = [DEFAULT_VAL_SEQUENCE]
    elif all_sequences:
        val_sequences = [all_sequences[-1]]
    else:
        raise SystemExit(f"No sequences found under {root / train_split}")

    if requested_train is not None:
        train_sequences = requested_train
    else:
        train_sequences = [
            sequence for sequence in all_sequences if sequence not in set(val_sequences)
        ]

    if not train_sequences:
        raise SystemExit("No training sequences selected after reserving validation sequences.")
    return train_sequences, val_sequences


class LimitedDataset(Dataset):
    def __init__(self, dataset: Dataset, max_samples: int = 0, seed: int = 0) -> None:
        self.dataset = dataset
        if max_samples > 0 and max_samples < len(dataset):
            rng = random.Random(seed)
            self.indices = sorted(rng.sample(range(len(dataset)), max_samples))
        else:
            self.indices = list(range(len(dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def collate_batch(samples: list[dict]) -> dict[str, object]:
    events = torch.stack([torch.from_numpy(sample["events"]).float() for sample in samples])
    cls_targets = torch.stack([torch.from_numpy(sample["label"][0]).long() for sample in samples])
    bbox_targets = torch.stack([torch.from_numpy(sample["label"][1]).float() for sample in samples])
    pos_mask = torch.stack([torch.from_numpy(sample["label"][2]).bool() for sample in samples])
    return {
        "events": events,
        "cls_targets": cls_targets,
        "bbox_targets": bbox_targets,
        "pos_mask": pos_mask,
        "meta": [sample["meta"] for sample in samples],
    }


def make_dataset(
    root: Path,
    split: str,
    sequences: list[str],
    representation: str,
    num_bins: int,
    time_window_us: int,
    feature_stride: int,
    positive_radius: int,
    include_unannotated: bool,
    eros_cache_root: Path,
    class_ids: list[int] | None,
) -> DSECDenseRepresentationDataset:
    return DSECDenseRepresentationDataset(
        root=root,
        split=split,
        sequences=sequences,
        representation=representation,
        num_bins=num_bins,
        time_window_us=time_window_us,
        feature_stride=feature_stride,
        positive_radius=positive_radius,
        include_unannotated=include_unannotated,
        eros_cache_root=eros_cache_root,
        class_ids=class_ids,
    )


def run_epoch(
    model: SimpleDenseDetector,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    background_weight: float,
    bbox_weight: float,
    grad_clip_norm: float,
    grad_accum_steps: int,
    log_every: int,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: defaultdict[str, float] = defaultdict(float)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and is_train)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        events = normalise_event_tensor(batch["events"].to(device, non_blocking=True))
        cls_targets = batch["cls_targets"].to(device, non_blocking=True)
        bbox_targets = batch["bbox_targets"].to(device, non_blocking=True)
        pos_mask = batch["pos_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(events)
            loss, stats = simple_detector_loss(
                outputs=outputs,
                cls_targets=cls_targets,
                bbox_targets=bbox_targets,
                pos_mask=pos_mask,
                background_weight=background_weight,
                bbox_weight=bbox_weight,
            )

        if optimizer is not None:
            scaler.scale(loss / grad_accum_steps).backward()
            should_step = step % grad_accum_steps == 0 or step == len(loader)
            if should_step:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        batch_size = int(events.shape[0])
        for key, value in stats.items():
            totals[key] += float(value) * batch_size
        totals["samples"] += batch_size

        if log_every > 0 and (step == 1 or step % log_every == 0 or step == len(loader)):
            elapsed = time.perf_counter() - started
            seen = int(totals["samples"])
            rate = seen / elapsed if elapsed > 0 else 0.0
            print(
                f"epoch {epoch:03d} {phase} step {step:05d}/{len(loader):05d} "
                f"samples={seen}/{len(loader.dataset)} loss={stats['loss']:.4f} "
                f"rate={rate:.2f} samples/s",
                flush=True,
            )

    samples = max(totals.pop("samples"), 1.0)
    return {key: value / samples for key, value in totals.items()}


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
    parser.add_argument("--representation", choices=REPRESENTATION_CHOICES, default="voxel_grid")
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument("--eros-cache-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--feature-stride", type=int, default=8)
    parser.add_argument("--positive-radius", type=int, default=1)
    parser.add_argument("--include-unannotated", action="store_true")
    parser.add_argument(
        "--class-ids",
        default=None,
        help="Comma-separated original DSEC-MOT class ids to keep. Use 0 for car-only.",
    )
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print batch progress every N steps. Set 0 to disable.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument(
        "--fusion-mode",
        choices=("single", "two_branch", "three_branch", "gated_two_branch"),
        default="single",
        help=(
            "Input fusion adapter. 'single' keeps the original one-stream detector. "
            "Multi-branch modes independently process each representation component before fusion."
        ),
    )
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--bbox-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simple_detector"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.feature_stride != 8:
        raise SystemExit("The simple detector currently supports --feature-stride 8 only.")
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad-accum-steps must be positive.")
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
    if args.fusion_mode == "single" and len(component_splits) > 1 and "eros" in args.representation:
        raise SystemExit(
            "EROS fusion variants in this benchmark must use two_branch or three_branch."
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_sequences, val_sequences = choose_train_val_sequences(
        root=args.root,
        train_split=args.train_split,
        train_sequences_arg=args.train_sequences,
        val_sequences_arg=args.val_sequences,
    )
    representation = BenchmarkRepresentation(args.representation, args.num_bins)
    run_name = f"{args.representation}_bins{args.num_bins}_w{args.model_width}"
    if args.fusion_mode != "single":
        run_name = f"{run_name}_{args.fusion_mode}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(
        root=args.root,
        split=args.train_split,
        sequences=train_sequences,
        representation=args.representation,
        num_bins=args.num_bins,
        time_window_us=args.time_window_us,
        feature_stride=args.feature_stride,
        positive_radius=args.positive_radius,
        include_unannotated=args.include_unannotated,
        eros_cache_root=args.eros_cache_root,
        class_ids=class_ids,
    )
    val_dataset = make_dataset(
        root=args.root,
        split=args.train_split,
        sequences=val_sequences,
        representation=args.representation,
        num_bins=args.num_bins,
        time_window_us=args.time_window_us,
        feature_stride=args.feature_stride,
        positive_radius=args.positive_radius,
        include_unannotated=args.include_unannotated,
        eros_cache_root=args.eros_cache_root,
        class_ids=class_ids,
    )
    train_dataset_limited = LimitedDataset(train_dataset, args.max_train_samples, args.seed)
    val_dataset_limited = LimitedDataset(val_dataset, args.max_val_samples, args.seed)
    if len(train_dataset_limited) == 0 or len(val_dataset_limited) == 0:
        raise SystemExit(
            f"Empty dataset: train={len(train_dataset_limited)} val={len(val_dataset_limited)}. "
            "Check selected sequences and extracted DSEC-MOT files."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
    use_amp = (not args.no_amp) and device.type == "cuda"

    model_config = SimpleDetectorConfig(
        in_channels=representation.channels,
        num_classes=args.num_classes,
        feature_stride=args.feature_stride,
        width=args.model_width,
        fusion_mode=args.fusion_mode,
        event_frame_channels=2,
        voxel_grid_channels=2 * args.num_bins
        if args.representation == "event_frame_voxel_grid"
        else 0,
        component_channels=component_splits
        if args.fusion_mode in {"gated_two_branch", "three_branch"}
        or ("eros" in args.representation and args.fusion_mode != "single")
        else (),
    )
    model = SimpleDenseDetector(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_loader = DataLoader(
        train_dataset_limited,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset_limited,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
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
        "include_unannotated": args.include_unannotated,
        "class_ids": class_ids,
        "num_classes": args.num_classes,
        "model": model_config.to_dict(),
        "fusion_mode": args.fusion_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "background_weight": args.background_weight,
        "bbox_weight": args.bbox_weight,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_val = float("inf")
    print(f"Training {run_name}")
    print(f"Train sequences: {train_sequences} ({len(train_dataset_limited)} samples)")
    print(f"Val sequences: {val_sequences} ({len(val_dataset_limited)} samples)")
    print(f"Device: {device}, AMP: {use_amp}")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            use_amp=use_amp,
            background_weight=args.background_weight,
            bbox_weight=args.bbox_weight,
            grad_clip_norm=args.grad_clip_norm,
            grad_accum_steps=args.grad_accum_steps,
            log_every=args.log_every,
            epoch=epoch,
            phase="train",
        )
        with torch.inference_mode():
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                use_amp=use_amp,
                background_weight=args.background_weight,
                bbox_weight=args.bbox_weight,
                grad_clip_norm=0.0,
                grad_accum_steps=1,
                log_every=args.log_every,
                epoch=epoch,
                phase="val",
            )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch {epoch:03d} "
            f"train loss={train_stats['loss']:.4f} "
            f"cls={train_stats['cls_loss']:.4f} "
            f"bbox={train_stats['bbox_loss']:.4f} "
            f"val loss={val_stats['loss']:.4f} "
            f"cls={val_stats['cls_loss']:.4f} "
            f"bbox={val_stats['bbox_loss']:.4f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "benchmark_config": config,
            "epoch": epoch,
            "val_loss": val_stats["loss"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"Saved best checkpoint to {output_dir / 'best.pt'}")
    print(f"Saved last checkpoint to {output_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
