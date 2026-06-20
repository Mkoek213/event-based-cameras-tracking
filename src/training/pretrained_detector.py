#!/usr/bin/env python3
"""Fine-tune an RGB-pretrained Faster R-CNN on DSEC-MOT event representations."""

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

from src.data.pretrained_detection_dataset import (
    DSECPretrainedDetectionDataset,
    collate_detection_batch,
)
from src.data.representations import PRETRAINED_REPRESENTATION_CHOICES, representation_components
from src.models.pretrained_detector import PretrainedDetectorConfig, PretrainedEventDetector
from src.training.simple_detector import LimitedDataset, choose_train_val_sequences


def move_batch(images, targets, device: torch.device):
    images = [image.to(device, non_blocking=True) for image in images]
    targets = [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in targets
    ]
    return images, targets


def run_epoch(
    model: PretrainedEventDetector,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    grad_accum_steps: int,
    grad_clip_norm: float,
    epoch: int,
    phase: str,
    log_every: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train()
    model.freeze_batch_norm_stats()
    totals: defaultdict[str, float] = defaultdict(float)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and is_train)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, (images, targets, _) in enumerate(loader, start=1):
            images, targets = move_batch(images, targets, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = model(images, targets)
                loss = sum(losses.values())

            if optimizer is not None:
                scaler.scale(loss / grad_accum_steps).backward()
                if step % grad_accum_steps == 0 or step == len(loader):
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = len(images)
            totals["loss"] += float(loss.detach()) * batch_size
            for key, value in losses.items():
                totals[key] += float(value.detach()) * batch_size
            totals["samples"] += batch_size
            if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
                elapsed = time.perf_counter() - started
                rate = totals["samples"] / elapsed
                print(
                    f"epoch {epoch:03d} {phase} step {step:05d}/{len(loader):05d} "
                    f"samples={int(totals['samples'])}/{len(loader.dataset)} "
                    f"loss={float(loss.detach()):.4f} rate={rate:.2f} samples/s",
                    flush=True,
                )

    samples = max(totals.pop("samples"), 1.0)
    return {key: value / samples for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-sequences", default=None)
    parser.add_argument("--val-sequences", default=None)
    parser.add_argument(
        "--representation", choices=PRETRAINED_REPRESENTATION_CHOICES, required=True
    )
    parser.add_argument("--adapter-mode", choices=("single", "multi_branch"), default="single")
    parser.add_argument("--adapter-width", type=int, default=32)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument("--eros-cache-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--min-size", type=int, default=480)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--weights", choices=("coco", "none"), default="coco")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pretrained_detector"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.adapter_mode == "multi_branch"
        and len(representation_components(args.representation)) < 2
    ):
        raise SystemExit("--adapter-mode multi_branch requires a fused representation.")
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad-accum-steps must be positive.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_sequences, val_sequences = choose_train_val_sequences(
        args.root, args.train_split, args.train_sequences, args.val_sequences
    )
    train_dataset: Dataset = DSECPretrainedDetectionDataset(
        args.root,
        args.train_split,
        train_sequences,
        args.representation,
        args.num_bins,
        args.time_window_us,
        eros_cache_root=args.eros_cache_root,
    )
    val_dataset: Dataset = DSECPretrainedDetectionDataset(
        args.root,
        args.train_split,
        val_sequences,
        args.representation,
        args.num_bins,
        args.time_window_us,
        eros_cache_root=args.eros_cache_root,
    )
    train_dataset = LimitedDataset(train_dataset, args.max_train_samples, args.seed)
    val_dataset = LimitedDataset(val_dataset, args.max_val_samples, args.seed)
    if not len(train_dataset) or not len(val_dataset):
        raise SystemExit(f"Empty dataset: train={len(train_dataset)} val={len(val_dataset)}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    use_amp = not args.no_amp and device.type == "cuda"
    config = PretrainedDetectorConfig(
        representation=args.representation,
        num_bins=args.num_bins,
        adapter_mode=args.adapter_mode,
        adapter_width=args.adapter_width,
        min_size=args.min_size,
        max_size=args.max_size,
        pretrained_weights=args.weights == "coco",
    )
    model = PretrainedEventDetector(config).to(device)
    model.set_backbone_trainable(args.freeze_backbone_epochs <= 0)

    non_backbone = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("detector.backbone.")
    ]
    backbone = list(model.detector.backbone.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": non_backbone, "lr": args.lr},
            {"params": backbone, "lr": args.lr * args.backbone_lr_multiplier},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection_batch,
    )

    run_name = f"{args.representation}_{args.adapter_mode}_fasterrcnn_r50_fpn_v2"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_config = {
        **vars(args),
        "root": str(args.root),
        "eros_cache_root": str(args.eros_cache_root),
        "output_dir": str(args.output_dir),
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "model": config.to_dict(),
        "parameter_counts": model.parameter_counts,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
    }
    (output_dir / "config.json").write_text(
        json.dumps(benchmark_config, indent=2, default=str), encoding="utf-8"
    )

    print(f"Training {run_name}")
    print(f"Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")
    print(f"Parameters: {model.parameter_counts}")
    print(f"Device: {device}, AMP: {use_amp}")
    history: list[dict] = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            model.set_backbone_trainable(True)
            print("Unfroze pretrained backbone.")
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            use_amp,
            args.grad_accum_steps,
            args.grad_clip_norm,
            epoch,
            "train",
            args.log_every,
        )
        val_stats = run_epoch(
            model, val_loader, device, None, use_amp, 1, 0.0, epoch, "val", args.log_every
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "lr": scheduler.get_last_lr(),
        }
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch {epoch:03d} train loss={train_stats['loss']:.4f} "
            f"val loss={val_stats['loss']:.4f}"
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "benchmark_config": benchmark_config,
            "epoch": epoch,
            "val_loss": val_stats["loss"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(checkpoint, output_dir / "best.pt")
    print(f"Saved best checkpoint to {output_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
