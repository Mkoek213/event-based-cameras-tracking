#!/usr/bin/env python3
"""Train the lightweight controlled detector on DSEC-MOT event representations."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.data.pretrained_detection_dataset import DSECDenseRepresentationDataset
from src.data.representations import (
    REPRESENTATION_CHOICES,
    BenchmarkRepresentation,
    representation_channel_splits,
)
from src.data.unified_manifest import UnifiedDenseRepresentationDataset
from src.models.simple_detector import (
    SimpleDenseDetector,
    SimpleDetectorConfig,
    normalise_representation_tensor,
    simple_detector_loss,
)

DEFAULT_VAL_SEQUENCE = "zurich_city_01_d"


def distributed_is_available() -> bool:
    """Return True when torchrun started more than one worker."""

    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(requested_device: str) -> tuple[torch.device, int, int, int]:
    """Initialise single-node DDP when launched with torchrun."""

    if not distributed_is_available():
        return torch.device(requested_device), 0, 0, 1

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        backend = (
            "nccl" if torch.cuda.is_available() and requested_device.startswith("cuda") else "gloo"
        )
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested_device)
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_main(rank: int, *values: object) -> None:
    if is_main_process(rank):
        print(*values, flush=True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_stats(
    stats: dict[str, float], device: torch.device, distributed: bool
) -> dict[str, float]:
    """Average scalar epoch statistics across DDP ranks."""

    if not distributed:
        return stats
    keys = sorted(key for key in stats if key != "batches")
    values = torch.tensor([float(stats[key]) for key in keys], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    reduced = dict(stats)
    reduced.update({key: float(value.detach().cpu()) for key, value in zip(keys, values)})
    return reduced


def load_compatible_state(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, int]:
    """Load matching checkpoint tensors and skip incompatible heads."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint.get("model_state", checkpoint)
    source_state = {key.removeprefix("module."): value for key, value in source_state.items()}
    target = unwrap_model(model)
    target_state = target.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    target.load_state_dict(compatible, strict=False)
    return len(compatible), len(source_state) - len(compatible)


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
    component_splits: tuple[int, ...] = (),
    max_batches: int = 0,
    log_enabled: bool = True,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: defaultdict[str, float] = defaultdict(float)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and is_train)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    steps_done = 0
    for step, batch in enumerate(loader, start=1):
        if max_batches > 0 and steps_done >= max_batches:
            break
        events = normalise_representation_tensor(
            batch["events"].to(device, non_blocking=True), component_splits
        )
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

        steps_done += 1

        should_log = step == 1 or step % log_every == 0 or step == len(loader)
        if log_enabled and log_every > 0 and should_log:
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
    stats = {key: value / samples for key, value in totals.items()}
    stats["samples"] = float(samples)
    stats["batches"] = float(steps_done)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many training batches. Overrides epoch count when positive.",
    )
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
        "--architecture",
        choices=("simple", "csp_pan"),
        default="simple",
        help="Detector backbone: controlled small baseline or stronger CSP/PAN-style model.",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt in the run directory when it exists.",
    )
    parser.add_argument("--resume-from-pretrained", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simple_detector"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.feature_stride != 8:
        raise SystemExit("The simple detector currently supports --feature-stride 8 only.")
    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad-accum-steps must be positive.")

    device, rank, local_rank, world_size = setup_distributed(args.device)
    distributed = world_size > 1
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
    if args.fusion_mode == "single" and len(component_splits) > 1 and "eros" in args.representation:
        raise SystemExit(
            "EROS fusion variants in this benchmark must use two_branch or three_branch."
        )

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    representation = BenchmarkRepresentation(args.representation, args.num_bins)
    run_name = f"{args.representation}_bins{args.num_bins}_w{args.model_width}"
    if args.architecture != "simple":
        run_name = f"{run_name}_{args.architecture}"
    if args.fusion_mode != "single":
        run_name = f"{run_name}_{args.fusion_mode}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    using_manifest = args.train_manifest is not None or args.val_manifest is not None
    if using_manifest:
        if args.train_manifest is None or args.val_manifest is None:
            raise SystemExit("--train-manifest and --val-manifest must be provided together.")
        train_sequences: list[str] | None = None
        val_sequences: list[str] | None = None
        train_dataset = UnifiedDenseRepresentationDataset(
            manifest_path=args.train_manifest,
            representation=args.representation,
            feature_stride=args.feature_stride,
            positive_radius=args.positive_radius,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        val_dataset = UnifiedDenseRepresentationDataset(
            manifest_path=args.val_manifest,
            representation=args.representation,
            feature_stride=args.feature_stride,
            positive_radius=args.positive_radius,
            image_width=args.image_width,
            image_height=args.image_height,
        )
    else:
        train_sequences, val_sequences = choose_train_val_sequences(
            root=args.root,
            train_split=args.train_split,
            train_sequences_arg=args.train_sequences,
            val_sequences_arg=args.val_sequences,
        )
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
            "Check selected sequences/manifests and extracted files."
        )

    use_amp = (not args.no_amp) and device.type == "cuda"
    model_config = SimpleDetectorConfig(
        in_channels=representation.channels,
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
    )
    model = SimpleDenseDetector(model_config).to(device)
    if args.resume_from_pretrained is not None:
        loaded, skipped = load_compatible_state(model, args.resume_from_pretrained, device)
        print_main(
            rank,
            f"Loaded {loaded} tensors from {args.resume_from_pretrained}; "
            f"skipped {skipped} incompatible tensors.",
        )
    if distributed:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_steps = args.max_steps if args.max_steps > 0 else max(args.epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(scheduler_steps, 1))

    train_sampler = DistributedSampler(train_dataset_limited, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset_limited, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_dataset_limited,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset_limited,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    config = {
        "root": str(args.root),
        "train_manifest": str(args.train_manifest) if args.train_manifest else None,
        "val_manifest": str(args.val_manifest) if args.val_manifest else None,
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
        "architecture": args.architecture,
        "fusion_mode": args.fusion_mode,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps * world_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "background_weight": args.background_weight,
        "bbox_weight": args.bbox_weight,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "seed": args.seed,
        "world_size": world_size,
        "resume_from_pretrained": (
            str(args.resume_from_pretrained) if args.resume_from_pretrained else None
        ),
    }
    if is_main_process(rank):
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_val = float("inf")
    start_epoch = 1
    global_steps = 0
    last_checkpoint_path = output_dir / "last.pt"
    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        history_path = output_dir / "history.json"
        if history_path.exists() and is_main_process(rank):
            history = json.loads(history_path.read_text(encoding="utf-8"))
        else:
            history = checkpoint.get("history", [])

        completed_epoch = int(checkpoint.get("epoch", len(history)))
        start_epoch = completed_epoch + 1
        global_steps = int(checkpoint.get("global_steps", 0))
        if "best_val" in checkpoint:
            best_val = float(checkpoint["best_val"])
        elif history:
            best_val = min(float(row["val"]["loss"]) for row in history)
        print_main(
            rank,
            f"Resuming {run_name} from epoch {start_epoch}, global_steps={global_steps}",
        )

    if args.max_steps <= 0 and start_epoch > args.epochs:
        print_main(rank, f"Training {run_name} is already complete ({args.epochs} epochs).")
        cleanup_distributed()
        return 0
    if args.max_steps > 0 and global_steps >= args.max_steps:
        print_main(rank, f"Training {run_name} is already complete ({args.max_steps} steps).")
        cleanup_distributed()
        return 0

    print_main(rank, f"Training {run_name}")
    if using_manifest:
        print_main(
            rank,
            f"Train manifest: {args.train_manifest} ({len(train_dataset_limited)} samples)",
        )
        print_main(
            rank,
            f"Val manifest: {args.val_manifest} ({len(val_dataset_limited)} samples)",
        )
    else:
        print_main(
            rank,
            f"Train sequences: {train_sequences} ({len(train_dataset_limited)} samples)",
        )
        print_main(rank, f"Val sequences: {val_sequences} ({len(val_dataset_limited)} samples)")
    print_main(rank, f"Device: {device}, AMP: {use_amp}, world_size: {world_size}")

    epoch = start_epoch
    while True:
        if args.max_steps <= 0 and epoch > args.epochs:
            break
        if args.max_steps > 0 and global_steps >= args.max_steps:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        remaining_batches = max(args.max_steps - global_steps, 0) if args.max_steps > 0 else 0
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
            component_splits=component_splits,
            max_batches=remaining_batches,
            log_enabled=is_main_process(rank),
        )
        train_stats = reduce_stats(train_stats, device, distributed)
        global_steps += int(train_stats.get("batches", 0))

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
                component_splits=component_splits,
                log_enabled=is_main_process(rank),
            )
        val_stats = reduce_stats(val_stats, device, distributed)
        scheduler.step()
        row = {
            "epoch": epoch,
            "global_steps": global_steps,
            "train": train_stats,
            "val": val_stats,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        if is_main_process(rank):
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            print(
                f"epoch {epoch:03d} global_steps={global_steps} "
                f"train loss={train_stats['loss']:.4f} "
                f"cls={train_stats['cls_loss']:.4f} "
                f"bbox={train_stats['bbox_loss']:.4f} "
                f"val loss={val_stats['loss']:.4f} "
                f"cls={val_stats['cls_loss']:.4f} "
                f"bbox={val_stats['bbox_loss']:.4f}",
                flush=True,
            )

            checkpoint = {
                "model_state": unwrap_model(model).state_dict(),
                "model_config": model_config.to_dict(),
                "benchmark_config": config,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_val": best_val,
                "epoch": epoch,
                "global_steps": global_steps,
                "val_loss": val_stats["loss"],
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                checkpoint["best_val"] = best_val
                torch.save(checkpoint, output_dir / "best.pt")

        epoch += 1

    print_main(rank, f"Saved best checkpoint to {output_dir / 'best.pt'}")
    print_main(rank, f"Saved last checkpoint to {output_dir / 'last.pt'}")
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
