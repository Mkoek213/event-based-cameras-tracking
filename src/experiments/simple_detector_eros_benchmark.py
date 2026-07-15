#!/usr/bin/env python3
"""Train and evaluate the EROS variants of the controlled simple-detector benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.experiments.common import (
    DEFAULT_EVAL_TARGETS,
    CommandRunner,
    VariantSpec,
    require_checkpoint,
    simple_detector_eval_command,
    simple_detector_train_command,
    threshold_label,
)

VARIANTS = (
    VariantSpec("eros"),
    VariantSpec("voxel_grid_eros", "two_branch"),
    VariantSpec("event_frame_eros", "two_branch"),
    VariantSpec("event_frame_voxel_grid_eros", "three_branch"),
    VariantSpec("event_frame_voxel_grid", "gated_two_branch"),
)
VARIANT_CHOICES = tuple(variant.label for variant in VARIANTS)


def eros_config_label(radius: int, decay: float | None) -> str:
    decay_label = "default" if decay is None else f"{decay:.4f}".replace(".", "p")
    return f"eros_r{radius}_d{decay_label}"


def selected_variants(labels: list[str]) -> list[VariantSpec]:
    """Return selected EROS variants in stable declaration order."""

    selected = set(labels)
    return [variant for variant in VARIANTS if variant.label in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--eros-cache-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--eros-radius", type=int, default=10)
    parser.add_argument("--eros-decay", type=float, default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--architecture", choices=("simple", "csp_pan"), default="simple")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.97, 0.99],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_CHOICES,
        default=list(VARIANT_CHOICES),
        help="Subset of EROS benchmark variants to train and evaluate.",
    )
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simple_detector"))
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/dsec_mot_trackeval_simple_detector_eros")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/simple_detector_eros_logs"))
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable
    runner = CommandRunner(dry_run=args.dry_run)
    config_label = eros_config_label(args.eros_radius, args.eros_decay)
    default_cache_root = Path("data/cache/dsec_mot_eros")
    default_eros_config = args.eros_radius == 10 and args.eros_decay is None
    eros_cache_root = (
        args.eros_cache_root / config_label
        if args.eros_cache_root == default_cache_root and not default_eros_config
        else args.eros_cache_root
    )
    output_dir = args.output_dir if default_eros_config else args.output_dir / config_label
    arch_suffix = "" if args.architecture == "simple" else f"_{args.architecture}"

    if not args.skip_precompute:
        command = [
            python,
            "-m",
            "src.data.eros_precompute",
            "--root",
            str(args.root),
            "--output-root",
            str(eros_cache_root),
            "--radius",
            str(args.eros_radius),
        ]
        if args.eros_decay is not None:
            command.extend(["--decay", str(args.eros_decay)])
        code = runner.run(command, args.log_dir / f"precompute_{config_label}.log")
        if code != 0:
            return code

    for variant in selected_variants(args.variants):
        checkpoint = (
            output_dir
            / variant.checkpoint_name(args.num_bins, args.width, args.architecture)
            / "best.pt"
        )

        if not args.skip_train and (args.overwrite or not checkpoint.exists()):
            command = simple_detector_train_command(
                python=python,
                root=args.root,
                eros_cache_root=eros_cache_root,
                representation=variant.representation,
                fusion_mode=variant.fusion_mode,
                num_bins=args.num_bins,
                time_window_us=args.time_window_us,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                width=args.width,
                device=args.device,
                output_dir=output_dir,
                architecture=args.architecture,
            )
            code = runner.run(command, args.log_dir / f"train_{variant.label}{arch_suffix}.log")
            if code != 0:
                return code
        elif not args.skip_train:
            print(f"Skipping training, checkpoint exists: {checkpoint}")

        if args.skip_eval:
            continue
        if not require_checkpoint(checkpoint, args.dry_run):
            return 1

        for target in DEFAULT_EVAL_TARGETS:
            for threshold in args.thresholds:
                eval_name = (
                    f"{variant.label}{arch_suffix}_{target.label}_thr{threshold_label(threshold)}"
                )
                eval_run_name = eval_name if default_eros_config else f"{config_label}_{eval_name}"
                summary = args.results_root / eval_run_name / "metrics_summary.json"
                if summary.exists() and not args.overwrite:
                    print(f"Skipping eval, summary exists: {summary}")
                    continue
                command = simple_detector_eval_command(
                    python=python,
                    checkpoint=checkpoint,
                    root=args.root,
                    eros_cache_root=eros_cache_root,
                    target=target,
                    threshold=threshold,
                    device=args.device,
                    max_detections=args.max_detections,
                    output_root=args.results_root,
                    run_name=eval_run_name,
                )
                code = runner.run(command, args.log_dir / f"eval_{eval_name}.log")
                if code != 0:
                    return code

    print("EROS simple-detector benchmark queue completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
