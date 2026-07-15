#!/usr/bin/env python3
"""Run a controlled sweep over representation parameters for SimpleDenseDetector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.experiments.common import (
    DEFAULT_EVAL_TARGETS,
    CommandRunner,
    VariantSpec,
    checkpoint_has_completed_epochs,
    require_checkpoint,
    simple_detector_eval_command,
    simple_detector_train_command,
    sweep_label,
    threshold_label,
)

VARIANTS = (
    VariantSpec("event_frame"),
    VariantSpec("voxel_grid"),
    VariantSpec("event_frame_voxel_grid"),
    VariantSpec("event_frame_voxel_grid", "two_branch"),
    VariantSpec("event_frame_voxel_grid", "gated_two_branch"),
)
VARIANT_CHOICES = tuple(variant.label for variant in VARIANTS)


def selected_variants(labels: list[str]) -> list[VariantSpec]:
    """Return variants requested on the command line, preserving declaration order."""

    selected = set(labels)
    return [variant for variant in VARIANTS if variant.label in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--architecture", choices=("simple", "csp_pan"), default="simple")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-bins", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--time-window-us", type=int, nargs="+", default=[25_000, 50_000, 100_000])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.97, 0.99])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_CHOICES,
        default=list(VARIANT_CHOICES),
        help="Subset of representation variants to train and evaluate.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simple_detector_sweep"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/dsec_mot_trackeval_simple_detector_sweep"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/simple_detector_sweep_logs"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete training runs from last.pt instead of starting from scratch.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable
    runner = CommandRunner(dry_run=args.dry_run)
    arch_suffix = "" if args.architecture == "simple" else f"_{args.architecture}"

    for bins in args.num_bins:
        for window_us in args.time_window_us:
            sweep_name = sweep_label(bins, window_us)
            sweep_output_dir = args.output_dir / sweep_name
            for variant in selected_variants(args.variants):
                checkpoint = (
                    sweep_output_dir
                    / variant.checkpoint_name(bins, args.width, args.architecture)
                    / "best.pt"
                )
                training_complete = checkpoint_has_completed_epochs(checkpoint, args.epochs)
                if not args.skip_train and (args.overwrite or not training_complete):
                    command = simple_detector_train_command(
                        python=python,
                        root=args.root,
                        representation=variant.representation,
                        fusion_mode=variant.fusion_mode,
                        num_bins=bins,
                        time_window_us=window_us,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        width=args.width,
                        grad_accum_steps=args.grad_accum_steps,
                        device=args.device,
                        architecture=args.architecture,
                        output_dir=sweep_output_dir,
                        resume=args.resume and not args.overwrite,
                    )
                    code = runner.run(
                        command,
                        args.log_dir / f"train_{sweep_name}_{variant.label}{arch_suffix}.log",
                    )
                    if code != 0:
                        return code
                elif not args.skip_train:
                    print(f"Skipping training, checkpoint complete: {checkpoint}")

                if args.skip_eval:
                    continue
                if not require_checkpoint(checkpoint, args.dry_run):
                    return 1
                for target in DEFAULT_EVAL_TARGETS:
                    for threshold in args.thresholds:
                        eval_name = (
                            f"{sweep_name}_{variant.label}{arch_suffix}_{target.label}_thr"
                            f"{threshold_label(threshold)}"
                        )
                        summary = args.results_root / eval_name / "metrics_summary.json"
                        if summary.exists() and not args.overwrite:
                            print(f"Skipping eval, summary exists: {summary}")
                            continue
                        command = simple_detector_eval_command(
                            python=python,
                            checkpoint=checkpoint,
                            root=args.root,
                            target=target,
                            threshold=threshold,
                            device=args.device,
                            max_detections=args.max_detections,
                            output_root=args.results_root,
                            run_name=eval_name,
                        )
                        code = runner.run(command, args.log_dir / f"eval_{eval_name}.log")
                        if code != 0:
                            return code
    print("Representation sweep completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
