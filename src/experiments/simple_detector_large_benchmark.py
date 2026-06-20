#!/usr/bin/env python3
"""Run a larger controlled benchmark for simple detector input representations."""

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
    threshold_label,
    variant_label,
)

VARIANTS = (
    VariantSpec("event_frame"),
    VariantSpec("voxel_grid"),
    VariantSpec("event_frame_voxel_grid"),
    VariantSpec("event_frame_voxel_grid", "two_branch"),
    VariantSpec("event_frame_voxel_grid", "gated_two_branch"),
)


def train_command(
    *,
    python: str,
    args: argparse.Namespace,
    variant: VariantSpec,
    seed: int,
    output_dir: Path,
) -> list[str]:
    """Build the large-model training command, including optional grad accumulation."""

    physical_batch_size = args.batch_size
    grad_accum_steps = 1
    if variant.fusion_mode in {"two_branch", "gated_two_branch"} and args.two_branch_batch_size:
        physical_batch_size = args.two_branch_batch_size
        if args.batch_size % physical_batch_size != 0:
            raise SystemExit("--batch-size must be divisible by --two-branch-batch-size.")
        grad_accum_steps = args.batch_size // physical_batch_size

    command = [
        python,
        "-m",
        "src.training.simple_detector",
        "--root",
        str(args.root),
        "--representation",
        variant.representation,
        "--fusion-mode",
        variant.fusion_mode,
        "--num-bins",
        str(args.num_bins),
        "--time-window-us",
        str(args.time_window_us),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(physical_batch_size),
        "--grad-accum-steps",
        str(grad_accum_steps),
        "--num-workers",
        str(args.num_workers),
        "--model-width",
        str(args.width),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
    ]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--two-branch-batch-size",
        type=int,
        default=0,
        help="Physical batch size for two-branch. Gradient accumulation preserves --batch-size.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.97, 0.99])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simple_detector_large"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/dsec_mot_trackeval_simple_detector_large"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/simple_detector_large_logs"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable
    runner = CommandRunner(dry_run=args.dry_run)

    for seed in args.seeds:
        seed_output_dir = args.output_dir / f"w{args.width}_seed{seed}"
        for variant in VARIANTS:
            run_name = variant.checkpoint_name(args.num_bins, args.width)
            label = variant_label(variant.representation, variant.fusion_mode)
            checkpoint = seed_output_dir / run_name / "best.pt"

            if not args.skip_train:
                if checkpoint.exists() and not args.overwrite:
                    print(f"Skipping training, checkpoint exists: {checkpoint}")
                else:
                    code = runner.run(
                        train_command(
                            python=python,
                            args=args,
                            variant=variant,
                            seed=seed,
                            output_dir=seed_output_dir,
                        ),
                        args.log_dir / f"train_w{args.width}_seed{seed}_{label}.log",
                    )
                    if code != 0:
                        return code

            if args.skip_eval:
                continue
            if not require_checkpoint(checkpoint, args.dry_run):
                return 1

            for target in DEFAULT_EVAL_TARGETS:
                for threshold in args.thresholds:
                    eval_run_name = (
                        f"w{args.width}_seed{seed}_{label}_{target.label}_thr"
                        f"{threshold_label(threshold)}"
                    )
                    summary_path = args.results_root / eval_run_name / "metrics_summary.json"
                    if summary_path.exists() and not args.overwrite:
                        print(f"Skipping eval, summary exists: {summary_path}")
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
                        run_name=eval_run_name,
                    )
                    code = runner.run(command, args.log_dir / f"eval_{eval_run_name}.log")
                    if code != 0:
                        return code

    print("Large benchmark queue completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
