#!/usr/bin/env python3
"""Train and evaluate car-only SimpleDenseDetector representation benchmarks."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from src.experiments.common import (
    DEFAULT_EVAL_TARGETS,
    CommandRunner,
    require_checkpoint,
    simple_detector_eval_command,
    simple_detector_train_command,
    sweep_label,
    threshold_label,
    unique_specs,
    variant_label,
)

BASE_BINS = 5
BASE_WINDOW_US = 50_000
THRESHOLDS = (0.90, 0.95, 0.97, 0.99)


@dataclass(frozen=True)
class RunSpec:
    representation: str
    fusion_mode: str
    num_bins: int
    time_window_us: int

    @property
    def label(self) -> str:
        return f"{sweep_label(self.num_bins, self.time_window_us)}_{self.variant_label}"

    @property
    def variant_label(self) -> str:
        return variant_label(self.representation, self.fusion_mode)

    def checkpoint_name(self, width: int) -> str:
        name = f"{self.representation}_bins{self.num_bins}_w{width}"
        return name if self.fusion_mode == "single" else f"{name}_{self.fusion_mode}"


def build_specs(include_eros: bool, include_sweep: bool) -> list[RunSpec]:
    specs = [
        RunSpec("event_frame", "single", BASE_BINS, BASE_WINDOW_US),
        RunSpec("voxel_grid", "single", BASE_BINS, BASE_WINDOW_US),
        RunSpec("event_frame_voxel_grid", "single", BASE_BINS, BASE_WINDOW_US),
        RunSpec("event_frame_voxel_grid", "two_branch", BASE_BINS, BASE_WINDOW_US),
        RunSpec("event_frame_voxel_grid", "gated_two_branch", BASE_BINS, BASE_WINDOW_US),
    ]
    if include_eros:
        specs.extend(
            [
                RunSpec("eros", "single", BASE_BINS, BASE_WINDOW_US),
                RunSpec("event_frame_eros", "two_branch", BASE_BINS, BASE_WINDOW_US),
                RunSpec("voxel_grid_eros", "two_branch", BASE_BINS, BASE_WINDOW_US),
                RunSpec("event_frame_voxel_grid_eros", "three_branch", BASE_BINS, BASE_WINDOW_US),
            ]
        )
    if include_sweep:
        for window_us in (25_000, 50_000, 100_000):
            for fusion_mode in ("two_branch", "gated_two_branch"):
                specs.append(RunSpec("event_frame_voxel_grid", fusion_mode, BASE_BINS, window_us))
        for num_bins in (3, 7):
            for fusion_mode in ("two_branch", "gated_two_branch"):
                specs.append(
                    RunSpec("event_frame_voxel_grid", fusion_mode, num_bins, BASE_WINDOW_US)
                )
    return unique_specs(specs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simple_detector_car_only"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/dsec_mot_trackeval_simple_detector_car_only"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/simple_detector_car_only_logs"))
    parser.add_argument("--eros-cache-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--skip-eros", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable
    runner = CommandRunner(dry_run=args.dry_run)
    specs = build_specs(include_eros=not args.skip_eros, include_sweep=not args.skip_sweep)

    if not args.skip_eros and not args.skip_train:
        code = runner.run(
            [
                python,
                "-m",
                "src.data.eros_precompute",
                "--root",
                str(args.root),
                "--output-root",
                str(args.eros_cache_root),
            ],
            args.log_dir / "precompute_eros.log",
        )
        if code != 0:
            return code

    for spec in specs:
        sweep_output_dir = args.output_dir / sweep_label(spec.num_bins, spec.time_window_us)
        checkpoint = sweep_output_dir / spec.checkpoint_name(args.width) / "best.pt"
        if not args.skip_train and (args.overwrite or not checkpoint.exists()):
            command = simple_detector_train_command(
                python=python,
                root=args.root,
                representation=spec.representation,
                fusion_mode=spec.fusion_mode,
                num_bins=spec.num_bins,
                time_window_us=spec.time_window_us,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                width=args.width,
                device=args.device,
                output_dir=sweep_output_dir,
                eros_cache_root=args.eros_cache_root,
                class_ids=[0],
                num_classes=1,
            )
            code = runner.run(command, args.log_dir / f"train_{spec.label}.log")
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
                eval_name = f"{spec.label}_{target.label}_thr{threshold_label(threshold)}"
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
                    eros_cache_root=args.eros_cache_root,
                    classes_to_eval=["car"],
                )
                code = runner.run(command, args.log_dir / f"eval_{eval_name}.log")
                if code != 0:
                    return code

    code = runner.run(
        [
            python,
            "-m",
            "src.evaluation.detection_metrics_cli",
            "--results-root",
            str(args.results_root),
            "--output-csv",
            str(args.results_root / "detection_metrics.csv"),
        ],
        args.log_dir / "detection_metrics.log",
    )
    if code != 0:
        return code

    code = runner.run(
        [
            python,
            "-m",
            "src.experiments.summarise_car_only_results",
            "--results-root",
            str(args.results_root),
            "--output-csv",
            str(args.results_root / "car_only_metrics.csv"),
            "--selected-output-csv",
            str(args.results_root / "car_only_val_selected_metrics.csv"),
        ],
        args.log_dir / "car_only_summary.log",
    )
    if code != 0:
        return code

    print("Car-only benchmark queue completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
