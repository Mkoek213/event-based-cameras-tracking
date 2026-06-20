#!/usr/bin/env python3
"""Run the complete RGB-pretrained representation benchmark, including EROS fusions."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from src.experiments.common import (
    DEFAULT_EVAL_TARGETS,
    CommandRunner,
    require_checkpoint,
    threshold_label,
)


@dataclass(frozen=True)
class PretrainedVariant:
    """Representation and adapter mode used by the RGB-pretrained detector."""

    representation: str
    adapter_mode: str

    @property
    def short_label(self) -> str:
        return f"{self.representation}_{self.adapter_mode}"

    @property
    def run_label(self) -> str:
        return f"{self.short_label}_fasterrcnn_r50_fpn_v2"


VARIANTS = (
    PretrainedVariant("event_frame", "single"),
    PretrainedVariant("voxel_grid", "single"),
    PretrainedVariant("event_frame_voxel_grid", "single"),
    PretrainedVariant("event_frame_voxel_grid", "multi_branch"),
    PretrainedVariant("eros", "single"),
    PretrainedVariant("event_frame_eros", "single"),
    PretrainedVariant("event_frame_eros", "multi_branch"),
    PretrainedVariant("voxel_grid_eros", "single"),
    PretrainedVariant("voxel_grid_eros", "multi_branch"),
    PretrainedVariant("event_frame_voxel_grid_eros", "single"),
    PretrainedVariant("event_frame_voxel_grid_eros", "multi_branch"),
)


def selected_variants(labels: list[str] | None) -> list[PretrainedVariant]:
    """Return selected pretrained variants, preserving declaration order."""

    if labels is None:
        return list(VARIANTS)
    selected = set(labels)
    return [variant for variant in VARIANTS if variant.short_label in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--eros-cache-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--adapter-width", type=int, default=32)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional labels such as event_frame_single or event_frame_eros_multi_branch.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pretrained_detector"))
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/dsec_mot_trackeval_pretrained_detector")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/pretrained_detector_logs"))
    parser.add_argument("--skip-eros-precompute", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable
    runner = CommandRunner(dry_run=args.dry_run)
    if not args.skip_eros_precompute:
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
        if code:
            return code

    for variant in selected_variants(args.variants):
        checkpoint = args.output_dir / variant.run_label / "best.pt"
        if not args.skip_train and (args.overwrite or not checkpoint.exists()):
            command = [
                python,
                "-m",
                "src.training.pretrained_detector",
                "--root",
                str(args.root),
                "--representation",
                variant.representation,
                "--adapter-mode",
                variant.adapter_mode,
                "--adapter-width",
                str(args.adapter_width),
                "--num-bins",
                str(args.num_bins),
                "--time-window-us",
                str(args.time_window_us),
                "--eros-cache-root",
                str(args.eros_cache_root),
                "--epochs",
                str(args.epochs),
                "--freeze-backbone-epochs",
                str(args.freeze_backbone_epochs),
                "--batch-size",
                str(args.batch_size),
                "--grad-accum-steps",
                str(args.grad_accum_steps),
                "--num-workers",
                str(args.num_workers),
                "--lr",
                str(args.lr),
                "--device",
                args.device,
                "--output-dir",
                str(args.output_dir),
            ]
            code = runner.run(command, args.log_dir / f"train_{variant.run_label}.log")
            if code:
                return code
        if args.skip_eval:
            continue
        if not require_checkpoint(checkpoint, args.dry_run):
            return 1
        for target in DEFAULT_EVAL_TARGETS:
            for threshold in args.thresholds:
                run_name = f"{variant.run_label}_{target.label}_thr{threshold_label(threshold)}"
                summary = args.results_root / run_name / "metrics_summary.json"
                if summary.exists() and not args.overwrite:
                    continue
                command = [
                    python,
                    "-m",
                    "src.evaluation.pretrained_detector_trackeval_cli",
                    "--checkpoint",
                    str(checkpoint),
                    "--root",
                    str(args.root),
                    "--split",
                    target.split,
                    "--sequences",
                    target.sequence,
                    "--score-threshold",
                    str(threshold),
                    "--max-detections",
                    str(args.max_detections),
                    "--device",
                    args.device,
                    "--output-root",
                    str(args.results_root),
                    "--run-name",
                    run_name,
                ]
                code = runner.run(command, args.log_dir / f"eval_{run_name}.log")
                if code:
                    return code
    print("Pretrained representation benchmark completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
