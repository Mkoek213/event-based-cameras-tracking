#!/usr/bin/env python3
"""Orchestrate the R0/R1/R2 event-ReID benchmark on both DSEC-MOT scopes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import torch

from src.experiments.common import CommandRunner, threshold_label

ALL_CLASS_SCORES = (0.10, 0.25, 0.50, 0.70, 0.90, 0.95)
CAR_ONLY_SCORES = (0.90, 0.95, 0.97, 0.99)
APPEARANCE_THRESHOLDS = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50)
PROXIMITY_THRESHOLDS = (0.30, 0.50, 0.70)
VAL_SEQUENCE = "zurich_city_01_d"
TEST_SEQUENCES = ("interlaken_00_d", "zurich_city_00_b")
DEFAULT_APPEARANCE = 0.25
DEFAULT_PROXIMITY = 0.50

R0_CHECKPOINTS = {
    "all_classes": Path(
        "runs/simple_detector_sweep/bins3_win50ms/"
        "event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt"
    ),
    "car_only": Path(
        "runs/simple_detector_car_only/bins3_win50ms/"
        "event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt"
    ),
}


@dataclass(frozen=True)
class ScopeSpec:
    name: str
    score_thresholds: tuple[float, ...]
    class_ids: tuple[int, ...] | None
    num_classes: int
    classes_to_eval: tuple[str, ...] | None


@dataclass(frozen=True)
class EvaluationSpec:
    scope: str
    variant: str
    mode: str
    split: str
    score_threshold: float
    appearance_threshold: float
    proximity_threshold: float
    with_reid: bool

    @property
    def run_name(self) -> str:
        return (
            f"{self.scope}_{self.variant}_{self.mode}_{self.split}_"
            f"score{threshold_label(self.score_threshold)}_"
            f"appearance{threshold_label(self.appearance_threshold)}_"
            f"proximity{threshold_label(self.proximity_threshold)}"
        )


SCOPES = {
    "all_classes": ScopeSpec(
        name="all_classes",
        score_thresholds=ALL_CLASS_SCORES,
        class_ids=None,
        num_classes=7,
        classes_to_eval=None,
    ),
    "car_only": ScopeSpec(
        name="car_only",
        score_thresholds=CAR_ONLY_SCORES,
        class_ids=(0,),
        num_classes=1,
        classes_to_eval=("car",),
    ),
}


def selected_scopes(value: str) -> tuple[ScopeSpec, ...]:
    if value == "both":
        return SCOPES["all_classes"], SCOPES["car_only"]
    return (SCOPES[value],)


def embedding_run_name(variant: str) -> str:
    suffix = "r1_non_recurrent" if variant == "R1" else "r2_recurrent"
    return f"event_frame_voxel_grid_bins3_w32_gated_two_branch_{suffix}"


def checkpoint_path(scope: ScopeSpec, variant: str, runs_root: Path) -> Path:
    if variant == "R0":
        return R0_CHECKPOINTS[scope.name]
    return runs_root / scope.name / embedding_run_name(variant) / "best.pt"


def checkpoint_has_completed_epochs(checkpoint: Path, epochs: int) -> bool:
    """Return whether a checkpoint has a complete history file."""

    if not checkpoint.exists():
        return False
    history_path = checkpoint.parent / "history.json"
    if not history_path.exists():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(history, list) and len(history) >= epochs


def build_training_command(
    python: str,
    root: Path,
    runs_root: Path,
    scope: ScopeSpec,
    variant: str,
    args: argparse.Namespace,
) -> list[str]:
    if variant not in {"R1", "R2"}:
        raise ValueError("Only R1 and R2 have training commands.")
    command = [
        python,
        "-m",
        "src.training.recurrent_embedding_detector",
        "--root",
        str(root),
        "--representation",
        "event_frame_voxel_grid",
        "--fusion-mode",
        "gated_two_branch",
        "--num-bins",
        "3",
        "--time-window-us",
        "50000",
        "--model-width",
        "32",
        "--embedding-hidden-dim",
        "128",
        "--embedding-dim",
        "256",
        "--roi-size",
        "7",
        "--identity-ce-weight",
        "1.0",
        "--triplet-weight",
        "1.0",
        "--triplet-margin",
        "0.3",
        "--clip-length",
        "8",
        "--clip-stride",
        "8",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--grad-clip-norm",
        "5.0",
        "--seed",
        "0",
        "--device",
        args.device,
        "--output-dir",
        str(runs_root / scope.name),
        "--run-name",
        embedding_run_name(variant),
        "--no-recurrent-embedding" if variant == "R1" else "--recurrent-embedding",
    ]
    if scope.class_ids is not None:
        command.extend(["--class-ids", ",".join(str(value) for value in scope.class_ids)])
        command.extend(["--num-classes", str(scope.num_classes)])
    if args.resume:
        command.append("--resume")
    return command


def validation_matrix(scopes: Iterable[ScopeSpec]) -> list[EvaluationSpec]:
    matrix: list[EvaluationSpec] = []
    for scope in scopes:
        matrix.extend(
            EvaluationSpec(
                scope=scope.name,
                variant="R0",
                mode="motion",
                split="val",
                score_threshold=score,
                appearance_threshold=DEFAULT_APPEARANCE,
                proximity_threshold=DEFAULT_PROXIMITY,
                with_reid=False,
            )
            for score in scope.score_thresholds
        )
        for variant in ("R1", "R2"):
            matrix.extend(
                EvaluationSpec(
                    scope=scope.name,
                    variant=variant,
                    mode="reid",
                    split="val",
                    score_threshold=score,
                    appearance_threshold=appearance,
                    proximity_threshold=proximity,
                    with_reid=True,
                )
                for score, appearance, proximity in product(
                    scope.score_thresholds,
                    APPEARANCE_THRESHOLDS,
                    PROXIMITY_THRESHOLDS,
                )
            )
            matrix.extend(
                EvaluationSpec(
                    scope=scope.name,
                    variant=variant,
                    mode="motion",
                    split="val",
                    score_threshold=score,
                    appearance_threshold=DEFAULT_APPEARANCE,
                    proximity_threshold=DEFAULT_PROXIMITY,
                    with_reid=False,
                )
                for score in scope.score_thresholds
            )
    return matrix


def evaluation_command(
    python: str,
    root: Path,
    results_root: Path,
    checkpoint: Path,
    scope: ScopeSpec,
    spec: EvaluationSpec,
    device: str,
    max_detections: int,
) -> list[str]:
    sequences = (VAL_SEQUENCE,) if spec.split == "val" else TEST_SEQUENCES
    split = "train" if spec.split == "val" else "test"
    command = [
        python,
        "-m",
        "src.evaluation.simple_detector_trackeval_cli",
        "--checkpoint",
        str(checkpoint),
        "--root",
        str(root),
        "--split",
        split,
        "--sequences",
        *sequences,
        "--device",
        device,
        "--score-threshold",
        str(spec.score_threshold),
        "--max-detections",
        str(max_detections),
        "--tracker-backend",
        "boxmot_botsort",
        "--tracker-name",
        f"event_reid_{spec.mode}",
        "--track-appearance-thresh",
        str(spec.appearance_threshold),
        "--track-proximity-thresh",
        str(spec.proximity_threshold),
        "--output-root",
        str(results_root),
        "--run-name",
        spec.run_name,
    ]
    if spec.with_reid:
        command.append("--track-with-reid")
    if scope.classes_to_eval is not None:
        command.extend(["--classes-to-eval", *scope.classes_to_eval])
    return command


def summary_path(results_root: Path, spec: EvaluationSpec) -> Path:
    return results_root / spec.run_name / "metrics_summary.json"


def selection_key(payload: dict) -> tuple[float, float, float]:
    metrics = payload["aggregate"]
    return (
        float(metrics.get("HOTA", -1.0)),
        float(metrics.get("AssA", -1.0)),
        float(metrics.get("IDF1", -1.0)),
    )


def select_validation_configuration(
    results_root: Path,
    candidates: list[EvaluationSpec],
) -> tuple[EvaluationSpec, dict]:
    missing = [
        summary_path(results_root, spec)
        for spec in candidates
        if not summary_path(results_root, spec).exists()
    ]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Validation grid is incomplete ({len(missing)} summaries missing), e.g. {preview}"
        )
    loaded = [
        (
            spec,
            json.loads(summary_path(results_root, spec).read_text(encoding="utf-8")),
        )
        for spec in candidates
    ]
    return max(loaded, key=lambda item: selection_key(item[1]))


def grouped_validation_candidates(
    matrix: list[EvaluationSpec],
) -> dict[tuple[str, str, str], list[EvaluationSpec]]:
    grouped: dict[tuple[str, str, str], list[EvaluationSpec]] = {}
    for spec in matrix:
        grouped.setdefault((spec.scope, spec.variant, spec.mode), []).append(spec)
    return grouped


def test_spec_from_selection(spec: EvaluationSpec) -> EvaluationSpec:
    return EvaluationSpec(
        scope=spec.scope,
        variant=spec.variant,
        mode=spec.mode,
        split="test",
        score_threshold=spec.score_threshold,
        appearance_threshold=spec.appearance_threshold,
        proximity_threshold=spec.proximity_threshold,
        with_reid=spec.with_reid,
    )


def checkpoint_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "selected_epoch": "",
            "retrieval_map": "",
            "retrieval_rank1": "",
        }
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "selected_epoch": checkpoint.get("selected_epoch", checkpoint.get("epoch", "")),
        "retrieval_map": checkpoint.get("val_retrieval_map", ""),
        "retrieval_rank1": checkpoint.get("val_retrieval_rank1", ""),
    }


def write_selection_summary(
    path: Path,
    selections: dict[tuple[str, str, str], tuple[EvaluationSpec, dict]],
) -> None:
    payload = {
        "selection_policy": ["HOTA", "AssA", "IDF1"],
        "validation_sequence": VAL_SEQUENCE,
        "selections": [
            {
                **asdict(spec),
                "run_name": spec.run_name,
                "validation_metrics": summary["aggregate"],
            }
            for _, (spec, summary) in sorted(selections.items())
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_aggregate_outputs(
    results_root: Path,
    runs_root: Path,
    selections: dict[tuple[str, str, str], tuple[EvaluationSpec, dict]],
) -> None:
    records: list[dict] = []
    csv_rows: list[dict[str, object]] = []
    for key, (val_spec, val_summary) in sorted(selections.items()):
        scope = SCOPES[val_spec.scope]
        test_spec = test_spec_from_selection(val_spec)
        path = summary_path(results_root, test_spec)
        if not path.exists():
            continue
        test_summary = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = checkpoint_path(scope, val_spec.variant, runs_root)
        metadata = checkpoint_metadata(checkpoint)
        base = {
            "scope": val_spec.scope,
            "variant": val_spec.variant,
            "mode": val_spec.mode,
            "checkpoint": str(checkpoint),
            **metadata,
            "score_threshold": val_spec.score_threshold,
            "appearance_threshold": val_spec.appearance_threshold,
            "proximity_threshold": val_spec.proximity_threshold,
            "reid_enabled": val_spec.with_reid,
        }
        records.append(
            {
                **base,
                "validation_metrics": val_summary["aggregate"],
                "combined_test": test_summary["aggregate"],
                "per_sequence_test": {
                    sequence: payload["metrics"]
                    for sequence, payload in test_summary["per_sequence"].items()
                },
            }
        )
        sequence_metrics = {
            sequence: payload["metrics"]
            for sequence, payload in test_summary["per_sequence"].items()
        }
        sequence_metrics["COMBINED"] = test_summary["aggregate"]
        for sequence, metrics in sequence_metrics.items():
            csv_rows.append(
                {
                    **base,
                    "sequence": sequence,
                    "HOTA": metrics.get("HOTA", ""),
                    "AssA": metrics.get("AssA", ""),
                    "DetA": metrics.get("DetA", ""),
                    "MOTA": metrics.get("MOTA", ""),
                    "IDF1": metrics.get("IDF1", ""),
                    "IDS": metrics.get("IDS", ""),
                    "FP": metrics.get("FP", ""),
                    "FN": metrics.get("FN", ""),
                }
            )

    aggregate_json = results_root / "benchmark_summary.json"
    aggregate_csv = results_root / "benchmark_summary.csv"
    aggregate_json.write_text(
        json.dumps(
            {
                "selection_policy": ["HOTA", "AssA", "IDF1"],
                "validation_sequence": VAL_SEQUENCE,
                "test_sequences": list(TEST_SEQUENCES),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "scope",
        "variant",
        "mode",
        "checkpoint",
        "selected_epoch",
        "score_threshold",
        "appearance_threshold",
        "proximity_threshold",
        "retrieval_map",
        "retrieval_rank1",
        "reid_enabled",
        "sequence",
        "HOTA",
        "AssA",
        "DetA",
        "MOTA",
        "IDF1",
        "IDS",
        "FP",
        "FN",
    ]
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved aggregate JSON to {aggregate_json}")
    print(f"Saved aggregate CSV to {aggregate_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument(
        "--scope",
        choices=("all_classes", "car_only", "both"),
        default="both",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/event_reid_embedding"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/dsec_mot_event_reid_embedding"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/event_reid_embedding/logs"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scopes = selected_scopes(args.scope)
    python = sys.executable
    runner = CommandRunner(dry_run=args.dry_run)

    if not args.skip_train:
        for scope in scopes:
            for variant in ("R1", "R2"):
                checkpoint = checkpoint_path(scope, variant, args.runs_root)
                complete = checkpoint_has_completed_epochs(checkpoint, args.epochs)
                if complete and not args.overwrite:
                    print(f"Skipping completed training: {checkpoint}")
                    continue
                command = build_training_command(
                    python,
                    args.root,
                    args.runs_root,
                    scope,
                    variant,
                    args,
                )
                code = runner.run(
                    command,
                    args.log_dir / f"train_{scope.name}_{variant.lower()}.log",
                )
                if code != 0:
                    return code

    matrix = validation_matrix(scopes)
    if not args.skip_val:
        print(f"Validation matrix contains {len(matrix)} runs.")
        for spec in matrix:
            scope = SCOPES[spec.scope]
            checkpoint = checkpoint_path(scope, spec.variant, args.runs_root)
            if not checkpoint.exists() and not args.dry_run:
                print(f"Missing checkpoint, cannot validate: {checkpoint}")
                return 1
            path = summary_path(args.results_root, spec)
            if path.exists() and not args.overwrite:
                print(f"Skipping completed validation: {path}")
                continue
            command = evaluation_command(
                python,
                args.root,
                args.results_root,
                checkpoint,
                scope,
                spec,
                args.device,
                args.max_detections,
            )
            code = runner.run(
                command,
                args.log_dir / f"eval_{spec.run_name}.log",
            )
            if code != 0:
                return code

    if args.dry_run:
        if not args.skip_test:
            print(
                "Test stage requires completed validation summaries; it will run the two "
                "test sequences once per validation-selected configuration."
            )
        return 0

    grouped = grouped_validation_candidates(matrix)
    try:
        selections = {
            key: select_validation_configuration(args.results_root, candidates)
            for key, candidates in grouped.items()
        }
    except FileNotFoundError as exc:
        if args.skip_test:
            print(f"Validation selection unavailable: {exc}")
            return 0
        raise SystemExit(str(exc)) from exc

    write_selection_summary(args.results_root / "validation_selection.json", selections)
    if not args.skip_test:
        for _, (val_spec, _) in sorted(selections.items()):
            test_spec = test_spec_from_selection(val_spec)
            scope = SCOPES[test_spec.scope]
            checkpoint = checkpoint_path(scope, test_spec.variant, args.runs_root)
            path = summary_path(args.results_root, test_spec)
            if path.exists() and not args.overwrite:
                print(f"Skipping completed test: {path}")
                continue
            command = evaluation_command(
                python,
                args.root,
                args.results_root,
                checkpoint,
                scope,
                test_spec,
                args.device,
                args.max_detections,
            )
            code = runner.run(
                command,
                args.log_dir / f"eval_{test_spec.run_name}.log",
            )
            if code != 0:
                return code

    write_aggregate_outputs(args.results_root, args.runs_root, selections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
