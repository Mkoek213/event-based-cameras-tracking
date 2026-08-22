#!/usr/bin/env python3
"""Run module-contribution and temporal-memory ablations on DSEC-MOT."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from src.experiments.common import REPO_ROOT, threshold_label
from src.experiments.recurrent_placement_ablation import (
    APPEARANCE_THRESHOLDS,
    CHECKPOINT_SELECTION_SCORE,
    DEFAULT_APPEARANCE,
    DEFAULT_PROXIMITY,
    PROXIMITY_THRESHOLDS,
    SCOPES,
    TEST_SEQUENCES,
    VAL_SEQUENCE,
    Job,
    ScopeSpec,
    best_candidate,
    load_summary,
    run_parallel,
    save_selected_checkpoint,
    variant_for_name,
)

DEFAULT_RECURRENCE_VARIANT = "gru_direct_neck"
BASE_CHECKPOINTS = {
    "all_classes": {
        "two_branch": Path(
            "runs/simple_detector_sweep/bins3_win50ms/"
            "event_frame_voxel_grid_bins3_w32_two_branch/best.pt"
        ),
        "gated_two_branch": Path(
            "runs/simple_detector_sweep/bins3_win50ms/"
            "event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt"
        ),
    },
    "car_only": {
        "two_branch": Path(
            "runs/simple_detector_car_only/bins3_win50ms/"
            "event_frame_voxel_grid_bins3_w32_two_branch/best.pt"
        ),
        "gated_two_branch": Path(
            "runs/simple_detector_car_only/bins3_win50ms/"
            "event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt"
        ),
    },
}
PRIMARY_METRICS = ("HOTA", "AssA", "DetA", "MOTA", "IDF1")


@dataclass(frozen=True)
class BenchmarkVariant:
    suite: str
    name: str
    fusion_mode: str
    embedding: bool
    recurrence: bool
    clip_length: int = 8
    carry_state: bool = False
    burn_in_frames: int = 0
    static_checkpoint: bool = False

    @property
    def full_name(self) -> str:
        return f"{self.suite}/{self.name}"


def contribution_variants() -> tuple[BenchmarkVariant, ...]:
    return (
        BenchmarkVariant(
            "contribution", "baseline", "two_branch", False, False, static_checkpoint=True
        ),
        BenchmarkVariant(
            "contribution", "gating_only", "gated_two_branch", False, False, static_checkpoint=True
        ),
        BenchmarkVariant("contribution", "embedding_only", "two_branch", True, False),
        BenchmarkVariant("contribution", "recurrence_only", "two_branch", False, True),
        BenchmarkVariant("contribution", "all_modules", "gated_two_branch", True, True),
    )


def memory_variants() -> tuple[BenchmarkVariant, ...]:
    variants = [
        BenchmarkVariant(
            "memory_horizon",
            f"clip{length}_{state}",
            "gated_two_branch",
            True,
            True,
            clip_length=length,
            carry_state=state == "carry",
        )
        for length in (8, 16, 32)
        for state in ("reset", "carry")
    ]
    variants.extend(
        (
            BenchmarkVariant(
                "memory_horizon",
                "clip16_reset_burn4",
                "gated_two_branch",
                True,
                True,
                clip_length=16,
                burn_in_frames=4,
            ),
            BenchmarkVariant(
                "memory_horizon",
                "clip16_carry_burn4",
                "gated_two_branch",
                True,
                True,
                clip_length=16,
                carry_state=True,
                burn_in_frames=4,
            ),
        )
    )
    return tuple(variants)


ALL_VARIANTS = (*contribution_variants(), *memory_variants())
VARIANTS_BY_KEY = {(variant.suite, variant.name): variant for variant in ALL_VARIANTS}


def selected_suites(value: str) -> tuple[str, ...]:
    if value == "all":
        return ("contribution", "memory_horizon")
    return (value,)


def variants_for_suites(suites: tuple[str, ...]) -> tuple[BenchmarkVariant, ...]:
    return tuple(variant for variant in ALL_VARIANTS if variant.suite in suites)


def run_dir(runs_root: Path, scope: ScopeSpec, variant: BenchmarkVariant) -> Path:
    return runs_root / variant.suite / scope.name / variant.name


def result_dir(results_root: Path, scope: ScopeSpec, variant: BenchmarkVariant) -> Path:
    return results_root / variant.suite / scope.name / variant.name


def base_checkpoint(scope: ScopeSpec, variant: BenchmarkVariant) -> Path:
    return BASE_CHECKPOINTS[scope.name][variant.fusion_mode]


def training_complete(directory: Path, epochs: int) -> bool:
    history_path = directory / "history.json"
    final_checkpoint = directory / f"epoch_{epochs:03d}.pt"
    if not history_path.exists() or not final_checkpoint.exists():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(history, list) and len(history) >= epochs


def training_command(
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant: BenchmarkVariant,
) -> tuple[str, ...]:
    if variant.static_checkpoint:
        raise ValueError("Static variants do not have training commands.")
    recurrence = variant_for_name(args.recurrence_variant)
    batch_size = 1 if variant.suite == "memory_horizon" else args.batch_size
    grad_accum = (
        max(1, args.memory_effective_frames // variant.clip_length)
        if variant.suite == "memory_horizon"
        else args.grad_accum_steps
    )
    command = [
        sys.executable,
        "-m",
        "src.training.recurrent_embedding_detector",
        "--root",
        str(args.root),
        "--representation",
        "event_frame_voxel_grid",
        "--fusion-mode",
        variant.fusion_mode,
        "--num-bins",
        "3",
        "--time-window-us",
        "50000",
        "--model-width",
        "32",
        "--embedding-hidden-dim",
        "128",
        "--embedding-dim",
        "128" if variant.embedding else "0",
        "--embedding-head-type",
        "dense",
        "--no-recurrent-embedding",
        "--identity-ce-weight",
        "1.0" if variant.embedding else "0.0",
        "--triplet-weight",
        "1.0" if variant.embedding else "0.0",
        "--triplet-margin",
        "0.3",
        "--clip-length",
        str(variant.clip_length),
        "--clip-stride",
        str(variant.clip_length),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(grad_accum),
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
        "--initial-detector-checkpoint",
        str(base_checkpoint(scope, variant)),
        "--freeze-detector",
        "--save-every-epoch",
        "--output-dir",
        str(args.runs_root / variant.suite / scope.name),
        "--run-name",
        variant.name,
    ]
    if variant.recurrence:
        command.extend(
            [
                "--temporal-recurrence-locations",
                recurrence.locations_arg,
                "--temporal-recurrence-type",
                recurrence.cell_type,
                "--temporal-recurrence-mode",
                recurrence.mode,
                "--train-temporal-adapters",
            ]
        )
    if variant.suite == "memory_horizon":
        command.append("--ordered-clips")
    if variant.carry_state:
        command.append("--carry-state-between-clips")
    if variant.burn_in_frames:
        command.extend(["--burn-in-frames", str(variant.burn_in_frames)])
    if scope.class_ids is not None:
        command.extend(["--class-ids", ",".join(map(str, scope.class_ids))])
        command.extend(["--num-classes", str(scope.num_classes)])
    if args.max_train_clips:
        command.extend(["--max-train-clips", str(args.max_train_clips)])
    if args.max_val_clips:
        command.extend(["--max-val-clips", str(args.max_val_clips)])
    if not args.overwrite:
        command.append("--resume")
    return tuple(command)


def worker_command(
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant: BenchmarkVariant,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "src.experiments.novelty_contribution_benchmark",
        "--worker-eval",
        "--worker-suite",
        variant.suite,
        "--worker-scope",
        scope.name,
        "--worker-variant",
        variant.name,
        "--root",
        str(args.root),
        "--runs-root",
        str(args.runs_root),
        "--results-root",
        str(args.results_root),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--max-detections",
        str(args.max_detections),
        "--recurrence-variant",
        args.recurrence_variant,
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.max_eval_frames:
        command.extend(["--max-eval-frames", str(args.max_eval_frames)])
    if args.max_embedding_pairs:
        command.extend(["--max-embedding-pairs", str(args.max_embedding_pairs)])
    if args.skip_embedding_metrics:
        command.append("--skip-embedding-metrics")
    return tuple(command)


def evaluate_command(
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant: BenchmarkVariant,
    checkpoint: Path,
    tag: str,
    mode: str,
    split: str,
    score: float,
    appearance: float,
    proximity: float,
) -> tuple[list[str], Path]:
    output_root = result_dir(args.results_root, scope, variant)
    checkpoint_stat = checkpoint.stat()
    fingerprint = f"{checkpoint_stat.st_size}_{checkpoint_stat.st_mtime_ns}"
    cache_root = (
        output_root
        / "_detection_cache"
        / (f"{tag}_{split}_score{threshold_label(score)}_{fingerprint}")
    )
    run_name = (
        f"{tag}_{mode}_{split}_score{threshold_label(score)}_"
        f"appearance{threshold_label(appearance)}_proximity{threshold_label(proximity)}"
    )
    summary = output_root / run_name / "metrics_summary.json"
    dataset_split = "train" if split == "val" else "test"
    sequences = (VAL_SEQUENCE,) if split == "val" else TEST_SEQUENCES
    command = [
        sys.executable,
        "-m",
        "src.evaluation.simple_detector_trackeval_cli",
        "--checkpoint",
        str(checkpoint),
        "--root",
        str(args.root),
        "--split",
        dataset_split,
        "--sequences",
        *sequences,
        "--device",
        args.device,
        "--score-threshold",
        str(score),
        "--max-detections",
        str(args.max_detections),
        "--tracker-backend",
        "boxmot_botsort",
        "--track-per-class",
        "--track-disable-cmc",
        "--input-normalisation",
        "component",
        "--detection-cache-root",
        str(cache_root),
        "--track-appearance-thresh",
        str(appearance),
        "--track-proximity-thresh",
        str(proximity),
        "--tracker-name",
        f"novelty_{variant.suite}_{variant.name}_{mode}",
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
    ]
    if mode == "reid":
        command.append("--track-with-reid")
    if scope.classes_to_eval is not None:
        command.extend(["--classes-to-eval", *scope.classes_to_eval])
    if args.max_eval_frames:
        command.extend(["--max-frames", str(args.max_eval_frames)])
    return command, summary


def run_evaluation(
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant: BenchmarkVariant,
    checkpoint: Path,
    tag: str,
    mode: str,
    split: str,
    score: float,
    appearance: float,
    proximity: float,
) -> tuple[Path, dict]:
    command, summary = evaluate_command(
        args, scope, variant, checkpoint, tag, mode, split, score, appearance, proximity
    )
    if not summary.exists() or args.overwrite:
        print("\n$ " + " ".join(command), flush=True)
        code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        if code:
            raise SystemExit(f"Evaluation failed with exit {code}: {' '.join(command)}")
    return summary, load_summary(summary)


def run_embedding_metrics(
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant: BenchmarkVariant,
    checkpoint: Path,
    split: str,
    score: float,
) -> tuple[Path, dict]:
    dataset_split = "train" if split == "val" else "test"
    sequences = (VAL_SEQUENCE,) if split == "val" else TEST_SEQUENCES
    output = result_dir(args.results_root, scope, variant) / f"embedding_metrics_{split}"
    summary = output / "embedding_metrics.json"
    command = [
        sys.executable,
        "-m",
        "src.evaluation.embedding_metrics_cli",
        "--checkpoint",
        str(checkpoint),
        "--root",
        str(args.root),
        "--split",
        dataset_split,
        "--sequences",
        ",".join(sequences),
        "--representation",
        "event_frame_voxel_grid",
        "--num-bins",
        "3",
        "--time-window-us",
        "50000",
        "--input-normalisation",
        "component",
        "--score-threshold",
        str(score),
        "--max-detections",
        str(args.max_detections),
        "--max-pairs-per-group",
        str(args.max_embedding_pairs),
        "--device",
        args.device,
        "--output-dir",
        str(output),
    ]
    if scope.class_ids is not None:
        command.extend(["--class-ids", ",".join(map(str, scope.class_ids))])
    if args.max_eval_frames:
        command.extend(["--max-frames", str(args.max_eval_frames)])
    if not summary.exists() or args.overwrite:
        print("\n$ " + " ".join(command), flush=True)
        code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        if code:
            raise SystemExit(f"Embedding metrics failed with exit {code}: {' '.join(command)}")
    return summary, load_summary(summary)


def evaluate_variant(
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant: BenchmarkVariant,
) -> int:
    if variant.static_checkpoint:
        checkpoint_candidates = [(0, base_checkpoint(scope, variant))]
    else:
        directory = run_dir(args.runs_root, scope, variant)
        checkpoint_candidates = [
            (epoch, directory / f"epoch_{epoch:03d}.pt")
            for epoch in range(1, args.epochs + 1)
            if (directory / f"epoch_{epoch:03d}.pt").exists()
        ]
        if len(checkpoint_candidates) != args.epochs:
            raise SystemExit(
                f"Expected {args.epochs} epoch checkpoints in {directory}, "
                f"found {len(checkpoint_candidates)}."
            )

    selection_mode = "reid" if variant.embedding else "motion"
    validation_candidates: list[tuple[object, Path, dict]] = []
    for epoch, checkpoint in checkpoint_candidates:
        tag = "pretrained" if variant.static_checkpoint else f"epoch_{epoch:03d}"
        path, payload = run_evaluation(
            args,
            scope,
            variant,
            checkpoint,
            tag,
            selection_mode,
            "val",
            CHECKPOINT_SELECTION_SCORE,
            DEFAULT_APPEARANCE,
            DEFAULT_PROXIMITY,
        )
        validation_candidates.append((epoch, path, payload))
    selected_epoch_obj, _, checkpoint_validation = best_candidate(validation_candidates)
    selected_epoch = int(selected_epoch_obj)
    if variant.static_checkpoint:
        selected_checkpoint = base_checkpoint(scope, variant)
        selected_tag = "pretrained"
    else:
        directory = run_dir(args.runs_root, scope, variant)
        source = directory / f"epoch_{selected_epoch:03d}.pt"
        selected_checkpoint = directory / "best_hota.pt"
        save_selected_checkpoint(source, selected_checkpoint, selected_epoch, checkpoint_validation)
        selected_tag = "best_hota"

    score_candidates: list[tuple[object, Path, dict]] = []
    for score in scope.scores:
        path, payload = run_evaluation(
            args,
            scope,
            variant,
            selected_checkpoint,
            selected_tag,
            "motion",
            "val",
            score,
            DEFAULT_APPEARANCE,
            DEFAULT_PROXIMITY,
        )
        score_candidates.append((score, path, payload))
    selected_score_obj, _, score_validation = best_candidate(score_candidates)
    selected_score = float(selected_score_obj)

    appearance = DEFAULT_APPEARANCE
    proximity = DEFAULT_PROXIMITY
    association_validation = None
    if variant.embedding:
        association_candidates: list[tuple[object, Path, dict]] = []
        for candidate_appearance, candidate_proximity in product(
            APPEARANCE_THRESHOLDS, PROXIMITY_THRESHOLDS
        ):
            path, payload = run_evaluation(
                args,
                scope,
                variant,
                selected_checkpoint,
                selected_tag,
                "reid",
                "val",
                selected_score,
                candidate_appearance,
                candidate_proximity,
            )
            association_candidates.append(
                ((candidate_appearance, candidate_proximity), path, payload)
            )
        thresholds, _, association_validation = best_candidate(association_candidates)
        appearance, proximity = map(float, thresholds)

    _, motion_test = run_evaluation(
        args,
        scope,
        variant,
        selected_checkpoint,
        selected_tag,
        "motion",
        "test",
        selected_score,
        DEFAULT_APPEARANCE,
        DEFAULT_PROXIMITY,
    )
    reid_test = None
    if variant.embedding:
        _, reid_test = run_evaluation(
            args,
            scope,
            variant,
            selected_checkpoint,
            selected_tag,
            "reid",
            "test",
            selected_score,
            appearance,
            proximity,
        )

    embedding_metrics = None
    if variant.embedding and not args.skip_embedding_metrics:
        _, embedding_val = run_embedding_metrics(
            args, scope, variant, selected_checkpoint, "val", selected_score
        )
        _, embedding_test = run_embedding_metrics(
            args, scope, variant, selected_checkpoint, "test", selected_score
        )
        embedding_metrics = {"val": embedding_val, "test": embedding_test}

    primary = reid_test if reid_test is not None else motion_test
    selected_epoch_training_stats = None
    if not variant.static_checkpoint:
        history = json.loads(
            (run_dir(args.runs_root, scope, variant) / "history.json").read_text(encoding="utf-8")
        )
        selected_epoch_training_stats = next(
            item["train"] for item in history if int(item["epoch"]) == selected_epoch
        )
    summary = {
        "scope": scope.name,
        "variant": asdict(variant),
        "recurrence_variant": args.recurrence_variant if variant.recurrence else None,
        "selection_policy": ["validation HOTA", "AssA", "IDF1"],
        "checkpoint_selection_mode": selection_mode,
        "checkpoint_selection_score": CHECKPOINT_SELECTION_SCORE,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_checkpoint),
        "checkpoint_validation_metrics": checkpoint_validation["aggregate"],
        "score_threshold": selected_score,
        "score_validation_metrics": score_validation["aggregate"],
        "appearance_threshold": appearance if variant.embedding else None,
        "proximity_threshold": proximity if variant.embedding else None,
        "association_validation_metrics": (
            association_validation["aggregate"] if association_validation is not None else None
        ),
        "motion_test_metrics": motion_test["aggregate"],
        "reid_test_metrics": reid_test["aggregate"] if reid_test is not None else None,
        "primary_test_mode": "reid" if variant.embedding else "motion",
        "primary_test_metrics": primary["aggregate"],
        "selected_epoch_training_stats": selected_epoch_training_stats,
        "embedding_metrics": embedding_metrics,
    }
    path = result_dir(args.results_root, scope, variant) / "variant_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved variant summary to {path}")
    return 0


def _delta_metrics(metrics: dict, baseline: dict) -> dict[str, float]:
    return {metric: float(metrics[metric]) - float(baseline[metric]) for metric in PRIMARY_METRICS}


def write_contribution_aggregate(args: argparse.Namespace) -> None:
    records: list[dict] = []
    for scope in SCOPES.values():
        scope_records = {
            variant.name: load_summary(
                result_dir(args.results_root, scope, variant) / "variant_summary.json"
            )
            for variant in contribution_variants()
        }
        baseline = scope_records["baseline"]["primary_test_metrics"]
        individual_names = ("gating_only", "embedding_only", "recurrence_only")
        for name, record in scope_records.items():
            metrics = record["primary_test_metrics"]
            record["delta_vs_baseline"] = _delta_metrics(metrics, baseline)
            record["ids_reduction_vs_baseline"] = int(baseline["IDS"]) - int(metrics["IDS"])
            records.append(record)
        combined_delta = scope_records["all_modules"]["delta_vs_baseline"]
        additive_delta = {
            metric: sum(
                scope_records[name]["delta_vs_baseline"][metric] for name in individual_names
            )
            for metric in PRIMARY_METRICS
        }
        scope_records["all_modules"]["interaction_residual"] = {
            metric: combined_delta[metric] - additive_delta[metric] for metric in PRIMARY_METRICS
        }
    output_root = args.results_root / "contribution"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "benchmark_summary.json").write_text(
        json.dumps({"records": records}, indent=2), encoding="utf-8"
    )
    fieldnames = [
        "scope",
        "variant",
        "primary_test_mode",
        "selected_epoch",
        "score_threshold",
        "appearance_threshold",
        "proximity_threshold",
        *PRIMARY_METRICS,
        "IDS",
        "FP",
        "FN",
        *(f"delta_{metric}" for metric in PRIMARY_METRICS),
        *(f"interaction_{metric}" for metric in PRIMARY_METRICS),
        "ids_reduction_vs_baseline",
    ]
    with (output_root / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record["primary_test_metrics"]
            writer.writerow(
                {
                    "scope": record["scope"],
                    "variant": record["variant"]["name"],
                    "primary_test_mode": record["primary_test_mode"],
                    "selected_epoch": record["selected_epoch"],
                    "score_threshold": record["score_threshold"],
                    "appearance_threshold": record["appearance_threshold"],
                    "proximity_threshold": record["proximity_threshold"],
                    **{key: metrics.get(key) for key in (*PRIMARY_METRICS, "IDS", "FP", "FN")},
                    **{f"delta_{key}": value for key, value in record["delta_vs_baseline"].items()},
                    **{
                        f"interaction_{key}": value
                        for key, value in record.get("interaction_residual", {}).items()
                    },
                    "ids_reduction_vs_baseline": record["ids_reduction_vs_baseline"],
                }
            )


def write_memory_aggregate(args: argparse.Namespace) -> None:
    records: list[dict] = []
    for scope in SCOPES.values():
        scope_records = [
            load_summary(result_dir(args.results_root, scope, variant) / "variant_summary.json")
            for variant in memory_variants()
        ]
        reference = next(
            record for record in scope_records if record["variant"]["name"] == "clip8_reset"
        )["primary_test_metrics"]
        for record in scope_records:
            record["delta_vs_clip8_reset"] = _delta_metrics(
                record["primary_test_metrics"], reference
            )
            records.append(record)
    output_root = args.results_root / "memory_horizon"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "benchmark_summary.json").write_text(
        json.dumps({"records": records}, indent=2), encoding="utf-8"
    )
    fieldnames = [
        "scope",
        "variant",
        "clip_length",
        "carry_state",
        "burn_in_frames",
        "selected_epoch",
        "score_threshold",
        "appearance_threshold",
        "proximity_threshold",
        *PRIMARY_METRICS,
        "IDS",
        *(f"delta_{metric}" for metric in PRIMARY_METRICS),
    ]
    with (output_root / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            variant = record["variant"]
            metrics = record["primary_test_metrics"]
            writer.writerow(
                {
                    "scope": record["scope"],
                    "variant": variant["name"],
                    "clip_length": variant["clip_length"],
                    "carry_state": variant["carry_state"],
                    "burn_in_frames": variant["burn_in_frames"],
                    "selected_epoch": record["selected_epoch"],
                    "score_threshold": record["score_threshold"],
                    "appearance_threshold": record["appearance_threshold"],
                    "proximity_threshold": record["proximity_threshold"],
                    **{key: metrics.get(key) for key in (*PRIMARY_METRICS, "IDS")},
                    **{
                        f"delta_{key}": value
                        for key, value in record["delta_vs_clip8_reset"].items()
                    },
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument(
        "--runs-root", type=Path, default=Path("runs/novelty_contribution_benchmark")
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/dsec_mot_novelty_contribution_benchmark")
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("runs/novelty_contribution_benchmark/logs")
    )
    parser.add_argument("--suite", choices=("contribution", "memory_horizon", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--memory-effective-frames", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-val-clips", type=int, default=0)
    parser.add_argument("--max-eval-frames", type=int, default=0)
    parser.add_argument("--max-embedding-pairs", type=int, default=200_000)
    parser.add_argument("--recurrence-variant", default=DEFAULT_RECURRENCE_VARIANT)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-embedding-metrics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-concurrent-benchmarks", action="store_true")
    parser.add_argument("--worker-eval", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-suite", choices=("contribution", "memory_horizon"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-scope", choices=tuple(SCOPES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant", help=argparse.SUPPRESS)
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.max_parallel <= 0:
        raise SystemExit("--epochs and --max-parallel must be positive.")
    if args.memory_effective_frames <= 0:
        raise SystemExit("--memory-effective-frames must be positive.")
    recurrence = variant_for_name(args.recurrence_variant)
    if not recurrence.locations:
        raise SystemExit("The recurrence variant must contain at least one placement.")
    for scope_checkpoints in BASE_CHECKPOINTS.values():
        for checkpoint in scope_checkpoints.values():
            if not checkpoint.exists() and not args.dry_run:
                raise SystemExit(f"Missing base checkpoint: {checkpoint}")


def guard_against_other_trainers(args: argparse.Namespace) -> None:
    if args.dry_run or args.allow_concurrent_benchmarks or args.worker_eval:
        return
    result = subprocess.run(
        ["pgrep", "-f", "src[.]training[.]recurrent_embedding_detector"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        raise SystemExit(
            "Another recurrent embedding trainer is active. Wait for it or pass "
            "--allow-concurrent-benchmarks."
        )


def orchestrate(args: argparse.Namespace) -> int:
    suites = selected_suites(args.suite)
    variants = variants_for_suites(suites)
    scopes = tuple(SCOPES.values())
    train_variants = tuple(variant for variant in variants if not variant.static_checkpoint)
    print(
        f"Novelty benchmark: suites={','.join(suites)}, scopes={len(scopes)}, "
        f"training runs={len(train_variants) * len(scopes)}, epochs={args.epochs}, "
        f"max_parallel={args.max_parallel}."
    )
    print(
        f"Recurrence={args.recurrence_variant}; checkpoint, detector score and ReID thresholds "
        "are selected only on validation HOTA (AssA and IDF1 tie-breakers)."
    )
    if not args.skip_train:
        jobs: list[Job] = []
        for scope in scopes:
            for variant in train_variants:
                directory = run_dir(args.runs_root, scope, variant)
                if training_complete(directory, args.epochs) and not args.overwrite:
                    print(f"Skipping completed training: {directory}")
                    continue
                jobs.append(
                    Job(
                        name=f"train_{scope.name}_{variant.suite}_{variant.name}",
                        command=training_command(args, scope, variant),
                        log_path=args.log_dir
                        / f"train_{scope.name}_{variant.suite}_{variant.name}.log",
                    )
                )
        run_parallel(jobs, args.max_parallel, args.dry_run)
    if not args.skip_eval:
        jobs = [
            Job(
                name=f"eval_{scope.name}_{variant.suite}_{variant.name}",
                command=worker_command(args, scope, variant),
                log_path=args.log_dir / f"eval_{scope.name}_{variant.suite}_{variant.name}.log",
            )
            for scope in scopes
            for variant in variants
        ]
        run_parallel(jobs, args.max_parallel, args.dry_run)
    if args.dry_run:
        print(
            f"Dry-run complete: {len(train_variants) * len(scopes)} train jobs and "
            f"{len(variants) * len(scopes)} evaluation workers in the full selected matrix."
        )
        return 0
    if "contribution" in suites and not args.skip_eval:
        write_contribution_aggregate(args)
    if "memory_horizon" in suites and not args.skip_eval:
        write_memory_aggregate(args)
    return 0


def main() -> int:
    args = parse_args()
    require_inputs(args)
    if args.worker_eval:
        key = (args.worker_suite, args.worker_variant)
        if args.worker_scope is None or None in key or key not in VARIANTS_BY_KEY:
            raise SystemExit("Evaluation worker received an invalid scope or variant.")
        return evaluate_variant(args, SCOPES[args.worker_scope], VARIANTS_BY_KEY[key])
    guard_against_other_trainers(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
