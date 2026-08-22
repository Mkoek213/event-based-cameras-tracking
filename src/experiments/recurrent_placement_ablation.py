#!/usr/bin/env python3
"""Run the staged recurrent-placement ablation on the small EF+VG detector."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import torch

from src.experiments.common import REPO_ROOT, threshold_label

VAL_SEQUENCE = "zurich_city_01_d"
TEST_SEQUENCES = ("interlaken_00_d", "zurich_city_00_b")
ALL_CLASS_SCORES = (0.10, 0.25, 0.50, 0.70, 0.90, 0.95)
CAR_ONLY_SCORES = (0.90, 0.95, 0.97, 0.99)
APPEARANCE_THRESHOLDS = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50)
PROXIMITY_THRESHOLDS = (0.30, 0.50, 0.70)
CHECKPOINT_SELECTION_SCORE = 0.90
DEFAULT_APPEARANCE = 0.25
DEFAULT_PROXIMITY = 0.50

BASE_CHECKPOINTS = {
    "all_classes": Path(
        "runs/dense_event_reid_embedding_frozen/warmstart/all_classes/"
        "event_frame_voxel_grid_bins3_w32_gated_two_branch_"
        "dense_d1_frozen_r0_warmstart/best_hota.pt"
    ),
    "car_only": Path(
        "runs/dense_event_reid_embedding_frozen/warmstart/car_only/"
        "event_frame_voxel_grid_bins3_w32_gated_two_branch_"
        "dense_d1_frozen_r0_warmstart/best_hota.pt"
    ),
}

PLACEMENTS = {
    "association": ("embedding",),
    "neck": ("neck",),
    "detection": ("detection_heads",),
    "middle": ("backbone_s4",),
    "early": ("backbone_s2",),
    "all": (
        "backbone_s2",
        "backbone_s4",
        "neck",
        "detection_heads",
        "embedding",
    ),
}
TYPE_LABELS = {"convgru": "gru", "convlstm": "lstm", "convrnn": "rnn"}
MODE_LABELS = {"residual": "res", "direct": "direct"}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    cell_type: str
    mode: str
    placement: str
    locations: tuple[str, ...]

    @property
    def locations_arg(self) -> str:
        return ",".join(self.locations)


@dataclass(frozen=True)
class ScopeSpec:
    name: str
    scores: tuple[float, ...]
    class_ids: tuple[int, ...] | None
    num_classes: int
    classes_to_eval: tuple[str, ...] | None


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    log_path: Path


SCOPES = {
    "all_classes": ScopeSpec(
        name="all_classes",
        scores=ALL_CLASS_SCORES,
        class_ids=None,
        num_classes=7,
        classes_to_eval=None,
    ),
    "car_only": ScopeSpec(
        name="car_only",
        scores=CAR_ONLY_SCORES,
        class_ids=(0,),
        num_classes=1,
        classes_to_eval=("car",),
    ),
}


def benchmark_variants() -> tuple[VariantSpec, ...]:
    """Return the fixed 24-run matrix in a memory-balanced launch order."""

    variants: list[VariantSpec] = []
    for cell_type in ("convgru", "convlstm", "convrnn"):
        for placement, locations in PLACEMENTS.items():
            variants.append(
                VariantSpec(
                    name=(f"{TYPE_LABELS[cell_type]}_{MODE_LABELS['residual']}_{placement}"),
                    cell_type=cell_type,
                    mode="residual",
                    placement=placement,
                    locations=locations,
                )
            )
    for placement, locations in PLACEMENTS.items():
        variants.append(
            VariantSpec(
                name=f"gru_{MODE_LABELS['direct']}_{placement}",
                cell_type="convgru",
                mode="direct",
                placement=placement,
                locations=locations,
            )
        )
    return tuple(variants)


VARIANTS = benchmark_variants()
VARIANTS_BY_NAME = {variant.name: variant for variant in VARIANTS}
CONTROL_VARIANT = VariantSpec(
    name="control",
    cell_type="none",
    mode="none",
    placement="none",
    locations=(),
)


def variant_for_name(name: str) -> VariantSpec:
    if name == CONTROL_VARIANT.name:
        return CONTROL_VARIANT
    return VARIANTS_BY_NAME[name]


def selected_variants(value: str | None) -> tuple[VariantSpec, ...]:
    if value is None:
        return VARIANTS
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(VARIANTS_BY_NAME))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")
    return tuple(VARIANTS_BY_NAME[name] for name in names)


def run_dir(runs_root: Path, scope: ScopeSpec, variant: VariantSpec) -> Path:
    return runs_root / scope.name / variant.name


def result_dir(results_root: Path, scope: ScopeSpec, variant_name: str) -> Path:
    return results_root / scope.name / variant_name


def training_complete(path: Path, epochs: int) -> bool:
    history_path = path / "history.json"
    checkpoint = path / f"epoch_{epochs:03d}.pt"
    if not history_path.exists() or not checkpoint.exists():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(history, list) and len(history) >= epochs


def training_command(
    python: str,
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant: VariantSpec,
) -> tuple[str, ...]:
    command = [
        python,
        "-m",
        "src.training.recurrent_embedding_detector",
        "--root",
        str(args.root),
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
        "128",
        "--embedding-head-type",
        "dense",
        "--no-recurrent-embedding",
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
        "--initial-model-checkpoint",
        str(BASE_CHECKPOINTS[scope.name]),
        "--freeze-detector",
        "--save-every-epoch",
        "--output-dir",
        str(args.runs_root / scope.name),
        "--run-name",
        variant.name,
    ]
    if not args.overwrite:
        command.append("--resume")
    if variant.locations:
        command.extend(
            [
                "--temporal-recurrence-locations",
                variant.locations_arg,
                "--temporal-recurrence-type",
                variant.cell_type,
                "--temporal-recurrence-mode",
                variant.mode,
                "--train-temporal-adapters",
            ]
        )
    if scope.class_ids is not None:
        command.extend(["--class-ids", ",".join(map(str, scope.class_ids))])
        command.extend(["--num-classes", str(scope.num_classes)])
    if args.max_train_clips:
        command.extend(["--max-train-clips", str(args.max_train_clips)])
    if args.max_val_clips:
        command.extend(["--max-val-clips", str(args.max_val_clips)])
    return tuple(command)


def worker_command(
    python: str,
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant_name: str,
) -> tuple[str, ...]:
    command = [
        python,
        "-m",
        "src.experiments.recurrent_placement_ablation",
        "--worker-eval",
        "--worker-scope",
        scope.name,
        "--worker-variant",
        variant_name,
        "--root",
        str(args.root),
        "--runs-root",
        str(args.runs_root),
        "--results-root",
        str(args.results_root),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--max-detections",
        str(args.max_detections),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.max_eval_frames:
        command.extend(["--max-eval-frames", str(args.max_eval_frames)])
    return tuple(command)


def print_command(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def run_parallel(jobs: list[Job], max_parallel: int, dry_run: bool) -> None:
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive.")
    if dry_run:
        for job in jobs:
            print(f"\n$ {print_command(job.command)}")
            print(f"log: {job.log_path}")
        return

    pending = list(jobs)
    active: list[tuple[Job, subprocess.Popen, object, float]] = []
    failed: list[tuple[str, int]] = []
    while pending or active:
        while pending and len(active) < max_parallel and not failed:
            job = pending.pop(0)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = job.log_path.open("a", encoding="utf-8")
            printable = print_command(job.command)
            handle.write(f"\n$ {printable}\n")
            handle.flush()
            process = subprocess.Popen(
                list(job.command),
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((job, process, handle, time.time()))
            print(f"Started {job.name} (pid={process.pid}); log: {job.log_path}")

        finished: list[int] = []
        for index, (job, process, handle, started) in enumerate(active):
            code = process.poll()
            if code is None:
                continue
            elapsed_h = (time.time() - started) / 3600.0
            handle.write(f"\nexit_code={code} elapsed_h={elapsed_h:.3f}\n")
            handle.close()
            finished.append(index)
            if code:
                failed.append((job.name, code))
                print(f"FAILED {job.name} (exit={code}); log: {job.log_path}")
            else:
                print(f"Finished {job.name} in {elapsed_h:.2f} h")
        for index in reversed(finished):
            active.pop(index)
        if failed and not active:
            break
        if active and not finished:
            time.sleep(2.0)

    if failed:
        details = ", ".join(f"{name}: exit {code}" for name, code in failed)
        raise SystemExit(f"Benchmark jobs failed: {details}")


def selection_key(payload: dict) -> tuple[float, float, float]:
    metrics = payload["aggregate"]
    return (
        float(metrics.get("HOTA", -1.0)),
        float(metrics.get("AssA", -1.0)),
        float(metrics.get("IDF1", -1.0)),
    )


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_command(
    args: argparse.Namespace,
    scope: ScopeSpec,
    variant_name: str,
    checkpoint: Path,
    tag: str,
    mode: str,
    split: str,
    score: float,
    appearance: float,
    proximity: float,
) -> tuple[list[str], Path]:
    output_root = result_dir(args.results_root, scope, variant_name)
    checkpoint_stat = checkpoint.stat()
    checkpoint_fingerprint = f"{checkpoint_stat.st_size}_{checkpoint_stat.st_mtime_ns}"
    detection_cache_root = (
        output_root
        / "_detection_cache"
        / f"{tag}_{split}_score{threshold_label(score)}_{checkpoint_fingerprint}"
    )
    run_name = (
        f"{tag}_{mode}_{split}_score{threshold_label(score)}_"
        f"appearance{threshold_label(appearance)}_"
        f"proximity{threshold_label(proximity)}"
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
        str(detection_cache_root),
        "--track-appearance-thresh",
        str(appearance),
        "--track-proximity-thresh",
        str(proximity),
        "--tracker-name",
        f"recurrent_ablation_{mode}",
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
    variant_name: str,
    checkpoint: Path,
    tag: str,
    mode: str,
    split: str,
    score: float,
    appearance: float,
    proximity: float,
) -> tuple[Path, dict]:
    command, summary = evaluate_command(
        args,
        scope,
        variant_name,
        checkpoint,
        tag,
        mode,
        split,
        score,
        appearance,
        proximity,
    )
    if not summary.exists() or args.overwrite:
        print(f"\n$ {print_command(command)}", flush=True)
        code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        if code:
            raise SystemExit(f"Evaluation failed with exit {code}: {print_command(command)}")
    return summary, load_summary(summary)


def best_candidate(candidates: list[tuple[object, Path, dict]]) -> tuple[object, Path, dict]:
    if not candidates:
        raise RuntimeError("No completed validation candidates.")
    return max(candidates, key=lambda item: selection_key(item[2]))


def save_selected_checkpoint(
    source: Path,
    target: Path,
    epoch: int,
    summary: dict,
) -> None:
    checkpoint = torch.load(source, map_location="cpu")
    checkpoint["selected_epoch"] = epoch
    checkpoint["selection_metric"] = "validation_hota"
    checkpoint["best_hota_selection"] = {
        key: summary["aggregate"][key] for key in ("HOTA", "AssA", "IDF1")
    }
    torch.save(checkpoint, target)


def evaluate_variant(args: argparse.Namespace, scope: ScopeSpec, variant_name: str) -> int:
    if variant_name == "reference":
        candidates = [(0, BASE_CHECKPOINTS[scope.name])]
    else:
        variant = variant_for_name(variant_name)
        directory = run_dir(args.runs_root, scope, variant)
        candidates = [
            (epoch, directory / f"epoch_{epoch:03d}.pt")
            for epoch in range(1, args.epochs + 1)
            if (directory / f"epoch_{epoch:03d}.pt").exists()
        ]
        if len(candidates) != args.epochs:
            raise SystemExit(
                f"Expected {args.epochs} epoch checkpoints in {directory}, found {len(candidates)}."
            )

    checkpoint_candidates: list[tuple[object, Path, dict]] = []
    for epoch, checkpoint in candidates:
        tag = "reference" if variant_name == "reference" else f"epoch_{epoch:03d}"
        path, payload = run_evaluation(
            args,
            scope,
            variant_name,
            checkpoint,
            tag,
            "reid",
            "val",
            CHECKPOINT_SELECTION_SCORE,
            DEFAULT_APPEARANCE,
            DEFAULT_PROXIMITY,
        )
        checkpoint_candidates.append((epoch, path, payload))
    selected_epoch_obj, _, checkpoint_summary = best_candidate(checkpoint_candidates)
    selected_epoch = int(selected_epoch_obj)

    if variant_name == "reference":
        selected_checkpoint = BASE_CHECKPOINTS[scope.name]
        selected_tag = "reference"
    else:
        source_checkpoint = (
            run_dir(args.runs_root, scope, variant_for_name(variant_name))
            / f"epoch_{selected_epoch:03d}.pt"
        )
        selected_checkpoint = source_checkpoint.parent / "best_hota.pt"
        save_selected_checkpoint(
            source_checkpoint,
            selected_checkpoint,
            selected_epoch,
            checkpoint_summary,
        )
        selected_tag = "best_hota"

    score_candidates: list[tuple[object, Path, dict]] = []
    for score in scope.scores:
        path, payload = run_evaluation(
            args,
            scope,
            variant_name,
            selected_checkpoint,
            selected_tag,
            "motion",
            "val",
            score,
            DEFAULT_APPEARANCE,
            DEFAULT_PROXIMITY,
        )
        score_candidates.append((score, path, payload))
    selected_score_obj, _, score_summary = best_candidate(score_candidates)
    selected_score = float(selected_score_obj)

    association_candidates: list[tuple[object, Path, dict]] = []
    for appearance, proximity in product(APPEARANCE_THRESHOLDS, PROXIMITY_THRESHOLDS):
        path, payload = run_evaluation(
            args,
            scope,
            variant_name,
            selected_checkpoint,
            selected_tag,
            "reid",
            "val",
            selected_score,
            appearance,
            proximity,
        )
        association_candidates.append(((appearance, proximity), path, payload))
    thresholds_obj, _, association_summary = best_candidate(association_candidates)
    selected_appearance, selected_proximity = thresholds_obj

    _, motion_test = run_evaluation(
        args,
        scope,
        variant_name,
        selected_checkpoint,
        selected_tag,
        "motion",
        "test",
        selected_score,
        DEFAULT_APPEARANCE,
        DEFAULT_PROXIMITY,
    )
    _, reid_test = run_evaluation(
        args,
        scope,
        variant_name,
        selected_checkpoint,
        selected_tag,
        "reid",
        "test",
        selected_score,
        float(selected_appearance),
        float(selected_proximity),
    )

    if variant_name == "reference":
        variant_payload = {
            "name": "reference",
            "locations": [],
            "cell_type": None,
            "mode": None,
            "placement": None,
        }
    else:
        variant_payload = asdict(variant_for_name(variant_name))
    summary_payload = {
        "scope": scope.name,
        "variant": variant_payload,
        "selection_policy": ["HOTA", "AssA", "IDF1"],
        "checkpoint_selection_score": CHECKPOINT_SELECTION_SCORE,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_checkpoint),
        "checkpoint_validation_metrics": checkpoint_summary["aggregate"],
        "score_threshold": selected_score,
        "score_validation_metrics": score_summary["aggregate"],
        "appearance_threshold": float(selected_appearance),
        "proximity_threshold": float(selected_proximity),
        "association_validation_metrics": association_summary["aggregate"],
        "motion_test_metrics": motion_test["aggregate"],
        "reid_test_metrics": reid_test["aggregate"],
    }
    path = result_dir(args.results_root, scope, variant_name) / "variant_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Saved variant summary to {path}")
    return 0


def top_car_variants(
    args: argparse.Namespace,
    variants: tuple[VariantSpec, ...],
) -> tuple[VariantSpec, ...]:
    ranked: list[tuple[tuple[float, float, float], VariantSpec]] = []
    scope = SCOPES["all_classes"]
    for variant in variants:
        path = result_dir(args.results_root, scope, variant.name) / "variant_summary.json"
        if not path.exists():
            raise SystemExit(f"Missing all-class summary for top-k selection: {path}")
        payload = load_summary(path)
        metrics = payload["association_validation_metrics"]
        ranked.append(
            (
                (
                    float(metrics["HOTA"]),
                    float(metrics["AssA"]),
                    float(metrics["IDF1"]),
                ),
                variant,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = tuple(variant for _, variant in ranked[: args.car_top_k])
    output = args.results_root / "car_only_selected_variants.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "selection_policy": ["all_classes validation HOTA", "AssA", "IDF1"],
                "top_k": args.car_top_k,
                "variants": [
                    {"rank": rank, **asdict(variant), "validation_key": list(key)}
                    for rank, (key, variant) in enumerate(ranked[: args.car_top_k], start=1)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Selected car-only variants: " + ", ".join(variant.name for variant in selected))
    return selected


def write_aggregate(args: argparse.Namespace) -> None:
    summaries = sorted(args.results_root.glob("*/*/variant_summary.json"))
    records = [load_summary(path) for path in summaries]
    output_json = args.results_root / "benchmark_summary.json"
    output_csv = args.results_root / "benchmark_summary.csv"
    output_json.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    fieldnames = [
        "scope",
        "variant",
        "cell_type",
        "mode",
        "placement",
        "locations",
        "selected_epoch",
        "score_threshold",
        "appearance_threshold",
        "proximity_threshold",
        "HOTA",
        "AssA",
        "DetA",
        "MOTA",
        "IDF1",
        "IDS",
        "FP",
        "FN",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            variant = record["variant"]
            metrics = record["reid_test_metrics"]
            writer.writerow(
                {
                    "scope": record["scope"],
                    "variant": variant["name"],
                    "cell_type": variant.get("cell_type"),
                    "mode": variant.get("mode"),
                    "placement": variant.get("placement"),
                    "locations": ",".join(variant.get("locations", [])),
                    "selected_epoch": record["selected_epoch"],
                    "score_threshold": record["score_threshold"],
                    "appearance_threshold": record["appearance_threshold"],
                    "proximity_threshold": record["proximity_threshold"],
                    **{key: metrics.get(key) for key in fieldnames[-8:]},
                }
            )
    print(f"Saved aggregate JSON to {output_json}")
    print(f"Saved aggregate CSV to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/recurrent_placement_ablation"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/dsec_mot_recurrent_placement_ablation"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs/recurrent_placement_ablation/logs"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--car-top-k", type=int, default=4)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-val-clips", type=int, default=0)
    parser.add_argument("--max-eval-frames", type=int, default=0)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-concurrent-benchmarks", action="store_true")
    parser.add_argument("--worker-eval", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-scope",
        choices=tuple(SCOPES),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-variant", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.max_parallel <= 0:
        raise SystemExit("--max-parallel must be positive.")
    if args.car_top_k <= 0:
        raise SystemExit("--car-top-k must be positive.")
    for scope, checkpoint in BASE_CHECKPOINTS.items():
        if not checkpoint.exists() and not args.dry_run:
            raise SystemExit(f"Missing {scope} base checkpoint: {checkpoint}")


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
            "Another recurrent embedding trainer is active. "
            "Wait for it or pass --allow-concurrent-benchmarks."
        )


def orchestrate(args: argparse.Namespace) -> int:
    variants = selected_variants(args.variants)
    if args.car_top_k > len(variants):
        raise SystemExit("--car-top-k cannot exceed the selected variant count.")
    python = sys.executable
    all_scope = SCOPES["all_classes"]
    car_scope = SCOPES["car_only"]

    print(
        f"Recurrent ablation: {len(variants)} recurrent all-class variants plus "
        f"one matched no-recurrence control, top-{args.car_top_k} recurrent "
        f"variants plus control repeated car-only, {args.epochs} epochs, "
        f"max_parallel={args.max_parallel}."
    )
    print(
        "Checkpoint selection uses validation HOTA at score=0.90; score, "
        "appearance and proximity thresholds are then selected on validation."
    )

    if not args.skip_train:
        jobs = []
        for variant in (CONTROL_VARIANT, *variants):
            directory = run_dir(args.runs_root, all_scope, variant)
            if training_complete(directory, args.epochs) and not args.overwrite:
                print(f"Skipping completed training: {directory}")
                continue
            jobs.append(
                Job(
                    name=f"train_all_classes_{variant.name}",
                    command=training_command(python, args, all_scope, variant),
                    log_path=args.log_dir / f"train_all_classes_{variant.name}.log",
                )
            )
        run_parallel(jobs, args.max_parallel, args.dry_run)

    if not args.skip_eval:
        eval_names = ("reference", "control", *(variant.name for variant in variants))
        jobs = [
            Job(
                name=f"eval_all_classes_{name}",
                command=worker_command(python, args, all_scope, name),
                log_path=args.log_dir / f"eval_all_classes_{name}.log",
            )
            for name in eval_names
        ]
        run_parallel(jobs, args.max_parallel, args.dry_run)

    if args.dry_run:
        print(
            f"\nAfter all-class validation, top-{args.car_top_k} variants are selected "
            "for car-only training and evaluation."
        )
        print(
            "Expected training count: "
            f"{len(variants) + 1} all-class + {args.car_top_k + 1} car-only = "
            f"{len(variants) + args.car_top_k + 2}."
        )
        return 0

    selected = top_car_variants(args, variants)

    if not args.skip_train:
        jobs = []
        for variant in (CONTROL_VARIANT, *selected):
            directory = run_dir(args.runs_root, car_scope, variant)
            if training_complete(directory, args.epochs) and not args.overwrite:
                print(f"Skipping completed training: {directory}")
                continue
            jobs.append(
                Job(
                    name=f"train_car_only_{variant.name}",
                    command=training_command(python, args, car_scope, variant),
                    log_path=args.log_dir / f"train_car_only_{variant.name}.log",
                )
            )
        run_parallel(jobs, args.max_parallel, dry_run=False)

    if not args.skip_eval:
        eval_names = ("reference", "control", *(variant.name for variant in selected))
        jobs = [
            Job(
                name=f"eval_car_only_{name}",
                command=worker_command(python, args, car_scope, name),
                log_path=args.log_dir / f"eval_car_only_{name}.log",
            )
            for name in eval_names
        ]
        run_parallel(jobs, args.max_parallel, dry_run=False)

    write_aggregate(args)
    return 0


def main() -> int:
    args = parse_args()
    require_inputs(args)
    if args.worker_eval:
        if args.worker_scope is None or args.worker_variant is None:
            raise SystemExit("Evaluation worker requires scope and variant.")
        valid_worker_names = {"reference", "control", *VARIANTS_BY_NAME}
        if args.worker_variant not in valid_worker_names:
            raise SystemExit(f"Unknown worker variant: {args.worker_variant}")
        return evaluate_variant(args, SCOPES[args.worker_scope], args.worker_variant)
    guard_against_other_trainers(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
