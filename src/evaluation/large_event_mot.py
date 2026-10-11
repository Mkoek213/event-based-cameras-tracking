"""BoT-SORT tracking with model embeddings, seven-class TrackEval, and validation selection.

Validation-only selection on DSEC-MOT ``val`` (``configs/large_event_mot/evaluation.json``),
followed by a separate frozen test evaluation:

1. ``probe``: for every fine-tuning validation point listed in
   ``<run>/finetune/validation_requests.json``, export the validation detections with
   embeddings (:mod:`src.evaluation.large_event_export`) and run the 9 probe trials
   (fixed BoT-SORT settings, score threshold 0.1 to 0.9). A point's probe score is its
   best trial (ties: AssA, IDF1, lower threshold). Once training has finished, the 3 best
   points (ties: probe AssA, probe IDF1, later step) are retained.
2. ``search``: a bounded ``reid`` BoT-SORT search per retained point on its cached
   export (fixed initial settings, seeded random sampling, local refinement every fourth
   trial; at most ``trials_per_candidate`` completed trials each), resumable from the
   trial logs. With ``--mode motion`` it runs the optional motion-only diagnostic on the
   selected point, which never enters selection.
3. ``select``: the completed ``reid`` trial with the highest seven-class validation
   HOTA (ties: AssA, IDF1, earlier step, canonical parameter hash), re-evaluated with full
   outputs into ``selected/``; writes ``selection.json``.
4. ``freeze``: an immutable, read-only freeze manifest (checkpoint, configurations,
   export identity, tracker parameters, BoxMOT and TrackEval versions, ground-truth
   hashes, sequence lists, aggregation rule, output directory).
5. ``test``: verifies every hash in the freeze manifest, exports the test sequences,
   tracks them with the frozen settings, and writes the final metrics once. No other
   command reads or writes test data.

Tracking: BoxMOT ``BotSort`` through :class:`src.evaluation.mot_trackers.BoxMotTracker`
with ``reid_model=None``, the exported embeddings as ``embs`` (``reid`` mode), class-aware
association (with ``isolated_classes``, one tracker per class, because BoxMOT's
``per_class`` mode shares BoT-SORT's lost tracks across classes), 20 frames/s, and
camera-motion compensation disabled (the event-only pipeline has no intensity image).
Every frame is fed to the tracker, including frames without detections. Boxes below the
trial's score threshold, and optionally boxes below 614.4 px², are dropped before tracking.

Metrics: TrackEval HOTA, CLEAR, and Identity at IoU 0.5 over all seven classes on every
image timestamp; the aggregate is ``COMBINED_SEQ`` / ``cls_comb_det_av`` with HOTA
averaged over its 19 alpha thresholds, stored as fractions. Classes without ground truth
in a sequence or split are reported as ``no_gt`` with their FP count instead of a score.
Final test outputs add the literature-compatible variants over car, pedestrian, bus, and
truck: L1 (class-aware) and L2 (classes collapsed).

Examples::

    python -m src.evaluation.large_event_mot probe \\
        --requests <run>/finetune/validation_requests.json \\
        --output-root <results>/validation --manifest-root <manifests> --cache-root <cache> \\
        --rvt-root <rvt> --dsec-root <dsec_mot> --device cpu
    python -m src.evaluation.large_event_mot search --output-root <results>/validation --workers 3
    python -m src.evaluation.large_event_mot select --output-root <results>/validation
    python -m src.evaluation.large_event_mot freeze --output-root <results>/validation \\
        --freeze <results>/freeze.json --final-output-dir <results>/final
    python -m src.evaluation.large_event_mot test --freeze <results>/freeze.json \\
        --manifest-root <manifests> --cache-root <cache> --rvt-root <rvt> --dsec-root <dsec_mot>
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import json
import math
import multiprocessing
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.data.large_event_schema import read_json, settings_hash, write_json_atomic
from src.evaluation.detection_export import load_image_timestamps
from src.evaluation.large_event_export import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    ExportError,
    export_sequences,
    load_detector,
    load_evaluation_config,
    load_export,
    shared_identity,
)
from src.evaluation.mot_trackers import BoxMotTracker, DetectionObservation, TrackingConfig
from src.evaluation.trackeval_adapter import (
    TRACKEVAL_CLASS_ID_TO_NAME,
    export_trackeval_bundle,
    run_trackeval,
    summarise_trackeval_results,
    write_summary_csv,
)
from src.experiments import tracker_tuning
from src.experiments.tracker_tuning import append_history, best_trial, metric_key, read_history
from src.experiments.tracker_tuning_runtime import digest, file_hash, initialize_worker

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "manifest.json"
PROBE = "probe.json"
SELECTION = "selection.json"
STOP_FILE = "STOP.request"
TRACKER_NAME = "large_event_mot"
DSEC_DIRECTORY = {"train": "train", "val": "train", "test": "test"}
METRICS = ("HOTA", "AssA", "DetA", "MOTA", "IDF1", "IDS", "FP", "FN")
RATIO_METRICS = ("HOTA", "AssA", "DetA", "MOTA", "IDF1")
MODES = ("reid", "motion")


class EvaluationError(RuntimeError):
    """The request violates the validation/test protocol or an artifact is inconsistent."""


# --------------------------------------------------------------------------- context


@dataclass
class Context:
    evaluation_config: dict[str, Any]
    data_config: dict[str, Any]
    dsec_root: Path
    trackeval_root: Path
    work_root: Path
    frozen: bool = False

    @property
    def classes(self) -> list[str]:
        return list(self.evaluation_config["classes"])

    def split_sequences(self, split: str) -> list[str]:
        return list(self.data_config[self.evaluation_config["dataset"]]["splits"][split])


def check_search_space(evaluation_config: Mapping[str, Any]) -> None:
    """The search space is the small-model tuning space plus the binary minimum-area option."""

    search = evaluation_config["search"]
    small_grids = {key: value for key, value in search["grids"].items() if key != "min_area"}
    small_base = {key: value for key, value in search["base"].items() if key != "min_area"}
    if small_grids != tracker_tuning.GRIDS or small_base != tracker_tuning.BASE_PARAMS:
        raise EvaluationError("the search space must equal the small-model tuning space")
    if search["grids"]["min_area"] != [False, True]:
        raise EvaluationError("the minimum-area option must be the binary grid [false, true]")


def build_context(args: argparse.Namespace, work_root: Path) -> Context:
    evaluation_config = load_evaluation_config(args.evaluation_config)
    data_config = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    names = list(TRACKEVAL_CLASS_ID_TO_NAME.values())
    if evaluation_config["classes"] != names:
        raise EvaluationError(f"evaluation classes must be the TrackEval classes {names}")
    splits = data_config[evaluation_config["dataset"]]["splits"]
    if set(splits["val"]) & set(splits["test"]) or set(splits["train"]) & set(splits["test"]):
        raise EvaluationError("validation or training sequences overlap the test split")
    check_search_space(evaluation_config)
    tracker = evaluation_config["tracker"]
    if tracker["backend"] != "boxmot_botsort" or tracker["cmc_method"] is not None:
        raise EvaluationError("the frozen tracker is BoxMOT BoT-SORT without CMC")
    if int(tracker["frame_rate"]) != 20 or not tracker["per_class"]:
        raise EvaluationError("the frozen tracker runs class-aware at 20 frames/s")
    return Context(
        evaluation_config=evaluation_config,
        data_config=data_config,
        dsec_root=Path(args.dsec_root),
        trackeval_root=Path(args.trackeval_root),
        work_root=work_root,
    )


def file_sha256(path: str | Path) -> str:
    return file_hash(Path(path))


def git_commit(root: Path) -> Optional[str]:
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.stdout.strip()


def boxmot_version() -> Optional[str]:
    try:
        return importlib.metadata.version("boxmot")
    except importlib.metadata.PackageNotFoundError:
        return None


def ground_truth_files(ctx: Context, split: str, sequence: str) -> dict[str, dict[str, str]]:
    directory = DSEC_DIRECTORY[split]
    files = {
        "annotations": Path("annotations") / directory / f"{sequence}.txt",
        "image_timestamps": Path(directory) / sequence / f"{sequence}_image_timestamps.txt",
    }
    return {
        name: {"path": str(path), "sha256": file_sha256(ctx.dsec_root / path)}
        for name, path in files.items()
    }


def check_alignment(arrays: Mapping[str, np.ndarray], ctx: Context, split: str, sequence: str):
    """The export must cover exactly the image timestamps that TrackEval evaluates."""

    path = ctx.dsec_root / DSEC_DIRECTORY[split] / sequence / f"{sequence}_image_timestamps.txt"
    expected = np.asarray(load_image_timestamps(path), np.int64)
    if not np.array_equal(arrays["timestamps"], expected):
        raise EvaluationError(
            f"{sequence}: export steps ({arrays['timestamps'].size}) differ from the "
            f"{expected.size} image timestamps of {path}"
        )


# --------------------------------------------------------------------------- parameters


def canonical_params(params: Mapping[str, Any], mode: str, config: Mapping[str, Any]) -> dict:
    """Validated tracker parameters on the search grid (motion: neutral appearance)."""

    if mode not in MODES:
        raise EvaluationError(f"unknown association mode {mode!r}")
    search = config["search"]
    grids = search["grids"]
    if set(params) != set(grids):
        raise EvaluationError(f"tracker parameters {sorted(params)} differ from {sorted(grids)}")
    result = {key: params[key] for key in grids}
    if mode == "motion":
        result.update(search["diagnostic_motion"]["neutral"])
    for key, grid in grids.items():
        if result[key] not in grid:
            raise EvaluationError(f"{key}={result[key]!r} is not on the search grid")
    if result["low"] >= result["high"]:
        raise EvaluationError("the low threshold must lie below the high threshold")
    result["buffer"] = int(result["buffer"])
    result["min_area"] = bool(result["min_area"])
    for key in ("score", "high", "low", "match", "appearance", "proximity"):
        result[key] = float(result[key])
    return result


def tracking_config(params: Mapping[str, Any], mode: str, ctx: Context) -> TrackingConfig:
    return TrackingConfig(
        backend="boxmot_botsort",
        iou_threshold=params["match"],
        max_missed_frames=params["buffer"],
        high_threshold=params["high"],
        low_threshold=params["low"],
        with_reid=mode == "reid",
        appearance_thresh=params["appearance"],
        proximity_thresh=params["proximity"],
        per_class=True,
        num_classes=len(ctx.classes),
        cmc_method=ctx.evaluation_config["tracker"]["cmc_method"],
        isolated_classes=bool(ctx.evaluation_config["tracker"]["isolated_classes"]),
    )


def select_rows(rows: np.ndarray, params: Mapping[str, Any], min_area_px2: float) -> np.ndarray:
    """Rows at or above the score threshold (a prefix of the frame), optionally large ones."""

    keep = rows[:, 4] >= np.float32(params["score"])
    if params["min_area"]:
        area = (rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1])
        keep &= area >= np.float32(min_area_px2)
    return np.flatnonzero(keep)


# --------------------------------------------------------------------------- tracking


def track_sequence(
    arrays: Mapping[str, np.ndarray],
    params: Mapping[str, Any],
    mode: str,
    ctx: Context,
    output_path: Path,
) -> dict[str, Any]:
    """Track one exported sequence and write TrackEval rows; IDs are checked for uniqueness."""

    tracker = BoxMotTracker("boxmot_botsort", tracking_config(params, mode, ctx))
    min_area = float(ctx.evaluation_config["search"]["min_area_px2"])
    classes = len(ctx.classes)
    id_class: dict[int, int] = {}
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for frame, timestamp in enumerate(arrays["timestamps"]):
            start, stop = arrays["offsets"][frame : frame + 2]
            rows = arrays["rows"][start:stop]
            indices = select_rows(rows, params, min_area)
            detections = [
                DetectionObservation(
                    frame_index=frame,
                    timestamp=int(timestamp),
                    class_id=int(row[5]),
                    score=float(row[4]),
                    bbox_left=float(row[0]),
                    bbox_top=float(row[1]),
                    bbox_width=float(row[2] - row[0]),
                    bbox_height=float(row[3] - row[1]),
                )
                for row in rows[indices]
            ]
            vectors = arrays["embeddings"][start:stop][indices] if mode == "reid" else None
            records = tracker.update(detections, frame, int(timestamp), vectors)
            seen: set[int] = set()
            for record in records:
                if record.track_id in seen:
                    raise EvaluationError(f"frame {frame}: track ID {record.track_id} repeated")
                if id_class.setdefault(record.track_id, record.class_id) != record.class_id:
                    raise EvaluationError(f"track ID {record.track_id} changed its class")
                if not 0 <= record.class_id < classes:
                    raise EvaluationError(f"tracker returned class {record.class_id}")
                seen.add(record.track_id)
                stream.write(record.to_trackeval_line() + "\n")
                count += 1
    return {
        "frames": int(arrays["timestamps"].size),
        "tracks_written": count,
        "track_ids": len(id_class),
        "requested_track_buffer": params["buffer"],
        "effective_max_time_lost": int(tracker._tracker.max_time_lost),
        "frame_rate": 20,
        "per_class": True,
        "isolated_classes": bool(ctx.evaluation_config["tracker"]["isolated_classes"]),
        "cmc_method": None,
        "with_reid": mode == "reid",
    }


# --------------------------------------------------------------------------- TrackEval


def _gt_counts(gt_path: Path) -> dict[str, dict[str, int]]:
    boxes: dict[str, int] = {name: 0 for name in TRACKEVAL_CLASS_ID_TO_NAME.values()}
    tracks: dict[str, set[int]] = {name: set() for name in TRACKEVAL_CLASS_ID_TO_NAME.values()}
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        name = TRACKEVAL_CLASS_ID_TO_NAME[int(fields[7])]
        boxes[name] += 1
        tracks[name].add(int(fields[1]))
    return {name: {"gt_boxes": boxes[name], "gt_tracks": len(tracks[name])} for name in boxes}


def _class_metrics(results: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "HOTA": float(np.mean(results["HOTA"]["HOTA"])),
        "AssA": float(np.mean(results["HOTA"]["AssA"])),
        "DetA": float(np.mean(results["HOTA"]["DetA"])),
        "MOTA": float(results["CLEAR"]["MOTA"]),
        "IDF1": float(results["Identity"]["IDF1"]),
        "IDS": int(results["CLEAR"]["IDSW"]),
        "FP": int(results["CLEAR"]["CLR_FP"]),
        "FN": int(results["CLEAR"]["CLR_FN"]),
    }


def _mark_no_gt(metrics: dict[str, Any], counts: Mapping[str, int]) -> dict[str, Any]:
    metrics = {**metrics, **counts}
    if counts["gt_boxes"] == 0:
        metrics.update({key: None for key in RATIO_METRICS}, status="no_gt")
    else:
        metrics["status"] = "evaluated"
    return metrics


def _collapse_bundle(bundle: Mapping[str, Any], sequences: Sequence[str], keep: Sequence[str]):
    """Rewrite GT and tracker files of ``keep`` classes as one class with unique IDs."""

    keep_ids = {
        class_id for class_id, name in TRACKEVAL_CLASS_ID_TO_NAME.items() if name in set(keep)
    }
    split_name = f"{bundle['benchmark']}-{bundle['split']}"
    for sequence in sequences:
        for path in (
            Path(bundle["gt_folder"]) / split_name / sequence / "gt" / "gt.txt",
            Path(bundle["trackers_folder"])
            / split_name
            / bundle["tracker_name"]
            / "data"
            / f"{sequence}.txt",
        ):
            remap: dict[tuple[int, int], int] = {}
            lines = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                fields = line.split(",")
                class_id = int(fields[7])
                if class_id not in keep_ids:
                    continue
                key = (class_id, int(fields[1]))
                fields[1] = str(remap.setdefault(key, len(remap) + 1))
                fields[7] = "1"
                lines.append(",".join(fields))
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def trackeval_summary(
    tracks_dir: Path,
    split: str,
    sequences: Sequence[str],
    ctx: Context,
    output: Path,
    classes: Optional[Sequence[str]] = None,
    collapse: bool = False,
) -> tuple[dict[str, Any], Any]:
    """TrackEval summary with ground-truth counts and ``no_gt`` classes marked."""

    if split == "test" and not ctx.frozen:
        raise EvaluationError("test metrics are computed only by the frozen final evaluation")
    classes = list(classes or ctx.classes)
    bundle = export_trackeval_bundle(
        ctx.dsec_root,
        DSEC_DIRECTORY[split],
        list(sequences),
        TRACKER_NAME,
        tracks_dir,
        output / "trackeval",
    )
    if collapse:
        _collapse_bundle(bundle, sequences, classes)
        classes = ["car"]
    iou = float(ctx.evaluation_config["trackeval"]["iou_threshold"])
    raw, messages = run_trackeval(
        bundle, output / "trackeval" / "reports", ctx.trackeval_root, iou, classes_to_eval=classes
    )
    summary = summarise_trackeval_results(raw, TRACKER_NAME, iou, ctx.trackeval_root, classes)
    split_name = f"{bundle['benchmark']}-{bundle['split']}"
    totals = {name: {"gt_boxes": 0, "gt_tracks": 0} for name in classes}
    for sequence in sequences:
        counts = _gt_counts(Path(bundle["gt_folder"]) / split_name / sequence / "gt" / "gt.txt")
        entry = summary["per_sequence"][sequence]
        entry["by_class"] = {
            name: _mark_no_gt(entry["by_class"][name], counts[name]) for name in classes
        }
        for name in classes:
            totals[name]["gt_boxes"] += counts[name]["gt_boxes"]
            totals[name]["gt_tracks"] += counts[name]["gt_tracks"]
    combined = raw[next(iter(raw))][TRACKER_NAME]["COMBINED_SEQ"]
    summary["by_class"] = {
        name: _mark_no_gt(_class_metrics(combined[name]), totals[name]) for name in classes
    }
    summary["classes"] = classes
    summary["collapsed"] = collapse
    summary["sequences"] = list(sequences)
    summary["split"] = split
    summary["scale"] = "fraction"
    summary["aggregate_rule"] = ctx.evaluation_config["trackeval"]["aggregate"]
    return summary, messages


def write_class_csv(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "class", "status", "gt_boxes", "gt_tracks", *METRICS])
        scopes = [(name, entry["by_class"]) for name, entry in summary["per_sequence"].items()]
        for scope, by_class in [*scopes, ("COMBINED", summary["by_class"])]:
            for name, metrics in by_class.items():
                writer.writerow(
                    [scope, name, metrics["status"], metrics["gt_boxes"], metrics["gt_tracks"]]
                    + ["n/a" if metrics[key] is None else metrics[key] for key in METRICS]
                )


# --------------------------------------------------------------------------- trials


def run_trial(
    exports: Mapping[str, Mapping[str, np.ndarray]],
    params: Mapping[str, Any],
    mode: str,
    split: str,
    ctx: Context,
    destination: Optional[Path] = None,
) -> dict[str, Any]:
    """Track every sequence with ``params`` and evaluate; full outputs go to ``destination``."""

    params = canonical_params(params, mode, ctx.evaluation_config)
    sequences = list(exports)
    started = time.monotonic()
    ctx.work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ctx.work_root, prefix="trial-") as temporary:
        output = destination or Path(temporary)
        output.mkdir(parents=True, exist_ok=True)
        runs = {}
        with (output / "evaluation.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                for sequence in sequences:
                    runs[sequence] = track_sequence(
                        exports[sequence], params, mode, ctx, output / "tracks" / f"{sequence}.txt"
                    )
                summary, messages = trackeval_summary(
                    output / "tracks", split, sequences, ctx, output
                )
        aggregate = summary["aggregate"]
        if not all(math.isfinite(aggregate[key]) for key in METRICS):
            raise EvaluationError(f"TrackEval returned non-finite metrics: {aggregate}")
        if destination is not None:
            write_json_atomic(output / "metrics_summary.json", summary)
            write_summary_csv(summary, output / "metrics_summary.csv")
            write_class_csv(summary, output / "metrics_by_class.csv")
            write_json_atomic(output / "tracker_runs.json", runs)
            write_json_atomic(output / "trackeval_messages.json", messages)
            write_json_atomic(output / "params.json", {"mode": mode, "params": params})
    return {
        "params": params,
        "candidate_id": digest(params),
        "mode": mode,
        "split": split,
        "metrics": aggregate,
        "per_sequence": {name: entry["metrics"] for name, entry in summary["per_sequence"].items()},
        "by_class": summary["by_class"],
        "elapsed_s": time.monotonic() - started,
        "tracker_runs": runs,
    }


# --------------------------------------------------------------------------- search


def anchors(config: Mapping[str, Any], mode: str, extra: Sequence[Mapping[str, Any]]) -> list:
    """Fixed initial settings: the small-model search anchors (minimum-area filter off, then
    on for the base setting), followed by the probe winner."""

    candidates = [dict(item, min_area=False) for item in tracker_tuning.anchors({}, mode)]
    candidates.insert(1, dict(config["search"]["base"], min_area=True))
    candidates += [dict(item) for item in extra]
    unique: dict[str, dict] = {}
    for candidate in candidates:
        candidate = canonical_params(candidate, mode, config)
        unique.setdefault(digest(candidate), candidate)
    return list(unique.values())


def propose(
    history: Sequence[Mapping[str, Any]],
    mode: str,
    config: Mapping[str, Any],
    extra: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Next untested setting: anchors first, then seeded random draws with local refinement."""

    search = config["search"]
    grids = search["grids"]
    seen = {digest(record["params"]) for record in history}
    for candidate in anchors(config, mode, extra):
        if digest(candidate) not in seen:
            return candidate
    rng = random.Random(f"{search['seed']}:{len(history)}")
    neutral = set(search["diagnostic_motion"]["neutral"]) if mode == "motion" else set()
    fields = [key for key in grids if key not in neutral]
    for attempt in range(10000):
        candidate = {key: rng.choice(values) for key, values in grids.items()}
        if (
            len(history) % int(search["refine_every"]) == search["refine_every"] - 1
            and attempt < 100
        ):
            elite = sorted(history, key=metric_key, reverse=True)[: int(search["elite"])]
            candidate = dict(rng.choice(elite)["params"])
            for key in rng.sample(fields, k=min(2, len(fields))):
                grid = grids[key]
                index = grid.index(candidate[key])
                candidate[key] = grid[max(0, min(len(grid) - 1, index + rng.choice([-1, 1])))]
        if candidate["low"] >= candidate["high"]:
            continue
        candidate = canonical_params(candidate, mode, config)
        if digest(candidate) not in seen:
            return candidate
    raise EvaluationError("could not generate an untested tracker configuration")


def _load_exports(candidate: Mapping[str, Any], ctx: Context, split: str) -> dict[str, dict]:
    exports = {}
    for sequence in candidate["sequences"]:
        arrays, _ = load_export(candidate["export_dir"], sequence)
        check_alignment(arrays, ctx, split, sequence)
        exports[sequence] = arrays
    return exports


def search_candidate(job: Mapping[str, Any]) -> dict[str, Any]:
    """Run trials of one candidate until ``target`` completed trials, a stop, or the deadline."""

    ctx = Context(**job["context"])
    candidate = job["candidate"]
    path = Path(job["history"])
    exports = _load_exports(candidate, ctx, "val")
    history = read_history(path)
    while len(history) < job["target"]:
        if (Path(job["output_root"]) / STOP_FILE).exists() or time.time() > job["deadline"]:
            break
        params = propose(history, job["mode"], ctx.evaluation_config, job["extra_anchors"])
        try:
            record = run_trial(exports, params, job["mode"], "val", ctx)
        except Exception:
            write_json_atomic(
                path.with_name(f"{job['mode']}_failure.json"),
                {"params": params, "traceback": traceback.format_exc()},
            )
            raise
        record.update(point=candidate["point"], step=candidate["step"], trial_index=len(history))
        append_history(path, record)
        history.append(record)
    return {
        "point": candidate["point"],
        "mode": job["mode"],
        "trials": len(history),
        "target": job["target"],
        "best": best_trial(history) if history else None,
        "trial_seconds": sum(record["elapsed_s"] for record in history),
    }


# --------------------------------------------------------------------------- validation


def load_requests(path: Path, ctx: Context) -> dict[str, Any]:
    requests = read_json(path)
    selection = requests["selection"]
    dataset = ctx.evaluation_config["dataset"]
    split = ctx.evaluation_config["splits"]["selection"]
    if selection["dataset"] != dataset or selection["split"] != split or split != "val":
        raise EvaluationError(f"selection must use {dataset} val, got {selection}")
    if list(selection["sequences"]) != ctx.split_sequences("val"):
        raise EvaluationError("requested validation sequences differ from the data config")
    if set(selection["sequences"]) & set(ctx.split_sequences("test")):
        raise EvaluationError("validation requests contain test sequences")
    if requests.get("weights") != "ema":
        raise EvaluationError("validation candidates must hold EMA weights")
    return requests


def validation_manifest(requests_path: Path, requests: Mapping[str, Any], ctx: Context) -> dict:
    sequences = ctx.split_sequences("val")
    return {
        "format": "large_event_mot_validation",
        "version": 1,
        "requests": str(requests_path.resolve()),
        "run_identity_sha256": requests["run_identity_sha256"],
        "sequences": sequences,
        "test_sequences": ctx.split_sequences("test"),
        "evaluation_config": ctx.evaluation_config,
        "evaluation_config_sha256": settings_hash(ctx.evaluation_config),
        "data_config_sha256": settings_hash(ctx.data_config),
        "ground_truth": {seq: ground_truth_files(ctx, "val", seq) for seq in sequences},
        "dsec_root": str(ctx.dsec_root.resolve()),
        "trackeval_root": str(ctx.trackeval_root.resolve()),
        "trackeval_commit": git_commit(ctx.trackeval_root),
        "boxmot_version": boxmot_version(),
    }


def check_manifest(output_root: Path, manifest: Mapping[str, Any], create: bool) -> None:
    path = output_root / MANIFEST
    if not path.exists():
        if not create:
            raise EvaluationError(f"{path} is missing; run the probe stage first")
        write_json_atomic(path, manifest)
        return
    stored = read_json(path)
    if stored != dict(manifest):
        changed = sorted(
            key for key in set(stored) | set(manifest) if stored.get(key) != manifest.get(key)
        )
        raise EvaluationError(f"{path}: evaluation inputs changed ({changed}); use a new root")


def candidate_record(entry: Mapping[str, Any], run_dir: Path, output_root: Path, ctx: Context):
    return {
        "point": int(entry["point"]),
        "step": int(entry["step"]),
        "checkpoint": str(run_dir / entry["path"]),
        "sha256": entry["sha256"],
        "export_dir": str(output_root / "exports" / f"point_{entry['point']}_step_{entry['step']}"),
        "sequences": ctx.split_sequences("val"),
    }


def probe_candidate(candidate: Mapping[str, Any], ctx: Context) -> dict[str, Any]:
    probe = ctx.evaluation_config["probe"]
    exports = _load_exports(candidate, ctx, "val")
    trials = []
    for score in probe["score_thresholds"]:
        params = dict(probe["params"], score=score)
        record = run_trial(exports, params, probe["mode"], "val", ctx)
        record.update(point=candidate["point"], step=candidate["step"])
        trials.append(record)
    best = min(trials, key=lambda r: tuple(-v for v in metric_key(r)) + (r["params"]["score"],))
    return {**candidate, "trials": trials, "best": best}


def rank_points(probes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Probe score, then probe AssA, probe IDF1, and the later step."""

    return sorted(probes, key=lambda p: tuple(-v for v in metric_key(p["best"])) + (-p["step"],))


def command_probe(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    ctx = build_context(args, output_root / "work")
    requests_path = Path(args.requests)
    requests = load_requests(requests_path, ctx)
    check_manifest(output_root, validation_manifest(requests_path, requests, ctx), create=True)
    run_dir = requests_path.resolve().parent.parent
    candidates = [candidate_record(e, run_dir, output_root, ctx) for e in requests["candidates"]]
    sequences = loaded = None
    probes = []
    for candidate in candidates:
        probe_path = output_root / "probe" / f"point_{candidate['point']}.json"
        if probe_path.exists():
            stored = read_json(probe_path)
            if {key: stored[key] for key in candidate} != candidate:
                raise EvaluationError(f"{probe_path} belongs to a different candidate")
            probes.append(stored)
            continue
        if sequences is None:
            from src.data.large_event_dataset import load_sequences

            sequences = load_sequences(
                args.manifest_root, args.cache_root, ctx.data_config, "dsec_mot", ["val"]
            )
            if [item.sequence for item in sequences] != candidate["sequences"]:
                raise EvaluationError("manifest validation sequences differ from the request")
        loaded = load_detector(
            candidate["checkpoint"], "dsec_mot", args.device, args.rvt_root, candidate["sha256"]
        )
        export_sequences(
            loaded,
            sequences,
            ctx.data_config,
            ctx.evaluation_config,
            candidate["export_dir"],
            args.device,
        )
        del loaded
        print(f"probe point {candidate['point']} (step {candidate['step']})", flush=True)
        result = probe_candidate(candidate, ctx)
        write_json_atomic(probe_path, result)
        probes.append(result)
        best = result["best"]
        print(
            f"point {candidate['point']}: probe HOTA {best['metrics']['HOTA']:.6f} "
            f"at score {best['params']['score']}",
            flush=True,
        )
    complete = bool(requests.get("finished")) and len(probes) == len(requests["candidates"])
    ranking = rank_points(probes)
    retained = int(ctx.evaluation_config["probe"]["retained_candidates"])
    summary = {
        "complete": complete,
        "training_finished": bool(requests.get("finished")),
        "points": [
            {
                **{k: p[k] for k in ("point", "step", "checkpoint", "sha256")},
                "best": {k: p["best"][k] for k in ("params", "metrics")},
            }
            for p in ranking
        ],
        "retained": [
            {
                key: p[key]
                for key in ("point", "step", "checkpoint", "sha256", "export_dir", "sequences")
            }
            | {"probe_params": p["best"]["params"]}
            for p in ranking[:retained]
        ]
        if complete
        else [],
        "policy": ctx.evaluation_config["probe"],
    }
    write_json_atomic(output_root / PROBE, summary)
    print(f"probe: {len(probes)} points, complete={complete}", flush=True)
    return 0


def _job(
    ctx: Context,
    output_root: Path,
    candidate: Mapping[str, Any],
    mode: str,
    target: int,
    deadline: float,
    extra: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "context": {
            "evaluation_config": ctx.evaluation_config,
            "data_config": ctx.data_config,
            "dsec_root": ctx.dsec_root,
            "trackeval_root": ctx.trackeval_root,
            "work_root": ctx.work_root,
        },
        "candidate": dict(candidate),
        "history": str(output_root / "trials" / f"point_{candidate['point']}" / f"{mode}.jsonl"),
        "output_root": str(output_root),
        "mode": mode,
        "target": target,
        "deadline": deadline,
        "extra_anchors": list(extra),
    }


def _verified_probe(output_root: Path, ctx: Context) -> dict[str, Any]:
    probe = read_json(output_root / PROBE)
    if not probe["complete"] or not probe["retained"]:
        raise EvaluationError(
            "the probe stage is incomplete (training unfinished or points missing)"
        )
    for candidate in probe["retained"]:
        if file_sha256(candidate["checkpoint"]) != candidate["sha256"]:
            raise EvaluationError(f"{candidate['checkpoint']}: checkpoint changed since the probe")
    return probe


def command_search(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    manifest = read_json(output_root / MANIFEST)
    ctx = build_context(args, output_root / "work")
    check_manifest(output_root, {**manifest, **_current_inputs(ctx, manifest)}, create=False)
    probe = _verified_probe(output_root, ctx)
    search = ctx.evaluation_config["search"]
    if args.mode == "reid":
        candidates = probe["retained"]
        limit = int(search["trials_per_candidate"])
    else:
        if not search["diagnostic_motion"]["enabled"]:
            raise EvaluationError("the motion diagnostic is disabled in the configuration")
        selection = read_json(output_root / SELECTION)
        candidates = [c for c in probe["retained"] if c["point"] == selection["winner"]["point"]]
        limit = int(search["diagnostic_motion"]["trials"])
    target = limit if args.max_trials is None else int(args.max_trials)
    if not 0 < target <= limit:
        raise EvaluationError(f"--max-trials must lie in [1, {limit}]")
    deadline = time.time() + 3600.0 * args.max_wall_hours if args.max_wall_hours else math.inf
    extra = {c["point"]: [c["probe_params"]] if args.mode == "reid" else [] for c in candidates}
    jobs = [
        _job(ctx, output_root, c, args.mode, target, deadline, extra[c["point"]])
        for c in candidates
    ]
    results = []
    if args.workers <= 1:
        results = [search_candidate(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)),
            mp_context=context,
            initializer=initialize_worker,
            initargs=(args.threads,),
        ) as pool:
            futures = [pool.submit(search_candidate, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    for result in sorted(results, key=lambda r: r["point"]):
        best = result["best"]
        print(
            f"point {result['point']} {result['mode']}: {result['trials']}/{result['target']} "
            f"trials, best HOTA {best['metrics']['HOTA'] if best else None}",
            flush=True,
        )
    # 0 only once the full budget is reached; a stop below it (limit, deadline, STOP) is a pause.
    return 0 if all(r["trials"] >= limit for r in results) else 3


def _current_inputs(ctx: Context, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of the manifest that later stages recompute from the current inputs."""

    return {
        "evaluation_config": ctx.evaluation_config,
        "evaluation_config_sha256": settings_hash(ctx.evaluation_config),
        "data_config_sha256": settings_hash(ctx.data_config),
        "ground_truth": {seq: ground_truth_files(ctx, "val", seq) for seq in manifest["sequences"]},
        "trackeval_commit": git_commit(ctx.trackeval_root),
        "boxmot_version": boxmot_version(),
    }


def command_select(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    manifest = read_json(output_root / MANIFEST)
    ctx = build_context(args, output_root / "work")
    check_manifest(output_root, {**manifest, **_current_inputs(ctx, manifest)}, create=False)
    if (output_root / SELECTION).exists():
        raise EvaluationError(f"{output_root / SELECTION} exists; selection is written once")
    probe = _verified_probe(output_root, ctx)
    limit = int(ctx.evaluation_config["search"]["trials_per_candidate"])
    candidates, incomplete = [], []
    for candidate in probe["retained"]:
        history = read_history(
            output_root / "trials" / f"point_{candidate['point']}" / "reid.jsonl"
        )
        if len(history) < limit:
            incomplete.append({"point": candidate["point"], "trials": len(history)})
        if not history:
            continue
        best = best_trial(history)
        candidates.append({**candidate, "trials": len(history), "best": best})
    if incomplete and not args.allow_incomplete:
        raise EvaluationError(f"search budget not reached: {incomplete} (see --allow-incomplete)")
    if not candidates:
        raise EvaluationError("no completed trials")
    winner = min(
        candidates,
        key=lambda c: (
            tuple(-v for v in metric_key(c["best"])) + (c["step"], digest(c["best"]["params"]))
        ),
    )
    exports = _load_exports(winner, ctx, "val")
    params, mode = winner["best"]["params"], winner["best"]["mode"]
    destination = output_root / "selected"
    if destination.exists():
        shutil.rmtree(destination)
    replay = run_trial(exports, params, mode, "val", ctx, destination=destination)
    if replay["metrics"] != winner["best"]["metrics"]:
        raise EvaluationError(
            f"re-evaluating the winner gave {replay['metrics']}, the search stored "
            f"{winner['best']['metrics']} (non-deterministic tracking)"
        )
    export_metadata = read_json(Path(winner["export_dir"]) / f"{winner['sequences'][0]}.json")
    selection = {
        "format": "large_event_mot_selection",
        "version": 1,
        "manifest_sha256": digest(read_json(output_root / MANIFEST)),
        "policy": ctx.evaluation_config["selection"],
        "winner": {
            **{key: winner[key] for key in ("point", "step", "checkpoint", "sha256", "export_dir")},
            "sequences": winner["sequences"],
            "mode": mode,
            "params": params,
            "metrics": winner["best"]["metrics"],
            "trials": winner["trials"],
            "export_identity": shared_identity(export_metadata["identity"]),
            "validation_outputs": str(destination),
        },
        "candidates": [
            {key: c[key] for key in ("point", "step", "trials")}
            | {"best_params": c["best"]["params"], "best_metrics": c["best"]["metrics"]}
            for c in candidates
        ],
        "incomplete": incomplete,
        "test_outputs_read": False,
    }
    write_json_atomic(output_root / SELECTION, selection)
    print(
        f"selected point {winner['point']} (step {winner['step']}): validation HOTA "
        f"{winner['best']['metrics']['HOTA']:.6f}",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------- freeze and test


def command_freeze(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    manifest = read_json(output_root / MANIFEST)
    ctx = build_context(args, output_root / "work")
    check_manifest(output_root, {**manifest, **_current_inputs(ctx, manifest)}, create=False)
    selection = read_json(output_root / SELECTION)
    winner = selection["winner"]
    if file_sha256(winner["checkpoint"]) != winner["sha256"]:
        raise EvaluationError(f"{winner['checkpoint']}: checkpoint changed since selection")
    test_sequences = ctx.split_sequences("test")
    freeze = {
        "format": "large_event_mot_freeze",
        "version": 1,
        "created_unix": time.time(),
        "selection": {
            "path": str((output_root / SELECTION).resolve()),
            "sha256": file_sha256(output_root / SELECTION),
        },
        "checkpoint": {key: winner[key] for key in ("checkpoint", "sha256", "point", "step")},
        "export_identity": winner["export_identity"],
        "tracker": {
            "mode": winner["mode"],
            "params": winner["params"],
            **ctx.evaluation_config["tracker"],
        },
        "validation_metrics": winner["metrics"],
        "evaluation_config": ctx.evaluation_config,
        "evaluation_config_sha256": settings_hash(ctx.evaluation_config),
        "data_config_sha256": settings_hash(ctx.data_config),
        "boxmot_version": boxmot_version(),
        "trackeval_commit": git_commit(ctx.trackeval_root),
        "sequences": {"validation": manifest["sequences"], "test": test_sequences},
        "ground_truth": {
            "validation": manifest["ground_truth"],
            "test": {seq: ground_truth_files(ctx, "test", seq) for seq in test_sequences},
        },
        "aggregate_rule": ctx.evaluation_config["trackeval"]["aggregate"],
        "output_dir": str(Path(args.final_output_dir).resolve()),
    }
    path = Path(args.freeze)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        raise EvaluationError(f"{path} exists; a freeze manifest is immutable") from None
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"freeze manifest {path} (SHA-256 {file_sha256(path)})", flush=True)
    return 0


def verify_freeze(freeze: Mapping[str, Any], ctx: Context) -> None:
    """Every frozen input must be byte-identical to the current one."""

    problems = []
    checkpoint = freeze["checkpoint"]
    if file_sha256(checkpoint["checkpoint"]) != checkpoint["sha256"]:
        problems.append("checkpoint SHA-256")
    if settings_hash(ctx.evaluation_config) != freeze["evaluation_config_sha256"]:
        problems.append("evaluation config")
    if settings_hash(ctx.data_config) != freeze["data_config_sha256"]:
        problems.append("data config")
    if boxmot_version() != freeze["boxmot_version"]:
        problems.append(f"BoxMOT {boxmot_version()} != {freeze['boxmot_version']}")
    if git_commit(ctx.trackeval_root) != freeze["trackeval_commit"]:
        problems.append("TrackEval commit")
    if ctx.split_sequences("test") != freeze["sequences"]["test"]:
        problems.append("test sequence list")
    for seq, files in freeze["ground_truth"]["test"].items():
        if ground_truth_files(ctx, "test", seq) != files:
            problems.append(f"{seq} ground truth")
    if problems:
        raise EvaluationError(f"freeze manifest does not match the current inputs: {problems}")


def command_test(args: argparse.Namespace) -> int:
    freeze_path = Path(args.freeze)
    freeze = read_json(freeze_path)
    if freeze.get("format") != "large_event_mot_freeze":
        raise EvaluationError(f"{freeze_path}: not a freeze manifest")
    output = Path(freeze["output_dir"])
    ctx = build_context(args, output / "work")
    verify_freeze(freeze, ctx)
    if (output / "metrics_summary.json").exists():
        raise EvaluationError(f"{output} already holds the final evaluation; it runs once")
    ctx.frozen = True
    from src.data.large_event_dataset import load_sequences

    sequences = load_sequences(
        args.manifest_root,
        args.cache_root,
        ctx.data_config,
        ctx.evaluation_config["dataset"],
        ["test"],
    )
    if [item.sequence for item in sequences] != freeze["sequences"]["test"]:
        raise EvaluationError("manifest test sequences differ from the freeze manifest")
    started = time.monotonic()
    checkpoint = freeze["checkpoint"]
    loaded = load_detector(
        checkpoint["checkpoint"], "dsec_mot", args.device, args.rvt_root, checkpoint["sha256"]
    )
    export_dir = output / "exports"
    metadata = export_sequences(
        loaded,
        sequences,
        ctx.data_config,
        ctx.evaluation_config,
        export_dir,
        args.device,
        allow_test=True,
    )
    for seq, record in metadata.items():
        if shared_identity(record["identity"]) != freeze["export_identity"]:
            raise EvaluationError(f"{seq}: test export settings differ from the frozen export")
    exports = {}
    for seq in freeze["sequences"]["test"]:
        arrays, _ = load_export(export_dir, seq)
        check_alignment(arrays, ctx, "test", seq)
        exports[seq] = arrays
    tracker = freeze["tracker"]
    record = run_trial(exports, tracker["params"], tracker["mode"], "test", ctx, destination=output)
    supplementary = {}
    literature = ctx.evaluation_config["supplementary"]["literature_classes"]
    for name, collapse in (("L1", False), ("L2", True)):
        target = output / "supplementary" / name
        target.mkdir(parents=True, exist_ok=True)
        with (target / "evaluation.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary, _ = trackeval_summary(
                    output / "tracks", "test", list(exports), ctx, target, literature, collapse
                )
        summary["label"] = "approximate literature protocol"
        summary["definition"] = ctx.evaluation_config["supplementary"]["variants"][name]
        write_json_atomic(target / "metrics_summary.json", summary)
        write_summary_csv(summary, target / "metrics_summary.csv")
        supplementary[name] = summary["aggregate"]
    write_json_atomic(
        output / "test_run.json",
        {
            "freeze": {"path": str(freeze_path.resolve()), "sha256": file_sha256(freeze_path)},
            "metrics": record["metrics"],
            "supplementary": supplementary,
            "exports": metadata,
            "tracking_and_trackeval_s": record["elapsed_s"],
            "total_s": time.monotonic() - started,
            "parameters": loaded.payload["metadata"].get("parameters"),
            "device": args.device,
        },
    )
    print(f"final test HOTA {record['metrics']['HOTA']:.6f} written to {output}", flush=True)
    return 0


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dsec-root", default="data/datasets/dsec_mot")
        sub.add_argument("--trackeval-root", default="external/TrackEval")
        sub.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
        sub.add_argument("--evaluation-config", default=str(DEFAULT_EVALUATION_CONFIG))

    def model(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--manifest-root", required=True)
        sub.add_argument("--cache-root", required=True)
        sub.add_argument("--rvt-root", default=None, help="pinned RVT checkout (or RVT_ROOT)")
        sub.add_argument("--device", default="cpu")

    probe = commands.add_parser("probe", help="export validation points and run the probes")
    probe.add_argument("--requests", required=True, help="finetune/validation_requests.json")
    probe.add_argument("--output-root", required=True)
    common(probe)
    model(probe)

    search = commands.add_parser("search", help="bounded BoT-SORT search on retained points")
    search.add_argument("--output-root", required=True)
    search.add_argument("--mode", default="reid", choices=MODES)
    search.add_argument("--max-trials", type=int, default=None, help="stop each point earlier")
    search.add_argument("--max-wall-hours", type=float, default=None)
    search.add_argument("--workers", type=int, default=1)
    search.add_argument("--threads", type=int, default=1, help="BLAS threads per worker")
    common(search)

    select = commands.add_parser("select", help="write selection.json from completed trials")
    select.add_argument("--output-root", required=True)
    select.add_argument("--allow-incomplete", action="store_true")
    common(select)

    freeze = commands.add_parser("freeze", help="write the immutable freeze manifest")
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--freeze", required=True)
    freeze.add_argument("--final-output-dir", required=True)
    common(freeze)

    test = commands.add_parser("test", help="frozen final evaluation on the test split")
    test.add_argument("--freeze", required=True)
    common(test)
    model(test)
    return parser.parse_args(argv)


COMMANDS = {
    "probe": command_probe,
    "search": command_search,
    "select": command_select,
    "freeze": command_freeze,
    "test": command_test,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (EvaluationError, ExportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
