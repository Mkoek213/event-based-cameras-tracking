"""Inference cache and CPU evaluation for the fixed-checkpoint tracker search.

The cache keeps up to 100 NMS survivors *per class*. This is deliberate: the
legacy decoder applies its total limit in class order, before sorting by score.
Filtering a conventional, already limited low-score export is not equivalent.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

VAL_SEQUENCE = "zurich_city_01_d"
TEST_SEQUENCES = ("interlaken_00_d", "zurich_city_00_b")
SCORE_FLOOR = 0.05
MAX_DETECTIONS = 100
METRICS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDS", "FP", "FN")


def split_sequences(split: str, args: dict) -> list[str]:
    if split == "val":
        return [VAL_SEQUENCE]
    if split == "test":
        return list(TEST_SEQUENCES)
    if split != "train":
        raise ValueError(split)
    directory = Path(args["root"]) / "train"
    reserved = {VAL_SEQUENCE, *TEST_SEQUENCES}
    sequences = (
        sorted(
            path.name for path in directory.iterdir() if path.is_dir() and path.name not in reserved
        )
        if directory.is_dir()
        else []
    )
    if not sequences:
        raise ValueError(
            "No training sequences remain after excluding validation and test sequences"
        )
    return sequences


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def check_disk(path: Path, minimum_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 1024**3
    if free < minimum_gb:
        raise RuntimeError(
            f"Only {free:.2f} GiB free at {path}; need {minimum_gb:.2f} GiB. "
            "Free disk space and rerun the same command to resume."
        )


def select_indices(rows: np.ndarray, score: float, limit: int = MAX_DETECTIONS) -> np.ndarray:
    """Reproduce the existing class-first NMS budget after score filtering."""
    indices = np.flatnonzero(rows[:, 4] >= np.float32(score))
    ordered = sorted(indices, key=lambda i: (rows[i, 5], -rows[i, 4]))[:limit]
    return np.asarray(sorted(ordered, key=lambda i: -rows[i, 4]), dtype=np.int64)


def pack_frame(detections: list, embedding_dim: int) -> tuple[np.ndarray, np.ndarray]:
    # Preserve the NMS order for score ties, including across classes.
    counts: dict[int, int] = {}
    kept = []
    for detection in detections:
        count = counts.get(detection.class_id, 0)
        if count < MAX_DETECTIONS:
            kept.append(detection)
            counts[detection.class_id] = count + 1
    rows = np.asarray(
        [
            [
                d.bbox_left,
                d.bbox_top,
                d.bbox_left + d.bbox_width,
                d.bbox_top + d.bbox_height,
                d.score,
                d.class_id,
            ]
            for d in kept
        ],
        dtype=np.float32,
    ).reshape(-1, 6)
    embeddings = np.empty((len(kept), embedding_dim), dtype=np.float32)
    if embedding_dim:
        for index, detection in enumerate(kept):
            if detection.embedding is None:
                raise ValueError("Missing embedding in an embedding-head export")
            embeddings[index] = detection.embedding
    return rows, embeddings


def export_cache(entry: dict, split: str, args: dict) -> dict:
    """Export each sequence once, with recurrent state reset at its boundary."""
    import torch

    from src.data.pretrained_detection_dataset import ErosSnapshotStore
    from src.data.representations import BenchmarkRepresentation, representation_channel_splits
    from src.evaluation.detection_export import load_event_file, load_image_timestamps, read_events
    from src.evaluation.simple_detector_trackeval_cli import load_model
    from src.models.simple_detector import (
        attach_detection_embeddings,
        decode_dense_detections,
        detection_boxes_xyxy,
        normalise_event_tensor,
        normalise_representation_tensor,
    )

    root = Path(args["root"])
    cache_root = Path(args["output_root"]) / "cache" / entry["id"] / split
    sequences = split_sequences(split, args)
    dataset_split = "test" if split == "test" else "train"
    checkpoint_path = Path(entry["checkpoint"])
    if file_identity(checkpoint_path) != entry["checkpoint_identity"]:
        raise RuntimeError(f"Checkpoint changed: {checkpoint_path}")
    model = None
    device = torch.device(args["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu if intentional.")
    result = {}
    for sequence in sequences:
        check_disk(cache_root, args["min_free_gb"])
        seq_dir = root / dataset_split / sequence
        timestamps_path = seq_dir / f"{sequence}_image_timestamps.txt"
        events_path = seq_dir / "events_left/events.h5"
        path = cache_root / f"{sequence}.npz"
        metadata_path = path.with_suffix(".json")
        identity = {
            "entry": entry,
            "events": file_identity(events_path),
            "timestamps": file_identity(timestamps_path),
            "score_floor": SCORE_FLOOR,
            "nms_iou": 0.5,
            "stable_nms": True,
            "per_class_limit": MAX_DETECTIONS,
            "runtime": args["runtime_identity"],
        }
        if path.exists() and metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata["identity"] != identity or metadata["sha256"] != file_hash(path):
                raise RuntimeError(f"Invalid/stale detection cache: {path}")
            result[sequence] = metadata
            continue
        if model is None:
            torch.set_num_threads(args["threads"])
            model, checkpoint = load_model(checkpoint_path, device)
            if model.config.num_classes != entry["num_classes"]:
                raise ValueError(f"Model class count differs from manifest: {entry['id']}")
            if model.config.embedding_dim != entry["embedding_dim"]:
                raise ValueError(f"Model embedding width differs from manifest: {entry['id']}")
        representation = entry["representation"]
        transform = BenchmarkRepresentation(
            representation=representation,
            num_bins=entry["num_bins"],
            height=480,
            width=640,
        )
        component_splits = representation_channel_splits(representation, entry["num_bins"])
        eros = (
            ErosSnapshotStore(Path(args["eros_cache_root"])) if "eros" in representation else None
        )
        timestamps = load_image_timestamps(timestamps_path)
        handle, x, y, polarity, event_t, ms_to_idx, offset, _ = load_event_file(events_path)
        rows, embeddings, offsets = [], [], [0]
        embedding_state = temporal_state = None
        started = time.monotonic()
        try:
            for frame, timestamp in enumerate(timestamps):
                events = read_events(
                    x=x,
                    y=y,
                    p=polarity,
                    t=event_t,
                    ms_to_idx=ms_to_idx,
                    t_offset=offset,
                    timestamp_us=timestamp,
                    window_us=entry["time_window_us"],
                )
                dense = transform(
                    events,
                    eros=eros.get(dataset_split, sequence, frame, timestamp) if eros else None,
                )
                tensor = torch.from_numpy(dense).float().unsqueeze(0).to(device)
                tensor = (
                    normalise_representation_tensor(tensor, component_splits)
                    if entry["normalisation"] == "component"
                    else normalise_event_tensor(tensor)
                )
                with torch.inference_mode():
                    if model.config.temporal_recurrence_locations:
                        outputs = model(
                            tensor, embedding_state=embedding_state, temporal_state=temporal_state
                        )
                    else:
                        outputs = model(tensor, embedding_state)
                    embedding_state = outputs.get("embedding_state")
                    temporal_state = outputs.get("temporal_state")
                    is_dense = model.config.embedding_head_type == "dense"
                    # No total cap: later score filtering must not lose other classes.
                    detections = decode_dense_detections(
                        outputs,
                        frame,
                        int(timestamp),
                        score_threshold=SCORE_FLOOR,
                        nms_iou_threshold=0.5,
                        max_detections=2**31 - 1,
                        stable_nms=True,
                        feature_stride=model.config.feature_stride,
                        embeddings=outputs.get("embeddings") if is_dense else None,
                    )
                    counts: dict[int, int] = {}
                    survivors = []
                    for detection in detections:
                        count = counts.get(detection.class_id, 0)
                        if count < MAX_DETECTIONS:
                            survivors.append(detection)
                            counts[detection.class_id] = count + 1
                    if entry["embedding_dim"] and not is_dense:
                        feature_map = outputs["embedding_feature_map"]
                        boxes = detection_boxes_xyxy(survivors, feature_map)
                        vectors = model.extract_roi_embeddings(feature_map, [boxes])
                        survivors = attach_detection_embeddings(survivors, vectors)
                    frame_rows, frame_embeddings = pack_frame(survivors, entry["embedding_dim"])
                rows.append(frame_rows)
                embeddings.append(frame_embeddings)
                offsets.append(offsets[-1] + len(frame_rows))
                if frame % 100 == 0 or frame == len(timestamps) - 1:
                    print(
                        f"export {entry['id']} {sequence}: {frame + 1}/{len(timestamps)}",
                        flush=True,
                    )
                    check_disk(cache_root, args["min_free_gb"])
        finally:
            handle.close()
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                rows=np.concatenate(rows),
                embeddings=np.concatenate(embeddings),
                offsets=np.asarray(offsets, dtype=np.int64),
                timestamps=np.asarray(timestamps, dtype=np.int64),
            )
        temporary.replace(path)
        metadata = {
            "identity": identity,
            "sha256": file_hash(path),
            "checkpoint_sha256": file_hash(checkpoint_path),
            "frames": len(timestamps),
            "detections": offsets[-1],
            "elapsed_s": time.monotonic() - started,
            "bytes": path.stat().st_size,
        }
        atomic_json(metadata_path, metadata)
        result[sequence] = metadata
    return result


@lru_cache(maxsize=2)
def load_cache(path: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_tracks(cache: dict, output_path: Path, params: dict, entry: dict, mode: str) -> dict:
    from src.evaluation.mot_trackers import BoxMotTracker, DetectionObservation, TrackingConfig

    reid = mode == "reid"
    if reid and entry["embedding_dim"] <= 0:
        raise ValueError("ReID requested for a checkpoint without embeddings")
    config = TrackingConfig(
        backend="boxmot_botsort",
        iou_threshold=params["match"],
        max_missed_frames=params["buffer"],
        high_threshold=params["high"],
        low_threshold=params["low"],
        with_reid=reid,
        appearance_thresh=params["appearance"],
        proximity_thresh=params["proximity"],
        per_class=True,
        num_classes=entry["num_classes"],
        cmc_method=None,
    )
    tracker = BoxMotTracker("boxmot_botsort", config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for frame, timestamp in enumerate(cache["timestamps"]):
            start, stop = cache["offsets"][frame : frame + 2]
            rows = cache["rows"][start:stop]
            indices = select_indices(rows, params["score"])
            detections = [
                DetectionObservation(
                    frame,
                    int(timestamp),
                    int(row[5]),
                    float(row[4]),
                    float(row[0]),
                    float(row[1]),
                    float(row[2]) - float(row[0]),
                    float(row[3]) - float(row[1]),
                )
                for row in rows[indices]
            ]
            vectors = cache["embeddings"][start:stop][indices] if reid else None
            for record in tracker.update(detections, frame, int(timestamp), vectors):
                stream.write(record.to_trackeval_line() + "\n")
                count += 1
    return {
        "frames": len(cache["timestamps"]),
        "tracks_written": count,
        "requested_track_buffer": params["buffer"],
        "effective_max_time_lost": tracker._tracker.max_time_lost,
        "frame_rate": 20,
        "per_class": True,
        "cmc_method": None,
    }


def evaluate(
    entry: dict, mode: str, params: dict, split: str, args: dict, destination: Path | None = None
) -> dict:
    """Test needs frozen selection unless explicit diagnostic test tuning is enabled."""
    from src.evaluation.trackeval_adapter import (
        export_trackeval_bundle,
        run_trackeval,
        summarise_trackeval_results,
    )

    sequences = split_sequences(split, args)
    if (
        split == "test"
        and args.get("tuning_split", "val") != "test"
        and not (Path(args["output_root"]) / "selection.json").is_file()
    ):
        raise RuntimeError("Test evaluation requires a frozen selection.json")
    work = Path(args["output_root"]) / "work"
    check_disk(work, args["min_free_gb"])
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=work, prefix="trial-") as temporary:
        output = destination or Path(temporary)
        output.mkdir(parents=True, exist_ok=True)
        tracker_runs = {}
        with (output / "evaluation.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                for sequence in sequences:
                    cache_path = (
                        Path(args["output_root"])
                        / "cache"
                        / entry["id"]
                        / split
                        / f"{sequence}.npz"
                    )
                    tracker_runs[sequence] = write_tracks(
                        load_cache(str(cache_path)),
                        output / "tracks" / f"{sequence}.txt",
                        params,
                        entry,
                        mode,
                    )
                bundle = export_trackeval_bundle(
                    Path(args["root"]),
                    "test" if split == "test" else "train",
                    sequences,
                    "tuned",
                    output / "tracks",
                    output / "trackeval",
                )
                raw, messages = run_trackeval(
                    bundle,
                    output / "trackeval/reports",
                    Path(args["trackeval_root"]),
                    classes_to_eval=["car"] if entry["scope"] == "car_only" else None,
                )
                summary = summarise_trackeval_results(
                    raw,
                    "tuned",
                    trackeval_root=Path(args["trackeval_root"]),
                    classes_to_eval=["car"] if entry["scope"] == "car_only" else None,
                )
                if not all(np.isfinite(summary["aggregate"][key]) for key in METRICS):
                    raise RuntimeError("TrackEval returned nonfinite metrics")
        if destination is not None:
            atomic_json(output / "metrics_summary.json", summary)
            atomic_json(output / "tracker_runs.json", tracker_runs)
            atomic_json(output / "trackeval_messages.json", messages)
        summary.pop("raw_trackeval", None)
        return {
            "params": params,
            "mode": mode,
            "split": split,
            "metrics": summary["aggregate"],
            "per_sequence": summary["per_sequence"],
            "elapsed_s": time.monotonic() - started,
            "tracker_runs": tracker_runs,
        }


def initialize_worker(threads: int) -> None:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    import torch

    torch.set_num_threads(threads)
