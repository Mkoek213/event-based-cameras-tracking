"""Convert DSEC-Detection and DSEC-MOT annotations into the large event MOT manifest.

Both datasets label DSEC recordings and share one canonical frame: the rectified left
event frame (640x480), the frame of the DSEC-MOT ground truth. Events stay in their
raw files and are rectified through ``events_left/rectify_map.h5`` by rounding;
events mapping outside the frame are dropped.

- DSEC-MOT boxes are already rectified; they are cropped to the frame. Steps are the
  sequence's image timestamps, so record-less frames are explicit unannotated steps.
  Its track IDs are the only identity supervision in the pipeline.
- DSEC-Detection boxes are in the raw (distorted) event frame; each box is mapped to the
  bounding box of its densely sampled boundary through the rectification map. Steps
  follow the time-based 20 Hz label grid. Boxes below the rectified area floor, boxes
  without events in their window, and classes without a target row become ignore
  regions. Its track IDs are stored but never identity supervision.

Recording exclusions, low-density exclusions, the held-out ``val_monitor`` split, and
label precedence (sequences labelled by DSEC-MOT use only DSEC-MOT labels) come from
the data configuration; every DSEC-Detection label sequence must be accounted for.

Example::

    python -m src.data.converters.dsec_detection \\
        --dsec-root <dsec_root> --detection-labels-root <labels_root> \\
        --mot-annotations-root <dsec_mot>/annotations --output-root <manifest_root> \\
        --config configs/large_event_mot/data.json --limit 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from src.data.converters.large_event_pretraining import map_polarity
from src.data.large_event_schema import (
    BOX_BELOW_AREA_FLOOR,
    BOX_POSITIVE,
    BOX_UNMAPPED_CLASS,
    BOX_ZERO_EVENTS,
    DSEC_DETECTION,
    DSEC_MOT,
    FRAMES,
    SCHEMA_VERSION,
    STEP_ANNOTATED,
    STEP_LABELS_FILTERED,
    STEP_NO_LABEL,
    SchemaError,
    alignment_sample,
    box_density_ratio,
    box_event_counts,
    build_label_grid,
    concatenate_boxes,
    dsec_recording,
    event_count_image,
    iter_window_events,
    ontology_record,
    read_json,
    reusable_record,
    settings_hash,
    sha256_file,
    shifted_control_ratio,
    store_sequence,
    summarize_alignment,
    validate_output_root,
    write_dataset_manifest,
)

CONVERTER_VERSION = "dsec/1"
WIDTH, HEIGHT = FRAMES["dsec_rectified"]
DETECTION_FIELDS = ("t", "x", "y", "w", "h", "class_id", "class_confidence", "track_id")
EVENT_BLOCK = 8_000_000
DATASET_ROOTS = {
    DSEC_MOT: "dsec_mot_annotations_root",
    DSEC_DETECTION: "dsec_detection_labels_root",
}
ROOT_FLAGS = {
    DSEC_MOT: "--mot-annotations-root",
    DSEC_DETECTION: "--detection-labels-root",
}


def dataset_settings(config: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    """Conversion settings of one DSEC dataset; their hash keys resumed records."""

    settings = {
        "schema_version": SCHEMA_VERSION,
        "converter_version": CONVERTER_VERSION,
        "window_us": int(config["window_us"]),
        "grid": dict(config["grid"]),
        **config[dataset],
    }
    if dataset == DSEC_DETECTION:
        settings["precedence_sequences"] = sorted(
            sequence for names in config[DSEC_MOT]["splits"].values() for sequence in names
        )
    return settings


# --------------------------------------------------------------------------- rectification


def load_rectify_map(path: str | Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        rectify = handle["rectify_map"][:]
    if rectify.shape != (HEIGHT, WIDTH, 2) or not np.all(np.isfinite(rectify)):
        raise SchemaError(f"{path}: rectify_map must be finite with shape {(HEIGHT, WIDTH, 2)}")
    return rectify.astype(np.float64)


def rectify_event_pixels(
    x: np.ndarray, y: np.ndarray, rectify: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map raw event pixels to rounded rectified pixels; return ``(x, y, inside)``."""

    mapped = rectify[np.asarray(y, np.int64), np.asarray(x, np.int64)]
    xr = np.rint(mapped[:, 0]).astype(np.int64)
    yr = np.rint(mapped[:, 1]).astype(np.int64)
    inside = (xr >= 0) & (xr < WIDTH) & (yr >= 0) & (yr < HEIGHT)
    return xr, yr, inside


def interpolate_rectify(rectify: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear lookup of rectified coordinates at raw positions (clamped to the map)."""

    height, width = rectify.shape[:2]
    u = np.clip(np.asarray(u, np.float64), 0, width - 1)
    v = np.clip(np.asarray(v, np.float64), 0, height - 1)
    u0 = np.minimum(np.floor(u).astype(np.int64), width - 2)
    v0 = np.minimum(np.floor(v).astype(np.int64), height - 2)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]
    top = rectify[v0, u0] * (1 - du) + rectify[v0, u0 + 1] * du
    bottom = rectify[v0 + 1, u0] * (1 - du) + rectify[v0 + 1, u0 + 1] * du
    return top * (1 - dv) + bottom * dv


def map_boxes_to_rectified(
    xywh: np.ndarray, rectify: np.ndarray, samples_per_edge: int
) -> np.ndarray:
    """Map raw-frame boxes to rectified boxes (bounding box of mapped boundary samples).

    The result is clipped to the rectified frame and may have zero area.
    """

    boxes = np.asarray(xywh, np.float64).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return np.empty((0, 4), np.float64)
    if samples_per_edge < 2:
        raise SchemaError("boundary sampling needs at least two samples per edge")
    fraction = np.linspace(0.0, 1.0, samples_per_edge)
    x0, y0 = boxes[:, :1], boxes[:, 1:2]
    x1, y1 = x0 + boxes[:, 2:3], y0 + boxes[:, 3:4]
    along_x = x0 + (x1 - x0) * fraction
    along_y = y0 + (y1 - y0) * fraction
    u = np.concatenate(
        [along_x, along_x, np.repeat(x0, samples_per_edge, 1), np.repeat(x1, samples_per_edge, 1)],
        1,
    )
    v = np.concatenate(
        [np.repeat(y0, samples_per_edge, 1), np.repeat(y1, samples_per_edge, 1), along_y, along_y],
        1,
    )
    mapped = interpolate_rectify(rectify, u, v)
    left = np.clip(mapped[..., 0].min(1), 0, WIDTH)
    right = np.clip(mapped[..., 0].max(1), 0, WIDTH)
    top = np.clip(mapped[..., 1].min(1), 0, HEIGHT)
    bottom = np.clip(mapped[..., 1].max(1), 0, HEIGHT)
    return np.stack([left, top, right - left, bottom - top], 1)


def crop_to_frame(xywh: np.ndarray) -> np.ndarray:
    boxes = np.asarray(xywh, np.float64).reshape(-1, 4)
    x0 = np.clip(boxes[:, 0], 0, WIDTH)
    y0 = np.clip(boxes[:, 1], 0, HEIGHT)
    x1 = np.clip(boxes[:, 0] + boxes[:, 2], 0, WIDTH)
    y1 = np.clip(boxes[:, 1] + boxes[:, 3], 0, HEIGHT)
    return np.stack([x0, y0, x1 - x0, y1 - y0], 1)


# --------------------------------------------------------------------------- events


def event_time_span(path: str | Path) -> tuple[int, int, int]:
    """Return ``(t_offset, first_us, last_us)`` in absolute microseconds."""

    import h5py
    import hdf5plugin  # noqa: F401  Registers the compression filters.

    with h5py.File(path, "r") as handle:
        t_offset = int(handle["t_offset"][()])
        times = handle["events/t"]
        if times.shape[0] == 0:
            raise SchemaError(f"{path}: no events")
        return t_offset, t_offset + int(times[0]), t_offset + int(times[-1])


def iter_dsec_events(
    path: str | Path,
    rectify: np.ndarray,
    polarity_map: Mapping[str, int],
    stats: Optional[dict[str, Any]] = None,
    block_events: int = EVENT_BLOCK,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield rectified, time-ordered event blocks with absolute timestamps.

    DSEC event files are time-sorted; any decrease is a schema error.
    """

    import h5py
    import hdf5plugin  # noqa: F401

    stats = {} if stats is None else stats
    stats.update(input_events=0, dropped_outside_rectified=0, events=0)
    with h5py.File(path, "r") as handle:
        events = handle["events"]
        lengths = {events[name].shape[0] for name in ("x", "y", "t", "p")}
        if len(lengths) != 1:
            raise SchemaError(f"{path}: event arrays have different lengths {sorted(lengths)}")
        total = lengths.pop()
        t_offset = int(handle["t_offset"][()])
        previous = None
        for start in range(0, total, block_events):
            stop = min(total, start + block_events)
            t = events["t"][start:stop].astype(np.int64) + t_offset
            x = events["x"][start:stop].astype(np.int64)
            y = events["y"][start:stop].astype(np.int64)
            p = events["p"][start:stop]
            if np.any(np.diff(t) < 0) or (previous is not None and t[0] < previous):
                raise SchemaError(f"{path}: event timestamps decrease near event {start}")
            if np.any((x < 0) | (x >= WIDTH) | (y < 0) | (y >= HEIGHT)):
                raise SchemaError(f"{path}: raw event outside the {WIDTH}x{HEIGHT} sensor")
            previous = int(t[-1])
            xr, yr, inside = rectify_event_pixels(x, y, rectify)
            stats["input_events"] += int(t.size)
            stats["dropped_outside_rectified"] += int(np.count_nonzero(~inside))
            stats["events"] += int(np.count_nonzero(inside))
            yield {
                "x": xr[inside].astype(np.uint16),
                "y": yr[inside].astype(np.uint16),
                "t": t[inside],
                "p": map_polarity(p[inside], polarity_map),
            }


# --------------------------------------------------------------------------- labels


def load_detection_labels(path: str | Path) -> dict[str, np.ndarray]:
    labels = np.load(path, allow_pickle=False)
    if labels.dtype.names is None or tuple(labels.dtype.names) != DETECTION_FIELDS:
        raise SchemaError(f"{path}: fields {labels.dtype.names}, expected {DETECTION_FIELDS}")
    for field in ("x", "y", "w", "h", "class_confidence"):
        if not np.all(np.isfinite(labels[field])):
            raise SchemaError(f"{path}: non-finite {field}")
    if np.any(labels["class_id"] > 7):
        raise SchemaError(f"{path}: class_id outside the DSEC-Detection ontology")
    rows = {
        "t": labels["t"].astype(np.int64),
        "xywh": np.stack([labels[key].astype(np.float64) for key in ("x", "y", "w", "h")], 1),
        "class_id": labels["class_id"].astype(np.int64),
        "track_id": labels["track_id"].astype(np.int64),
        "confidence": labels["class_confidence"].astype(np.float32),
    }
    _check_rows(path, rows)
    return rows


def load_mot_annotations(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``t, track_id, x, y, w, h, class_id`` rows of a DSEC-MOT annotation file."""

    table = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if table.size == 0:
        table = np.empty((0, 7), np.float64)
    if table.shape[1] != 7 or not np.all(np.isfinite(table)):
        raise SchemaError(f"{path}: expected seven finite columns per row")
    integers = table[:, [0, 1, 6]]
    if not np.array_equal(integers, np.round(integers)):
        raise SchemaError(f"{path}: timestamp, track ID, and class must be integers")
    rows = {
        "t": table[:, 0].astype(np.int64),
        "xywh": table[:, 2:6].copy(),
        "class_id": table[:, 6].astype(np.int64),
        "track_id": table[:, 1].astype(np.int64),
        "confidence": np.full(table.shape[0], np.nan, np.float32),
    }
    if np.any((rows["class_id"] < 0) | (rows["class_id"] > 6)):
        raise SchemaError(f"{path}: class_id outside the DSEC-MOT ontology")
    _check_rows(path, rows)
    return rows


def _check_rows(path: str | Path, rows: Mapping[str, np.ndarray]) -> None:
    if np.any(rows["t"] < 0) or np.any(rows["track_id"] < 0):
        raise SchemaError(f"{path}: negative timestamp or track ID")
    if np.any(rows["xywh"][:, 2:] <= 0):
        raise SchemaError(f"{path}: box with non-positive size")
    keys = np.stack([rows["t"], rows["track_id"]], 1)
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise SchemaError(f"{path}: duplicate (t, track_id) rows")


def _select(rows: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in rows.items()}


# --------------------------------------------------------------------------- conversion


def convert_sequence(
    dataset: str,
    sequence: str,
    split: str,
    roots: Mapping[str, str | Path],
    files: Mapping[str, str],
    settings: Mapping[str, Any],
    dataset_dir: str | Path,
    block_events: int = EVENT_BLOCK,
) -> dict[str, Any]:
    """Convert one DSEC sequence (or reuse an identical earlier conversion).

    ``files`` holds source-relative paths: ``events`` and ``rectify_map`` under
    ``roots['dsec_root']``, ``labels`` under the dataset's label root, and for
    DSEC-MOT ``image_timestamps`` under ``roots['dsec_root']``.
    """

    dsec_root = Path(roots["dsec_root"])
    label_root_key = DATASET_ROOTS[dataset]
    events_path = dsec_root / files["events"]
    rectify_path = dsec_root / files["rectify_map"]
    labels_path = Path(roots[label_root_key]) / files["labels"]
    digest = settings_hash(settings)
    sources = {
        "labels_sha256": sha256_file(labels_path),
        "events_bytes": events_path.stat().st_size,
        "rectify_map_sha256": sha256_file(rectify_path),
    }
    if dataset == DSEC_MOT:
        sources["image_timestamps_sha256"] = sha256_file(dsec_root / files["image_timestamps"])
    previous = reusable_record(dataset_dir, sequence, digest, sources)
    if previous is not None:
        return previous

    rectify = load_rectify_map(rectify_path)
    if dataset == DSEC_DETECTION:
        mapping = {int(key): value for key, value in settings["source_to_target"].items()}
    else:
        mapping = {index: index for index in range(7)}
    dropped = {"zero_area_in_frame": 0, "outside_event_span": 0}

    if dataset == DSEC_DETECTION:
        rows = load_detection_labels(labels_path)
        rows["xywh"] = map_boxes_to_rectified(
            rows["xywh"], rectify, int(settings["boundary_samples_per_edge"])
        )
    else:
        rows = load_mot_annotations(labels_path)
        rows["xywh"] = crop_to_frame(rows["xywh"])
    raw_times = np.unique(rows["t"])
    keep = (rows["xywh"][:, 2] > 0) & (rows["xywh"][:, 3] > 0)
    dropped["zero_area_in_frame"] = int(np.count_nonzero(~keep))
    rows = _select(rows, keep)

    t_offset, event_first, event_last = event_time_span(events_path)
    window_us = int(settings["window_us"])
    if settings["drop_boxes_outside_event_span"]:
        # A label is outside the span when no causal window ending at it can hold events.
        inside = (rows["t"] >= event_first) & (rows["t"] <= event_last + window_us)
        dropped["outside_event_span"] = int(np.count_nonzero(~inside))
        rows = _select(rows, inside)

    if dataset == DSEC_MOT:
        image_times = np.loadtxt(dsec_root / files["image_timestamps"], dtype=np.int64, ndmin=1)
        if image_times.size == 0 or np.any(np.diff(image_times) <= 0):
            raise SchemaError(f"{sequence}: image timestamps must be strictly increasing")
        if not np.all(np.isin(raw_times, image_times)):
            raise SchemaError(f"{sequence}: annotation timestamp is not an image timestamp")
        step_t = image_times
        snapped = np.isin(step_t, raw_times)
        grid = {"grid": "image_timestamps"}
    else:
        step_t, snapped = build_label_grid(
            rows["t"], settings["grid"]["step_us"], settings["grid"]["tolerance_us"]
        )
        grid = {"grid": "label_snap", **settings["grid"]}
    step_of_row = np.searchsorted(step_t, rows["t"])
    on_step = step_of_row < step_t.size
    on_step[on_step] &= step_t[step_of_row[on_step]] == rows["t"][on_step]
    on_step[on_step] &= snapped[step_of_row[on_step]]
    dropped["off_grid"] = int(np.count_nonzero(~on_step))
    rows = _select(rows, on_step)
    step_of_row = step_of_row[on_step]

    has_rows = np.bincount(step_of_row, minlength=step_t.size) > 0
    status = np.full(step_t.size, STEP_NO_LABEL, np.uint8)
    status[snapped] = STEP_LABELS_FILTERED
    status[has_rows] = STEP_ANNOTATED
    annotated = np.flatnonzero(status == STEP_ANNOTATED)
    sampled = alignment_sample(annotated, int(settings["alignment_windows"]))

    floor = settings.get("min_rectified_area_px2")
    cleaning: dict[str, Any] = {}
    event_counts = np.zeros(step_t.size, np.int64)
    parts, ratios, controls = [], [], []
    windows = iter_window_events(
        iter_dsec_events(events_path, rectify, settings["polarity_map"], cleaning, block_events),
        step_t - window_us,
        step_t,
    )
    for step, window in enumerate(windows):
        if not window:
            continue
        event_counts[step] = int(window["t"].size)
        if status[step] != STEP_ANNOTATED:
            continue
        index = np.flatnonzero(step_of_row == step)
        boxes = rows["xywh"][index]
        image = event_count_image(window["x"], window["y"], WIDTH, HEIGHT)
        inside = box_event_counts(image, boxes)
        source = rows["class_id"][index]
        target = np.array(
            [-1 if mapping[int(value)] is None else mapping[int(value)] for value in source],
            np.int16,
        )
        reason = np.full(index.size, BOX_POSITIVE, np.uint8)
        if settings["zero_event_boxes_ignored"]:
            reason[inside == 0] = BOX_ZERO_EVENTS
        if floor is not None:
            reason[boxes[:, 2] * boxes[:, 3] < float(floor)] = BOX_BELOW_AREA_FLOOR
        reason[target < 0] = BOX_UNMAPPED_CLASS
        positive = reason == BOX_POSITIVE
        if step in sampled and np.any(positive):
            ratio = box_density_ratio(image, boxes[positive])
            control = shifted_control_ratio(image, boxes[positive])
            if ratio is not None and control is not None:
                ratios.append(ratio)
                controls.append(control)
        parts.append(
            {
                "box_step": np.full(index.size, step, np.int32),
                "box_xywh": boxes.astype(np.float32),
                "box_source_class": source.astype(np.int16),
                "box_target_class": target,
                "box_track_id": rows["track_id"][index],
                "box_has_track_id": np.ones(index.size, np.bool_),
                "box_ignore": ~positive,
                "box_reason": reason,
                "box_event_count": inside.astype(np.int64),
                "box_confidence": rows["confidence"][index],
            }
        )

    arrays = {
        "step_t_us": step_t.astype(np.int64),
        "step_status": status,
        "step_snapped": snapped.astype(np.bool_),
        "step_event_count": event_counts,
        **concatenate_boxes(parts),
    }
    ontology = ontology_record(dataset, mapping)
    ontology["masked_target_classes"] = list(settings.get("masked_target_classes", []))
    supervision = bool(settings["identity_supervision"])
    record = {
        "schema_version": SCHEMA_VERSION,
        "converter_version": CONVERTER_VERSION,
        "settings_sha256": digest,
        "dataset": dataset,
        "family": "dsec",
        "sequence": sequence,
        "recording": dsec_recording(sequence),
        "split": split,
        "sources": sources,
        "events": {
            "root": "dsec_root",
            "path": files["events"],
            "format": "dsec_h5",
            "bytes": sources["events_bytes"],
            "t_offset_us": t_offset,
            "t_first_us": event_first,
            "t_last_us": event_last,
            "sha256": sha256_file(events_path),
            "polarity_map": dict(settings["polarity_map"]),
            "cleaning": cleaning,
        },
        "rectify_map": {
            "root": "dsec_root",
            "path": files["rectify_map"],
            "sha256": sources["rectify_map_sha256"],
            "pixel_assignment": "round",
        },
        "labels": {
            "root": label_root_key,
            "path": files["labels"],
            "sha256": sources["labels_sha256"],
            "frame": ("dsec_raw" if dataset == DSEC_DETECTION else "dsec_rectified"),
        },
        "frames": {
            "event_frame": "dsec_raw",
            "box_frame": "dsec_rectified",
            "width": WIDTH,
            "height": HEIGHT,
        },
        "time": {"unit": "us", "base": "absolute", "window_us": window_us, **grid},
        "ontology": ontology,
        "identity": {"supervision": supervision, "namespace": f"{dataset}/{sequence}"},
        "dropped_boxes": dropped,
        "alignment": summarize_alignment(ratios, controls),
    }
    if dataset == DSEC_MOT:
        record["image_timestamps"] = {
            "root": "dsec_root",
            "path": files["image_timestamps"],
            "sha256": sources["image_timestamps_sha256"],
        }
    else:
        record["identity"]["reason"] = settings["identity_reason"]
    return store_sequence(dataset_dir, record, arrays)


# --------------------------------------------------------------------------- planning


def find_event_sequences(dsec_root: str | Path) -> dict[str, str]:
    """Map sequence name to its directory relative to ``dsec_root`` (``<split>/<seq>``)."""

    root = Path(dsec_root)
    found: dict[str, str] = {}
    for events in sorted(root.glob("*/*/events_left/events.h5")):
        sequence_dir = events.parent.parent
        name = sequence_dir.name
        if name in found:
            raise SchemaError(f"sequence {name} found twice under {root}")
        dsec_recording(name)
        found[name] = sequence_dir.relative_to(root).as_posix()
    return found


def find_detection_labels(labels_root: str | Path) -> dict[str, str]:
    """Map sequence name to its ``tracks.npy`` relative to the label root.

    Accepts the official layout ``<split>/<seq>/object_detections/left/tracks.npy`` and
    the flattened ``<split>/<seq>/tracks.npy``.
    """

    root = Path(labels_root)
    found: dict[str, str] = {}
    for path in sorted(root.rglob("tracks.npy")):
        relative = path.relative_to(root)
        names = [part for part in relative.parts[:-1] if _is_sequence(part)]
        if len(names) != 1:
            raise SchemaError(f"cannot identify the sequence of {relative}")
        if "right" in relative.parts:
            continue
        if names[0] in found:
            raise SchemaError(f"sequence {names[0]} has two label files under {root}")
        found[names[0]] = relative.as_posix()
    return found


def find_mot_annotations(annotations_root: str | Path) -> dict[str, str]:
    root = Path(annotations_root)
    found: dict[str, str] = {}
    for path in sorted(root.glob("*/*.txt")):
        if path.stem in found:
            raise SchemaError(f"sequence {path.stem} has two annotation files under {root}")
        dsec_recording(path.stem)
        found[path.stem] = path.relative_to(root).as_posix()
    return found


def _is_sequence(name: str) -> bool:
    try:
        dsec_recording(name)
    except SchemaError:
        return False
    return True


def account_detection_sequences(
    settings: Mapping[str, Any], label_sequences: Sequence[str]
) -> dict[str, Any]:
    """Assign every DSEC-Detection label sequence to a split or an exclusion reason."""

    splits = settings["splits"]
    listed: dict[str, str] = {}
    for split, names in splits.items():
        for name in names:
            if name in listed:
                raise SchemaError(f"{name} is listed in {listed[name]} and {split}")
            listed[name] = split
    precedence = set(settings["precedence_sequences"])
    excluded_recordings = settings["excluded_recordings"]
    excluded_sequences = settings["excluded_sequences"]
    for name, split in listed.items():
        recording = dsec_recording(name)
        if name in precedence or recording in excluded_recordings or name in excluded_sequences:
            raise SchemaError(f"{name} is listed in {split} but excluded by the configuration")

    assignment: dict[str, str] = {}
    excluded: dict[str, str] = {}
    for name in sorted(label_sequences):
        recording = dsec_recording(name)
        if name in precedence:
            excluded[name] = "label_precedence:dsec_mot"
        elif recording in excluded_recordings:
            excluded[name] = f"excluded_recording:{recording}"
        elif name in excluded_sequences:
            excluded[name] = "excluded_sequence"
        elif name in listed:
            assignment[name] = listed[name]
        else:
            raise SchemaError(f"DSEC-Detection sequence {name} is not accounted for")
    missing = sorted(set(listed) - set(assignment))
    if missing:
        raise SchemaError(f"configured DSEC-Detection sequences without labels: {missing}")
    if len(label_sequences) != int(settings["expected_label_sequences"]):
        raise SchemaError(
            f"found {len(label_sequences)} DSEC-Detection label sequences, "
            f"expected {settings['expected_label_sequences']}"
        )
    return {"assignment": assignment, "excluded": excluded}


def plan_dataset(
    dataset: str,
    settings: Mapping[str, Any],
    roots: Mapping[str, str],
    limit: Optional[int],
) -> tuple[list[tuple[str, str, dict[str, str]]], dict[str, Any]]:
    """Return conversion jobs ``(sequence, split, files)`` and the audit plan."""

    events = find_event_sequences(roots["dsec_root"])
    plan: dict[str, Any] = {"limit": limit}
    if dataset == DSEC_DETECTION:
        labels = find_detection_labels(roots[DATASET_ROOTS[dataset]])
        accounting = account_detection_sequences(settings, list(labels))
        assignment = accounting["assignment"]
        plan["excluded"] = accounting["excluded"]
    else:
        labels = find_mot_annotations(roots[DATASET_ROOTS[dataset]])
        assignment = {name: split for split, names in settings["splits"].items() for name in names}
        unknown = sorted(set(labels) - set(assignment))
        missing = sorted(set(assignment) - set(labels))
        if unknown or missing:
            raise SchemaError(f"DSEC-MOT annotations unknown {unknown}, missing {missing}")

    jobs = []
    plan["sequences"] = {}
    for split in settings["splits"]:
        names = sorted(name for name, value in assignment.items() if value == split)
        selected = names if limit is None else names[:limit]
        plan["sequences"][split] = selected
        for name in selected:
            if name not in events:
                raise SchemaError(f"{dataset} {split} sequence {name} has no event file")
            directory = events[name]
            files = {
                "events": f"{directory}/events_left/events.h5",
                "rectify_map": f"{directory}/events_left/rectify_map.h5",
                "labels": labels[name],
            }
            if dataset == DSEC_MOT:
                files["image_timestamps"] = f"{directory}/{name}_image_timestamps.txt"
            jobs.append((name, split, files))
    return jobs, plan


def _convert_job(arguments: tuple) -> dict[str, Any]:
    return convert_sequence(*arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert DSEC-MOT and DSEC-Detection labels.")
    parser.add_argument("--dsec-root", required=True, help="directory with <split>/<seq>/")
    parser.add_argument("--detection-labels-root", help="extracted DSEC-Detection labels")
    parser.add_argument("--mot-annotations-root", help="DSEC-MOT annotations/ directory")
    parser.add_argument("--output-root", required=True, help="manifest root directory")
    parser.add_argument("--config", required=True, help="configs/large_event_mot/data.json")
    parser.add_argument("--datasets", default=f"{DSEC_MOT},{DSEC_DETECTION}")
    parser.add_argument("--limit", type=int, default=None, help="first N sequences per split")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--block-events", type=int, default=EVENT_BLOCK)
    parser.add_argument("--dry-run", action="store_true", help="plan and check only; no writes")
    args = parser.parse_args(argv)

    config = read_json(args.config)
    roots = {
        "dsec_root": args.dsec_root,
        DATASET_ROOTS[DSEC_MOT]: args.mot_annotations_root,
        DATASET_ROOTS[DSEC_DETECTION]: args.detection_labels_root,
    }
    datasets = args.datasets.split(",")
    planned = []
    for dataset in datasets:
        if dataset not in DATASET_ROOTS:
            raise SchemaError(f"unknown DSEC dataset {dataset}")
        if not roots[DATASET_ROOTS[dataset]]:
            raise SchemaError(f"{dataset} needs {ROOT_FLAGS[dataset]}")
        settings = dataset_settings(config, dataset)
        jobs, plan = plan_dataset(dataset, settings, roots, args.limit)
        print(json.dumps({"dataset": dataset, "plan": plan}, sort_keys=True), flush=True)
        planned.append((dataset, settings, jobs, plan))
    if args.dry_run:
        return 0

    os.umask(0o077)
    for dataset, settings, jobs, plan in planned:
        dataset_dir = Path(args.output_root) / dataset
        arguments = [
            (dataset, name, split, roots, files, settings, dataset_dir, args.block_events)
            for name, split, files in jobs
        ]
        records = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for index, record in enumerate(pool.map(_convert_job, arguments), 1):
                records.append(record)
                counts = record["counts"]
                print(
                    f"[{dataset} {index}/{len(arguments)}] {record['split']}/"
                    f"{record['sequence']} steps={counts['steps']} "
                    f"annotated={counts['steps_by_status']['annotated']} "
                    f"positives={counts['positive_boxes']} "
                    f"aligned={record['alignment']['aligned']}",
                    flush=True,
                )
        write_dataset_manifest(
            args.output_root, dataset, records, settings, complete=args.limit is None, plan=plan
        )
    summary = validate_output_root(args.output_root, config["recording_overlap_allowed"])
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
