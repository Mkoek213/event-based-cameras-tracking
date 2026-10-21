"""Versioned sequence-manifest schema for the large event-based MOT pipeline.

One converted sequence (a 1Mpx chunk, a DSEC-MOT sequence, or a DSEC-Detection
sequence) is stored as two files:

- ``<sequence>.json``: identity, split, source-relative file references with hashes,
  coordinate frames, time base, ontology, identity policy, and audit counts;
- ``<sequence>.npz``: step and box arrays (see ``STEP_ARRAYS`` and ``BOX_ARRAYS``).

A step is one causal event window ``[t - window_us, t)`` ending at ``step_t_us``.
Only steps with ``step_status == STEP_ANNOTATED`` carry boxes and detection loss;
every other step still advances recurrent state but must never be used as background.
Boxes are either positives or ignore regions (``box_ignore``). Track IDs are stored
with an explicit ``box_has_track_id`` flag, so an unknown ID is never confused with a
valid ID zero, and identity loss is allowed only when the sequence's identity policy
says so.

Each dataset directory holds a ``manifest.json`` index listing its sequence records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

GEN4_1MPX = "gen4_1mpx"
DSEC_MOT = "dsec_mot"
DSEC_DETECTION = "dsec_detection"
DATASETS = (GEN4_1MPX, DSEC_MOT, DSEC_DETECTION)

# Datasets recorded with the same physical sensors share one recording namespace.
DATASET_FAMILY = {GEN4_1MPX: "gen4", DSEC_MOT: "dsec", DSEC_DETECTION: "dsec"}
DATASET_SPLITS = {
    GEN4_1MPX: ("train", "val"),
    DSEC_MOT: ("train", "val", "test"),
    DSEC_DETECTION: ("train", "val_monitor"),
}
TRAINING_SPLITS = frozenset({"train"})

# Frame name -> (width, height). Event and box frames of one record must be compatible.
FRAMES = {
    "gen4_native": (1280, 720),
    "dsec_raw": (640, 480),
    "dsec_rectified": (640, 480),
}
DATASET_FRAMES = {
    GEN4_1MPX: ("gen4_native", "gen4_native"),
    DSEC_MOT: ("dsec_raw", "dsec_rectified"),
    DSEC_DETECTION: ("dsec_raw", "dsec_rectified"),
}

TARGET_CLASSES = ("car", "pedestrian", "bicycle", "motorcycle", "bus", "truck", "train")
GEN4_CLASSES = (
    "pedestrian",
    "two_wheeler",
    "car",
    "truck",
    "bus",
    "traffic_sign",
    "traffic_light",
)
DSEC_DETECTION_CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "train",
)
SOURCE_CLASSES = {
    GEN4_1MPX: GEN4_CLASSES,
    DSEC_MOT: TARGET_CLASSES,
    DSEC_DETECTION: DSEC_DETECTION_CLASSES,
}
# The detection head a dataset trains: 1Mpx pretraining keeps its native rows.
HEAD_ONTOLOGY = {GEN4_1MPX: "source", DSEC_MOT: "target", DSEC_DETECTION: "target"}

STEP_ANNOTATED = 0
STEP_NO_LABEL = 1
STEP_LABELS_FILTERED = 2
STEP_FRAME_SPANNING = 3
STEP_SPARSE_WINDOW = 4
STEP_STATUS_NAMES = {
    STEP_ANNOTATED: "annotated",
    STEP_NO_LABEL: "no_label",
    STEP_LABELS_FILTERED: "labels_filtered",
    STEP_FRAME_SPANNING: "frame_spanning",
    STEP_SPARSE_WINDOW: "sparse_window",
}

BOX_POSITIVE = 0
BOX_SMALL = 1
BOX_ZERO_EVENTS = 2
BOX_BELOW_AREA_FLOOR = 3
BOX_UNMAPPED_CLASS = 4
BOX_REASON_NAMES = {
    BOX_POSITIVE: "positive",
    BOX_SMALL: "small_box",
    BOX_ZERO_EVENTS: "zero_events",
    BOX_BELOW_AREA_FLOOR: "below_area_floor",
    BOX_UNMAPPED_CLASS: "unmapped_class",
}

STEP_ARRAYS = {
    "step_t_us": np.dtype(np.int64),
    "step_status": np.dtype(np.uint8),
    "step_snapped": np.dtype(np.bool_),
    "step_event_count": np.dtype(np.int64),
}
BOX_ARRAYS = {
    "box_step": np.dtype(np.int32),
    "box_xywh": np.dtype(np.float32),
    "box_source_class": np.dtype(np.int16),
    "box_target_class": np.dtype(np.int16),
    "box_track_id": np.dtype(np.int64),
    "box_has_track_id": np.dtype(np.bool_),
    "box_ignore": np.dtype(np.bool_),
    "box_reason": np.dtype(np.uint8),
    "box_event_count": np.dtype(np.int64),
    "box_confidence": np.dtype(np.float32),
}

_DSEC_SEQUENCE = re.compile(r"^(?P<recording>[a-z]+(?:_[a-z]+)?_\d{2})_[a-z]$")
_GEN4_CHUNK = re.compile(r"^(?P<session>moorea_.+?)_(?P<start>\d+)_(?P<end>\d+)$")


class SchemaError(ValueError):
    """A record, its arrays, or a set of manifests violates the schema."""


# --------------------------------------------------------------------------- identity


def dsec_recording(sequence: str) -> str:
    """Return the original DSEC recording of a sequence, e.g. ``zurich_city_01``."""

    match = _DSEC_SEQUENCE.match(sequence)
    if match is None:
        raise SchemaError(f"not a DSEC sequence name: {sequence!r}")
    return match.group("recording")


def gen4_chunk_identity(stem: str) -> tuple[str, int, int]:
    """Return ``(session, start_us, end_us)`` of a 1Mpx chunk stem."""

    match = _GEN4_CHUNK.match(stem)
    if match is None:
        raise SchemaError(f"not a 1Mpx chunk name: {stem!r}")
    session = match.group("session")
    if session.endswith("_td"):
        session = session[: -len("_td")]
    start, end = int(match.group("start")), int(match.group("end"))
    if end <= start:
        raise SchemaError(f"chunk {stem!r} has a non-positive time span")
    return session, start, end


def recording_of(dataset: str, sequence: str) -> str:
    if dataset == GEN4_1MPX:
        return gen4_chunk_identity(sequence)[0]
    if dataset in (DSEC_MOT, DSEC_DETECTION):
        return dsec_recording(sequence)
    raise SchemaError(f"unknown dataset {dataset!r}")


# --------------------------------------------------------------------------- hashing and I/O


def sha256_file(path: str | Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def settings_hash(settings: Mapping[str, Any]) -> str:
    """Hash of conversion settings; any change invalidates resumed records."""

    return hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: str | Path, value: Any) -> None:
    text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_annotations_atomic(path: str | Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Write the annotation arrays as a compressed ``.npz``; return its SHA-256."""

    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{name: arrays[name] for name in (*STEP_ARRAYS, *BOX_ARRAYS)})
    payload = buffer.getvalue()
    _atomic_write_bytes(Path(path), payload)
    return hashlib.sha256(payload).hexdigest()


def read_annotations(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def sequence_paths(dataset_dir: str | Path, sequence: str) -> tuple[Path, Path]:
    base = Path(dataset_dir) / "sequences"
    return base / f"{sequence}.json", base / f"{sequence}.npz"


# --------------------------------------------------------------------------- temporal grid


def build_label_grid(
    label_times_us: np.ndarray, step_us: int, tolerance_us: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(step_t_us, step_snapped)`` for the time-based 20 Hz supervised grid.

    The first step is the first label timestamp. Each next step targets the previous
    step time plus ``step_us``; when a label timestamp lies within ``tolerance_us`` of
    the target, the step ends exactly at the closest such label (ties: the earlier one),
    otherwise it ends at the target and is unannotated. The grid stops once no later
    label can be reached. Labels are never interpolated.
    """

    times = np.unique(np.asarray(label_times_us, dtype=np.int64))
    if times.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.bool_)
    if step_us <= 0 or tolerance_us < 0 or 2 * tolerance_us >= step_us:
        raise SchemaError("grid needs step_us > 2 * tolerance_us >= 0")
    steps = [int(times[0])]
    snapped = [True]
    last = int(times[-1])
    current = steps[0]
    while True:
        target = current + step_us
        if target > last + tolerance_us:
            break
        low = int(np.searchsorted(times, target - tolerance_us, side="left"))
        high = int(np.searchsorted(times, target + tolerance_us, side="right"))
        if high > low:
            candidates = times[low:high]
            current = int(candidates[int(np.argmin(np.abs(candidates - target)))])
            snapped.append(True)
        else:
            current = target
            snapped.append(False)
        steps.append(current)
    return np.asarray(steps, np.int64), np.asarray(snapped, np.bool_)


# --------------------------------------------------------------------------- event windows


def iter_window_events(
    blocks: Iterable[Mapping[str, np.ndarray]],
    starts: np.ndarray,
    ends: np.ndarray,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield the events of each half-open window ``[starts[k], ends[k])`` in order.

    ``blocks`` is a time-sorted stream of event blocks with equal-length arrays
    (at least ``t``). Windows must be sorted by start and by end; they may overlap.
    The stream is consumed to its end so that its own integrity checks run.
    """

    starts = np.asarray(starts, np.int64)
    ends = np.asarray(ends, np.int64)
    if starts.shape != ends.shape or np.any(ends < starts):
        raise SchemaError("window starts and ends must align and satisfy start <= end")
    if np.any(np.diff(starts) < 0) or np.any(np.diff(ends) < 0):
        raise SchemaError("windows must be sorted")
    stream = iter(blocks)
    buffer: Optional[dict[str, np.ndarray]] = None
    exhausted = False
    for index in range(starts.size):
        while not exhausted and (
            buffer is None or buffer["t"].size == 0 or int(buffer["t"][-1]) < int(ends[index])
        ):
            try:
                block = next(stream)
            except StopIteration:
                exhausted = True
                break
            if buffer is None or buffer["t"].size == 0:
                buffer = dict(block)
            else:
                buffer = {key: np.concatenate([buffer[key], block[key]]) for key in buffer}
        if buffer is None:
            yield {}
            continue
        low = int(np.searchsorted(buffer["t"], starts[index], side="left"))
        high = int(np.searchsorted(buffer["t"], ends[index], side="left"))
        yield {key: value[low:high] for key, value in buffer.items()}
        next_start = int(starts[index + 1]) if index + 1 < starts.size else int(ends[index])
        keep = int(np.searchsorted(buffer["t"], next_start, side="left"))
        buffer = {key: value[keep:] for key, value in buffer.items()}
    for _ in stream:
        pass


def box_pixel_bounds(boxes_xywh: np.ndarray, width: int, height: int) -> np.ndarray:
    """Integer pixel ranges ``[x0, x1) x [y0, y1)`` whose centres lie inside each box.

    Pixel ``j`` has its centre at ``j``; a box covers ``x <= j < x + w``.
    """

    boxes = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    x0 = np.clip(np.ceil(boxes[:, 0]), 0, width)
    y0 = np.clip(np.ceil(boxes[:, 1]), 0, height)
    x1 = np.clip(np.ceil(boxes[:, 0] + boxes[:, 2]), 0, width)
    y1 = np.clip(np.ceil(boxes[:, 1] + boxes[:, 3]), 0, height)
    return np.stack([x0, y0, np.maximum(x1, x0), np.maximum(y1, y0)], 1).astype(np.int64)


def event_count_image(x: np.ndarray, y: np.ndarray, width: int, height: int) -> np.ndarray:
    index = np.asarray(y, np.int64) * width + np.asarray(x, np.int64)
    return np.bincount(index, minlength=width * height).reshape(height, width)


def box_event_counts(count_image: np.ndarray, boxes_xywh: np.ndarray) -> np.ndarray:
    """Number of events whose pixel lies inside each box, via an integral image."""

    height, width = count_image.shape
    integral = np.zeros((height + 1, width + 1), np.int64)
    integral[1:, 1:] = count_image.cumsum(0).cumsum(1)
    bounds = box_pixel_bounds(boxes_xywh, width, height)
    x0, y0, x1, y1 = bounds.T
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def box_density_ratio(count_image: np.ndarray, boxes_xywh: np.ndarray) -> Optional[float]:
    """Event density inside the union of boxes divided by the density outside it."""

    height, width = count_image.shape
    mask = np.zeros((height, width), np.bool_)
    for x0, y0, x1, y1 in box_pixel_bounds(boxes_xywh, width, height):
        mask[y0:y1, x0:x1] = True
    area_in = int(mask.sum())
    area_out = mask.size - area_in
    hits_in = int(count_image[mask].sum())
    hits_out = int(count_image.sum()) - hits_in
    if area_in == 0 or area_out == 0 or hits_out == 0:
        return None
    return (hits_in / area_in) / (hits_out / area_out)


def shifted_control_ratio(count_image: np.ndarray, boxes_xywh: np.ndarray) -> Optional[float]:
    """Median density ratio of the same boxes shifted by one box size (alignment control)."""

    boxes = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    values = []
    for fx, fy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        moved = boxes.copy()
        moved[:, 0] += fx * boxes[:, 2]
        moved[:, 1] += fy * boxes[:, 3]
        ratio = box_density_ratio(count_image, moved)
        if ratio is not None:
            values.append(ratio)
    return float(np.median(values)) if values else None


def alignment_sample(annotated_steps: np.ndarray, count: int) -> set[int]:
    """Deterministic, evenly spaced subset of annotated steps for alignment statistics."""

    steps = np.asarray(annotated_steps, np.int64)
    if steps.size <= count:
        return set(steps.tolist())
    picks = np.linspace(0, steps.size - 1, count).round().astype(np.int64)
    return set(steps[picks].tolist())


def summarize_alignment(ratios: Sequence[float], controls: Sequence[float]) -> dict[str, Any]:
    ratio = float(np.median(ratios)) if ratios else None
    control = float(np.median(controls)) if controls else None
    return {
        "windows": len(ratios),
        "density_ratio_median": ratio,
        "shifted_control_median": control,
        "aligned": None if ratio is None or control is None else bool(ratio > control),
    }


# --------------------------------------------------------------------------- assembly


def empty_box_arrays() -> dict[str, np.ndarray]:
    arrays = {name: np.empty(0, dtype) for name, dtype in BOX_ARRAYS.items()}
    arrays["box_xywh"] = np.empty((0, 4), np.float32)
    return arrays


def concatenate_boxes(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return empty_box_arrays()
    return {
        name: np.concatenate([np.asarray(part[name], dtype) for part in parts])
        for name, dtype in BOX_ARRAYS.items()
    }


def annotation_counts(arrays: Mapping[str, np.ndarray], dataset: str) -> dict[str, Any]:
    """Audit counts that the sequence record must repeat exactly."""

    status = arrays["step_status"]
    ignore = arrays["box_ignore"]
    reasons = arrays["box_reason"]
    source = arrays["box_source_class"]
    names = SOURCE_CLASSES[dataset]
    positive_by_class = np.bincount(source[~ignore].astype(np.int64), minlength=len(names))
    tracked = arrays["box_has_track_id"]
    return {
        "steps": int(status.size),
        "steps_by_status": {
            name: int(np.count_nonzero(status == code)) for code, name in STEP_STATUS_NAMES.items()
        },
        "boxes": int(ignore.size),
        "positive_boxes": int(np.count_nonzero(~ignore)),
        "ignore_boxes_by_reason": {
            name: int(np.count_nonzero(ignore & (reasons == code)))
            for code, name in BOX_REASON_NAMES.items()
            if code != BOX_POSITIVE
        },
        "positive_boxes_by_source_class": {
            name: int(positive_by_class[index]) for index, name in enumerate(names)
        },
        "tracks": int(np.unique(arrays["box_track_id"][tracked]).size),
    }


def ontology_record(dataset: str, source_to_target: Mapping[int, Optional[int]]) -> dict[str, Any]:
    return {
        "head": HEAD_ONTOLOGY[dataset],
        "source_classes": list(SOURCE_CLASSES[dataset]),
        "target_classes": list(TARGET_CLASSES),
        "source_to_target": {
            str(key): (None if value is None else int(value))
            for key, value in sorted(source_to_target.items())
        },
    }


# --------------------------------------------------------------------------- validation


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def validate_sequence(record: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    """Reject any record or array set that violates the schema contract."""

    name = f"{record.get('dataset')}/{record.get('sequence')}"
    _require(record.get("schema_version") == SCHEMA_VERSION, f"{name}: schema version mismatch")
    dataset = record.get("dataset")
    _require(dataset in DATASETS, f"{name}: unknown dataset")
    _require(record.get("split") in DATASET_SPLITS[dataset], f"{name}: split not allowed")
    _require(
        record.get("recording") == recording_of(dataset, record["sequence"]),
        f"{name}: recording does not match the sequence name",
    )
    _require(record.get("family") == DATASET_FAMILY[dataset], f"{name}: wrong family")

    frames = record.get("frames", {})
    event_frame, box_frame = DATASET_FRAMES[dataset]
    _require(
        frames.get("event_frame") == event_frame and frames.get("box_frame") == box_frame,
        f"{name}: inconsistent coordinate frames {frames}",
    )
    width, height = FRAMES[box_frame]
    _require(
        frames.get("width") == width and frames.get("height") == height,
        f"{name}: frame size does not match {box_frame}",
    )
    needs_map = event_frame != box_frame
    _require(
        (record.get("rectify_map") is not None) == needs_map,
        f"{name}: rectification map presence does not match the frames",
    )
    for key in ("events", "labels"):
        reference = record.get(key) or {}
        _require(
            bool(reference.get("root"))
            and bool(reference.get("path"))
            and not Path(reference["path"]).is_absolute()
            and ".." not in Path(reference["path"]).parts,
            f"{name}: {key} must be a source-relative path",
        )

    ontology = record.get("ontology", {})
    _require(ontology.get("head") == HEAD_ONTOLOGY[dataset], f"{name}: wrong head ontology")
    _require(
        tuple(ontology.get("source_classes", ())) == SOURCE_CLASSES[dataset]
        and tuple(ontology.get("target_classes", ())) == TARGET_CLASSES,
        f"{name}: ontology class names differ from the schema",
    )
    mapping = {int(key): value for key, value in ontology.get("source_to_target", {}).items()}
    _require(
        set(mapping) == set(range(len(SOURCE_CLASSES[dataset]))),
        f"{name}: source_to_target must cover every source class",
    )

    for array_name, dtype in {**STEP_ARRAYS, **BOX_ARRAYS}.items():
        _require(array_name in arrays, f"{name}: missing array {array_name}")
        _require(
            arrays[array_name].dtype == dtype,
            f"{name}: {array_name} has dtype {arrays[array_name].dtype}, expected {dtype}",
        )
    steps = arrays["step_t_us"]
    for array_name in STEP_ARRAYS:
        _require(arrays[array_name].shape == steps.shape, f"{name}: step array length mismatch")
    boxes = arrays["box_step"]
    for array_name in BOX_ARRAYS:
        expected = (boxes.size, 4) if array_name == "box_xywh" else boxes.shape
        _require(arrays[array_name].shape == expected, f"{name}: box array length mismatch")

    _require(bool(np.all(np.diff(steps) > 0)), f"{name}: step timestamps not increasing")
    status = arrays["step_status"]
    _require(bool(np.all(np.isin(status, list(STEP_STATUS_NAMES)))), f"{name}: bad step status")
    annotated = status == STEP_ANNOTATED
    _require(bool(np.all(arrays["step_snapped"][annotated])), f"{name}: annotated unsnapped step")
    _require(bool(np.all(arrays["step_event_count"] >= 0)), f"{name}: negative event count")

    if boxes.size:
        _require(
            bool(np.all((boxes >= 0) & (boxes < steps.size))), f"{name}: box step out of range"
        )
        _require(bool(np.all(np.diff(boxes) >= 0)), f"{name}: boxes not ordered by step")
        _require(bool(np.all(annotated[boxes])), f"{name}: box on a non-annotated step")
    per_step = np.bincount(boxes.astype(np.int64), minlength=steps.size)
    _require(bool(np.all(per_step[annotated] > 0)), f"{name}: annotated step without records")

    xywh = arrays["box_xywh"].astype(np.float64)
    _require(bool(np.all(np.isfinite(xywh))), f"{name}: non-finite box coordinates")
    _require(bool(np.all(xywh[:, 2:] > 0)), f"{name}: box with non-positive size")
    tolerance = 1e-3
    _require(
        bool(
            np.all(xywh[:, 0] >= -tolerance)
            and np.all(xywh[:, 1] >= -tolerance)
            and np.all(xywh[:, 0] + xywh[:, 2] <= width + tolerance)
            and np.all(xywh[:, 1] + xywh[:, 3] <= height + tolerance)
        ),
        f"{name}: box outside the {box_frame} frame",
    )

    source = arrays["box_source_class"].astype(np.int64)
    target = arrays["box_target_class"].astype(np.int64)
    _require(
        bool(np.all((source >= 0) & (source < len(SOURCE_CLASSES[dataset])))),
        f"{name}: source class out of range",
    )
    expected_target = np.array(
        [-1 if mapping[int(value)] is None else mapping[int(value)] for value in source],
        np.int64,
    )
    _require(bool(np.array_equal(target, expected_target)), f"{name}: target class mismatch")
    reasons = arrays["box_reason"]
    ignore = arrays["box_ignore"]
    _require(bool(np.all(np.isin(reasons, list(BOX_REASON_NAMES)))), f"{name}: bad box reason")
    _require(
        bool(np.array_equal(ignore, reasons != BOX_POSITIVE)),
        f"{name}: ignore flags disagree with reasons",
    )
    if HEAD_ONTOLOGY[dataset] == "target":
        _require(bool(np.all(target[~ignore] >= 0)), f"{name}: positive box without a target class")

    track = arrays["box_track_id"]
    has_track = arrays["box_has_track_id"]
    _require(bool(np.all(track[~has_track] == -1)), f"{name}: unknown ID must be stored as -1")
    _require(bool(np.all(track[has_track] >= 0)), f"{name}: negative track ID")
    keys = np.stack([boxes[has_track].astype(np.int64), track[has_track]], 1)
    _require(
        np.unique(keys, axis=0).shape[0] == keys.shape[0],
        f"{name}: duplicate track ID within one step",
    )
    identity = record.get("identity", {})
    _require(
        identity.get("namespace") == f"{dataset}/{record['sequence']}",
        f"{name}: identity namespace must be dataset/sequence",
    )
    if identity.get("supervision"):
        _require(bool(np.all(has_track[~ignore])), f"{name}: identity positive without an ID")
        pairs = np.unique(np.stack([track[has_track], target[has_track]], 1), axis=0)
        _require(
            np.unique(pairs[:, 0]).size == pairs.shape[0],
            f"{name}: a supervised track changes class",
        )

    _require(
        record.get("counts") == annotation_counts(arrays, dataset),
        f"{name}: recorded counts disagree with the arrays",
    )


def load_sequence(dataset_dir: str | Path, entry: Mapping[str, Any]) -> tuple[dict, dict]:
    """Load and fully validate one manifest entry, including the annotation hash."""

    json_path, npz_path = sequence_paths(dataset_dir, entry["sequence"])
    record = read_json(json_path)
    _require(
        sha256_file(npz_path) == record["annotations"]["sha256"],
        f"{entry['sequence']}: annotation file hash mismatch",
    )
    arrays = read_annotations(npz_path)
    validate_sequence(record, arrays)
    for key in ("dataset", "split", "recording"):
        _require(record[key] == entry[key], f"{entry['sequence']}: manifest {key} mismatch")
    return record, arrays


# --------------------------------------------------------------------------- conversion runs


def reusable_record(
    dataset_dir: str | Path, sequence: str, settings_digest: str, source: Mapping[str, Any]
) -> Optional[dict]:
    """Return a previously written record when it matches settings and sources exactly.

    Stale, partial, or invalid outputs return ``None`` so the sequence is reconverted.
    """

    json_path, npz_path = sequence_paths(dataset_dir, sequence)
    if not json_path.is_file() or not npz_path.is_file():
        return None
    try:
        record = read_json(json_path)
        if record.get("schema_version") != SCHEMA_VERSION:
            return None
        if record.get("settings_sha256") != settings_digest:
            return None
        for key, expected in source.items():
            if record.get("sources", {}).get(key) != expected:
                return None
        if sha256_file(npz_path) != record["annotations"]["sha256"]:
            return None
        validate_sequence(record, read_annotations(npz_path))
    except (OSError, KeyError, ValueError):
        return None
    return record


def store_sequence(dataset_dir: str | Path, record: dict, arrays: Mapping[str, np.ndarray]) -> dict:
    """Validate, then write arrays and record atomically (the record commits the pair)."""

    record = dict(record)
    record["counts"] = annotation_counts(arrays, record["dataset"])
    json_path, npz_path = sequence_paths(dataset_dir, record["sequence"])
    record["annotations"] = {"path": npz_path.name, "sha256": ""}
    validate_sequence(record, arrays)
    record["annotations"]["sha256"] = write_annotations_atomic(npz_path, arrays)
    write_json_atomic(json_path, record)
    return record


def manifest_entry(record: Mapping[str, Any], dataset_dir: str | Path) -> dict[str, Any]:
    json_path, _ = sequence_paths(dataset_dir, record["sequence"])
    counts = record["counts"]
    return {
        "dataset": record["dataset"],
        "sequence": record["sequence"],
        "split": record["split"],
        "recording": record["recording"],
        "record_sha256": sha256_file(json_path),
        "steps": counts["steps"],
        "annotated_steps": counts["steps_by_status"]["annotated"],
        "positive_boxes": counts["positive_boxes"],
    }


def write_dataset_manifest(
    output_root: str | Path,
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    complete: bool,
    plan: Mapping[str, Any],
) -> Path:
    """Write ``<output_root>/<dataset>/manifest.json`` for the converted records."""

    dataset_dir = Path(output_root) / dataset
    entries = sorted(
        (manifest_entry(record, dataset_dir) for record in records),
        key=lambda entry: (entry["split"], entry["sequence"]),
    )
    totals: dict[str, dict[str, int]] = {}
    for entry in entries:
        split = totals.setdefault(
            entry["split"], {"sequences": 0, "steps": 0, "annotated_steps": 0, "positive_boxes": 0}
        )
        split["sequences"] += 1
        for key in ("steps", "annotated_steps", "positive_boxes"):
            split[key] += entry[key]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "family": DATASET_FAMILY[dataset],
        "complete": bool(complete),
        "settings": settings,
        "settings_sha256": settings_hash(settings),
        "plan": plan,
        "totals_by_split": totals,
        "sequences": entries,
    }
    path = dataset_dir / MANIFEST_FILENAME
    write_json_atomic(path, manifest)
    return path


# --------------------------------------------------------------------------- split overlap


def check_split_overlap(
    entries: Iterable[Mapping[str, Any]],
    allowed_recording_overlap: Mapping[str, Sequence[str] | str],
) -> dict[str, Any]:
    """Reject split overlap at sequence and original-recording level.

    A sequence may appear only once per family (label precedence must already have
    removed duplicates). A recording that hosts a held-out sequence (any split other
    than ``train``) may share recordings with training sequences only inside the same
    dataset and only when listed in ``allowed_recording_overlap`` (``"*"`` allows every
    recording of that dataset). Returns the permitted overlaps for the audit record.
    """

    entries = list(entries)
    seen: dict[tuple[str, str], str] = {}
    for entry in entries:
        key = (DATASET_FAMILY[entry["dataset"]], entry["sequence"])
        if key in seen:
            raise SchemaError(
                f"sequence {key[1]} appears in {seen[key]} and {entry['dataset']}; "
                "label precedence must keep only one copy"
            )
        seen[key] = entry["dataset"]

    held_out: dict[tuple[str, str], set[str]] = {}
    training: dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        key = (DATASET_FAMILY[entry["dataset"]], entry["recording"])
        target = training if entry["split"] in TRAINING_SPLITS else held_out
        target.setdefault(key, set()).add(entry["dataset"])

    permitted = []
    for key, held_datasets in sorted(held_out.items()):
        train_datasets = training.get(key, set())
        for held_dataset in sorted(held_datasets):
            for train_dataset in sorted(train_datasets):
                allowed = allowed_recording_overlap.get(held_dataset, ())
                if train_dataset == held_dataset and (allowed == "*" or key[1] in allowed):
                    permitted.append({"dataset": held_dataset, "recording": key[1]})
                    continue
                raise SchemaError(
                    f"recording {key[1]} hosts {held_dataset} held-out data and "
                    f"{train_dataset} training data"
                )
    return {"permitted_recording_overlaps": permitted}


def validate_output_root(
    output_root: str | Path,
    allowed_recording_overlap: Mapping[str, Sequence[str] | str],
    deep: bool = False,
) -> dict[str, Any]:
    """Validate every dataset manifest under ``output_root`` and their joint splits."""

    root = Path(output_root)
    entries = []
    summary: dict[str, Any] = {"datasets": {}}
    for dataset in DATASETS:
        manifest_path = root / dataset / MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        _require(manifest.get("schema_version") == SCHEMA_VERSION, f"{dataset}: schema mismatch")
        _require(manifest.get("dataset") == dataset, f"{manifest_path}: dataset mismatch")
        for entry in manifest["sequences"]:
            _require(entry["dataset"] == dataset, f"{manifest_path}: foreign entry")
            if deep:
                load_sequence(root / dataset, entry)
        entries.extend(manifest["sequences"])
        summary["datasets"][dataset] = {
            "sequences": len(manifest["sequences"]),
            "complete": manifest.get("complete"),
        }
    summary.update(check_split_overlap(entries, allowed_recording_overlap))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate large event MOT manifests.")
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--config", required=True, help="data configuration JSON")
    parser.add_argument("--deep", action="store_true", help="re-validate every sequence file")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    summary = validate_output_root(
        args.manifest_root, config["recording_overlap_allowed"], deep=args.deep
    )
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
