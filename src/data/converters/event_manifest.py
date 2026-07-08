"""Build unified manifest rows from external event-detection datasets.

The converters in this module deliberately target a small common contract:
event windows are converted to dense EF/VG tensors and paired with boxes in
absolute image coordinates. Dataset-specific wrappers only need to provide event
files and annotation files.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from src.data.preprocessing import EventPreprocessor
from src.data.unified_manifest import write_jsonl

EVENT_DTYPE = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.bool_)])

DEFAULT_COARSE_CLASS_NAMES = ("vehicle", "pedestrian", "two_wheeler", "object")
DEFAULT_TRAFFIC_ID_MAP = {
    0: "vehicle",
    1: "pedestrian",
    2: "two_wheeler",
    3: "object",
}


@dataclass(frozen=True)
class BoxRecord:
    """One normalized dataset-agnostic annotation."""

    timestamp_us: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    track_id: int | None = None


@dataclass(frozen=True)
class RepresentationConfig:
    """Dense representation cache settings."""

    width: int
    height: int
    num_bins: int
    time_window_us: int
    dtype: str = "float16"


def parse_class_id_map(value: str | None) -> dict[int, str]:
    """Parse a CLI class map like ``0:vehicle,1:pedestrian``."""

    if value is None or not value.strip():
        return dict(DEFAULT_TRAFFIC_ID_MAP)
    parsed: dict[int, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        raw_key, raw_label = item.split(":", maxsplit=1)
        parsed[int(raw_key.strip())] = raw_label.strip()
    return parsed


def event_array(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return a canonical structured event array sorted by timestamp."""

    events = np.empty(len(t), dtype=EVENT_DTYPE)
    events["x"] = np.asarray(x, dtype=np.uint16)
    events["y"] = np.asarray(y, dtype=np.uint16)
    events["t"] = np.asarray(t, dtype=np.int64)
    events["p"] = np.asarray(p).astype(bool)
    if len(events) > 1 and np.any(events["t"][1:] < events["t"][:-1]):
        events = events[np.argsort(events["t"])]
    return events


def select_event_window(
    events: np.ndarray,
    end_timestamp_us: int,
    time_window_us: int,
) -> np.ndarray:
    """Slice an event array to ``[end - window, end]`` using sorted timestamps."""

    if len(events) == 0:
        return events
    timestamps = events["t"]
    start = int(end_timestamp_us) - int(time_window_us)
    left = int(np.searchsorted(timestamps, start, side="left"))
    right = int(np.searchsorted(timestamps, int(end_timestamp_us), side="right"))
    return events[left:right]


def _as_float_array(array: np.ndarray, field_names: tuple[str, ...]) -> np.ndarray | None:
    if array.dtype.names is None:
        return None
    for name in field_names:
        if name in array.dtype.names:
            return np.asarray(array[name], dtype=np.float32)
    return None


def _as_int_array(array: np.ndarray, field_names: tuple[str, ...]) -> np.ndarray | None:
    if array.dtype.names is None:
        return None
    for name in field_names:
        if name in array.dtype.names:
            return np.asarray(array[name], dtype=np.int64)
    return None


def load_structured_boxes(
    path: str | Path,
    class_id_map: dict[int, str] | None = None,
    default_label: str = "object",
    default_timestamp_us: int = 0,
) -> list[BoxRecord]:
    """Load bbox rows from a structured ``.npy`` file.

    Supported field aliases cover DSEC-like and Prophesee-like arrays:
    ``t/ts/timestamp``, ``x/y/w/h`` or ``x1/y1/x2/y2`` and optional
    ``class_id`` / ``track_id``.
    """

    class_id_map = dict(DEFAULT_TRAFFIC_ID_MAP if class_id_map is None else class_id_map)
    array = np.load(path)
    if array.dtype.names is None:
        raise ValueError(f"{path} must be a structured numpy array with bbox fields.")

    timestamps = _as_int_array(array, ("t", "ts", "timestamp", "timestamp_us"))
    if timestamps is None:
        timestamps = np.full(len(array), int(default_timestamp_us), dtype=np.int64)

    x1 = _as_float_array(array, ("x1", "x_min", "left"))
    y1 = _as_float_array(array, ("y1", "y_min", "top"))
    x2 = _as_float_array(array, ("x2", "x_max", "right"))
    y2 = _as_float_array(array, ("y2", "y_max", "bottom"))
    if x1 is None or y1 is None or x2 is None or y2 is None:
        x = _as_float_array(array, ("x", "left"))
        y = _as_float_array(array, ("y", "top"))
        w = _as_float_array(array, ("w", "width"))
        h = _as_float_array(array, ("h", "height"))
        if x is None or y is None or w is None or h is None:
            raise ValueError(f"{path} does not contain recognized bbox fields.")
        x1, y1, x2, y2 = x, y, x + w, y + h

    class_ids = _as_int_array(array, ("class_id", "class", "label", "category_id"))
    track_ids = _as_int_array(array, ("track_id", "track", "id", "obj_id"))

    records: list[BoxRecord] = []
    for index in range(len(array)):
        if x2[index] <= x1[index] or y2[index] <= y1[index]:
            continue
        class_id = int(class_ids[index]) if class_ids is not None else None
        label = class_id_map.get(class_id, default_label) if class_id is not None else default_label
        records.append(
            BoxRecord(
                timestamp_us=int(timestamps[index]),
                x1=float(x1[index]),
                y1=float(y1[index]),
                x2=float(x2[index]),
                y2=float(y2[index]),
                label=label,
                track_id=int(track_ids[index]) if track_ids is not None else None,
            )
        )
    return records


def group_boxes_by_timestamp(boxes: Iterable[BoxRecord]) -> dict[int, list[BoxRecord]]:
    """Group box records by annotation timestamp."""

    grouped: dict[int, list[BoxRecord]] = {}
    for box in boxes:
        grouped.setdefault(int(box.timestamp_us), []).append(box)
    return grouped


def save_dense_representations(
    events: np.ndarray,
    output_prefix: Path,
    config: RepresentationConfig,
) -> dict[str, str]:
    """Save EF and VG tensors for one sample and return manifest paths."""

    ef = EventPreprocessor(
        config.height,
        config.width,
        representation="event_frame",
        num_bins=config.num_bins,
    )(events)
    vg = EventPreprocessor(
        config.height,
        config.width,
        representation="voxel_grid",
        num_bins=config.num_bins,
    )(events)
    dtype = np.float16 if config.dtype == "float16" else np.float32
    ef_path = output_prefix.with_name(output_prefix.name + "_ef.npy")
    vg_path = output_prefix.with_name(output_prefix.name + "_vg.npy")
    ef_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ef_path, ef.astype(dtype, copy=False))
    np.save(vg_path, vg.astype(dtype, copy=False))
    return {"event_frame": str(ef_path), "voxel_grid": str(vg_path)}


def make_manifest_row(
    dataset: str,
    sequence: str,
    timestamp_us: int,
    frame_index: int,
    width: int,
    height: int,
    representation_paths: dict[str, str],
    boxes: list[BoxRecord],
) -> dict[str, Any]:
    """Create one JSONL manifest row."""

    return {
        "dataset": dataset,
        "sequence": sequence,
        "timestamp_us": int(timestamp_us),
        "frame_index": int(frame_index),
        "width": int(width),
        "height": int(height),
        "representation_paths": representation_paths,
        "boxes": [[box.x1, box.y1, box.x2, box.y2] for box in boxes],
        "labels": [box.label for box in boxes],
        "track_ids": [box.track_id for box in boxes],
    }


def split_rows(
    rows: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split rows into train and validation lists."""

    if not rows:
        return [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    val_size = int(round(len(rows) * val_fraction))
    val_size = min(max(val_size, 1), len(rows) - 1) if len(rows) > 1 else 0
    val_indices = set(indices[:val_size].tolist())
    train = [row for index, row in enumerate(rows) if index not in val_indices]
    val = [row for index, row in enumerate(rows) if index in val_indices]
    return train, val


def write_train_val_manifests(
    output_dir: Path,
    rows: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
    train_name: str = "pretrain_train.jsonl",
    val_name: str = "pretrain_val.jsonl",
) -> tuple[Path, Path]:
    """Write train/val JSONL manifests and return their paths."""

    train_rows, val_rows = split_rows(rows, val_fraction=val_fraction, seed=seed)
    manifest_dir = output_dir / "manifests"
    train_path = manifest_dir / train_name
    val_path = manifest_dir / val_name
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    return train_path, val_path


def iter_sampled_timestamps(
    grouped_boxes: dict[int, list[BoxRecord]],
    max_samples: int,
    sample_stride: int,
) -> Iterator[tuple[int, list[BoxRecord]]]:
    """Yield timestamp groups with deterministic stride/limit sampling."""

    yielded = 0
    stride = max(int(sample_stride), 1)
    for index, timestamp in enumerate(sorted(grouped_boxes)):
        if index % stride != 0:
            continue
        yield timestamp, grouped_boxes[timestamp]
        yielded += 1
        if max_samples > 0 and yielded >= max_samples:
            break


def read_h5_event_file(path: str | Path) -> np.ndarray:
    """Read events from an HDF5 file with flexible key discovery.

    DSEC stores event timestamps relative to a per-file ``t_offset``. When that
    scalar is present, add it so event timestamps share the annotation time
    domain.
    """

    import h5py
    import hdf5plugin  # noqa: F401

    def find_dataset(handle: h5py.File, names: tuple[str, ...]):
        for name in names:
            if name in handle:
                return handle[name]
        matches = []

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in names:
                matches.append(obj)

        handle.visititems(visitor)
        if not matches:
            raise KeyError(f"Could not find any of {names} in {path}.")
        return matches[0]

    def find_optional_scalar(handle: h5py.File, names: tuple[str, ...]) -> int:
        try:
            dataset = find_dataset(handle, names)
        except KeyError:
            return 0
        return int(np.asarray(dataset).item())

    with h5py.File(path, "r") as handle:
        x = np.asarray(find_dataset(handle, ("x", "xs")), dtype=np.uint16)
        y = np.asarray(find_dataset(handle, ("y", "ys")), dtype=np.uint16)
        t = np.asarray(find_dataset(handle, ("t", "ts", "timestamp")), dtype=np.int64)
        p = np.asarray(find_dataset(handle, ("p", "polarity")), dtype=bool)
        t_offset = find_optional_scalar(handle, ("t_offset",))
    return event_array(x, y, t + t_offset, p)


def read_metavision_dat(path: str | Path, max_events: int = 0) -> np.ndarray:
    """Read a Prophesee/Metavision CD ``.dat`` file.

    The function first tries Metavision's official ``EventDatReader`` when
    available. The fallback parser handles the common DAT v2 CD encoding used by
    Prophesee detection datasets.
    """

    try:
        from metavision_core.event_io import EventDatReader

        reader = EventDatReader(str(path))
        events = reader.load_n_events(max_events) if max_events > 0 else reader.load_delta_t(-1)
        return event_array(events["x"], events["y"], events["t"], events["p"])
    except Exception:
        return _read_metavision_dat_fallback(path, max_events=max_events)


def _read_metavision_dat_fallback(path: str | Path, max_events: int = 0) -> np.ndarray:
    with Path(path).open("rb") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                return np.empty(0, dtype=EVENT_DTYPE)
            if not line.startswith(b"%"):
                handle.seek(position)
                break
        payload = handle.read()

    event_size = 8
    if len(payload) < event_size:
        return np.empty(0, dtype=EVENT_DTYPE)
    count = len(payload) // event_size
    if max_events > 0:
        count = min(count, int(max_events))
    raw = np.frombuffer(
        payload[: count * event_size], dtype=np.dtype([("t", "<u4"), ("data", "<u4")])
    )
    data = raw["data"]
    x = data & 0x3FFF
    y = (data >> 14) & 0x3FFF
    p = (data >> 28) & 0x1
    return event_array(x, y, raw["t"], p)


def inspect_numpy(path: str | Path, max_rows: int = 3) -> dict[str, Any]:
    """Return dtype/shape/sample metadata for a numpy file."""

    array = np.load(path)
    return {
        "path": str(path),
        "shape": tuple(int(value) for value in array.shape),
        "dtype": str(array.dtype),
        "fields": list(array.dtype.names or ()),
        "sample": repr(array[:max_rows]),
    }


def inspect_h5(path: str | Path) -> dict[str, Any]:
    """Return a shallow HDF5 key listing."""

    import h5py
    import hdf5plugin  # noqa: F401

    datasets: list[str] = []
    with h5py.File(path, "r") as handle:

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append(f"{name}: shape={obj.shape} dtype={obj.dtype}")

        handle.visititems(visitor)
    return {"path": str(path), "datasets": datasets[:50]}


def csv_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact CSV with dataset/sequence/sample counts."""

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("dataset", "unknown")), str(row.get("sequence", "unknown")))
        counts[key] = counts.get(key, 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "sequence", "samples"])
        for (dataset, sequence), count in sorted(counts.items()):
            writer.writerow([dataset, sequence, count])


def print_json(data: Any) -> None:
    """Pretty-print JSON-compatible data."""

    print(json.dumps(data, indent=2, ensure_ascii=False))
