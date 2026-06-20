"""Shared DSEC-MOT dataset and export helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.data.dataset import _find_dataset, _import_h5, _timestamp_window_indices

CLASS_NAMES = {
    0: "car",
    1: "pedestrian",
    2: "bicycle",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "train",
}


@dataclass(frozen=True)
class Annotation:
    timestamp: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    class_id: int


@dataclass(frozen=True)
class DetectionRecord:
    frame_index: int
    timestamp: int
    class_id: int
    score: float
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_annotations(path: Path) -> list[Annotation]:
    rows: list[Annotation] = []
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append(
                Annotation(
                    timestamp=int(row[0].strip()),
                    track_id=int(row[1].strip()),
                    left=float(row[2].strip()),
                    top=float(row[3].strip()),
                    width=float(row[4].strip()),
                    height=float(row[5].strip()),
                    class_id=int(row[6].strip()),
                )
            )
    return rows


def load_image_timestamps(path: Path) -> list[int]:
    return [
        int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_event_file(events_h5: Path):
    h5py, np_h5 = _import_h5()
    handle = h5py.File(events_h5, "r")
    x = _find_dataset(handle, ["events/x", "x"])
    y = _find_dataset(handle, ["events/y", "y"])
    p = _find_dataset(handle, ["events/p", "p"])
    t = _find_dataset(handle, ["events/t", "t"])
    ms_to_idx = _find_dataset(handle, ["ms_to_idx", "events/ms_to_idx"])
    try:
        t_offset = int(_find_dataset(handle, ["t_offset", "events/t_offset"])[()])
    except KeyError:
        t_offset = 0
    return handle, x, y, p, t, ms_to_idx, t_offset, np_h5


def read_events(
    x, y, p, t, ms_to_idx, t_offset: int, timestamp_us: int, window_us: int
) -> np.ndarray:
    start_us = timestamp_us - window_us
    start_idx, end_idx = _timestamp_window_indices(ms_to_idx, t, t_offset, start_us, timestamp_us)

    dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.bool_)])
    if end_idx <= start_idx:
        return np.empty(0, dtype=dtype)

    xs = x[start_idx:end_idx].astype(np.uint16)
    ys = y[start_idx:end_idx].astype(np.uint16)
    ts = t[start_idx:end_idx].astype(np.int64) + int(t_offset)
    ps = p[start_idx:end_idx].astype(np.bool_)
    events = np.empty(xs.shape[0], dtype=dtype)
    events["x"] = xs
    events["y"] = ys
    events["t"] = ts
    events["p"] = ps
    return events


def load_detection_export(path: Path) -> dict:
    with path.open("rt", encoding="utf-8") as handle:
        return json.load(handle)
