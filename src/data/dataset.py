"""DSEC-MOT dataset utilities for event-based detection training."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


EVENT_WIDTH = 640
EVENT_HEIGHT = 480


@dataclass(frozen=True)
class Annotation:
    timestamp: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    class_id: int


def _import_h5() -> tuple[object, object]:
    import hdf5plugin  # noqa: F401  Registers HDF5 compression plugins.
    import h5py

    return h5py, np


def _find_dataset(handle, candidates: list[str]):
    for candidate in candidates:
        if candidate in handle:
            return handle[candidate]
    for candidate in candidates:
        node = handle
        found = True
        for part in candidate.split("/"):
            if not part:
                continue
            if part not in node:
                found = False
                break
            node = node[part]
        if found:
            return node
    available = list(handle.keys())
    raise KeyError(f"Could not find any of {candidates}. Top-level keys: {available}")


def _timestamp_window_indices(ms_to_idx, t, t_offset: int, start_us: int, end_us: int):
    start_ms = max((start_us - t_offset) // 1000, 0)
    end_ms = max((end_us - t_offset) // 1000, 0)

    start_idx = int(ms_to_idx[min(start_ms, len(ms_to_idx) - 1)])
    next_ms = end_ms + 1
    end_idx = int(ms_to_idx[next_ms]) if next_ms < len(ms_to_idx) else len(t)

    raw_t = t[start_idx:end_idx]
    if raw_t.size == 0:
        return start_idx, start_idx

    timestamps = raw_t.astype(np.int64) + t_offset
    mask = (timestamps >= start_us) & (timestamps <= end_us)
    if not np.any(mask):
        return start_idx, start_idx

    local = np.flatnonzero(mask)
    return start_idx + int(local[0]), start_idx + int(local[-1]) + 1


class EventDataset:
    """PyTorch-style dataset for DSEC-MOT event-based detection.

    Each sample contains:
      - ``events``: dense event representation tensor-like numpy array
      - ``label``: tuple ``(cls_targets, bbox_targets, pos_mask)``
      - ``meta``: sample metadata
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        time_window_us: int = 50_000,
        feature_stride: int = 8,
        include_unannotated: bool = False,
        class_offset: int = 1,
        positive_radius: int = 1,
        sequences: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.time_window_us = int(time_window_us)
        self.feature_stride = int(feature_stride)
        self.include_unannotated = include_unannotated
        self.class_offset = int(class_offset)
        self.positive_radius = max(int(positive_radius), 0)
        self.sequence_filter = set(sequences) if sequences is not None else None
        self._samples = self._load_manifest()
        self._event_cache: dict[str, tuple] = {}
        self._h5py = None

    def _load_manifest(self) -> list[dict]:
        split_dir = self.root / self.split
        ann_dir = self.root / "annotations" / self.split
        if not split_dir.exists():
            return []

        samples: list[dict] = []
        for seq_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            sequence = seq_dir.name
            if self.sequence_filter is not None and sequence not in self.sequence_filter:
                continue
            ann_path = ann_dir / f"{sequence}.txt"
            if not ann_path.exists():
                continue

            grouped = self._load_annotations(ann_path)
            timestamps = self._load_timestamps(seq_dir / f"{sequence}_image_timestamps.txt")
            for frame_index, timestamp in enumerate(timestamps):
                annotations = grouped.get(timestamp, [])
                if annotations or self.include_unannotated:
                    samples.append(
                        {
                            "sequence": sequence,
                            "seq_dir": seq_dir,
                            "timestamp": timestamp,
                            "frame_index": frame_index,
                            "annotations": annotations,
                        }
                    )
        return samples

    @staticmethod
    def _load_annotations(path: Path) -> dict[int, list[Annotation]]:
        grouped: dict[int, list[Annotation]] = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                annotation = Annotation(
                    timestamp=int(row[0].strip()),
                    track_id=int(row[1].strip()),
                    left=float(row[2].strip()),
                    top=float(row[3].strip()),
                    width=float(row[4].strip()),
                    height=float(row[5].strip()),
                    class_id=int(row[6].strip()),
                )
                grouped[annotation.timestamp].append(annotation)
        return dict(grouped)

    @staticmethod
    def _load_timestamps(path: Path) -> list[int]:
        return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def sequence_names(self) -> list[str]:
        return sorted({sample["sequence"] for sample in self._samples})

    def _get_event_handle(self, sequence: str, seq_dir: Path):
        if sequence in self._event_cache:
            return self._event_cache[sequence]

        if self._h5py is None:
            self._h5py, _ = _import_h5()

        handle = self._h5py.File(seq_dir / "events_left" / "events.h5", "r")
        x = _find_dataset(handle, ["events/x", "x"])
        y = _find_dataset(handle, ["events/y", "y"])
        p = _find_dataset(handle, ["events/p", "p"])
        t = _find_dataset(handle, ["events/t", "t"])
        ms_to_idx = _find_dataset(handle, ["ms_to_idx", "events/ms_to_idx"])
        try:
            t_offset = int(_find_dataset(handle, ["t_offset", "events/t_offset"])[()])
        except KeyError:
            t_offset = 0
        self._event_cache[sequence] = (handle, x, y, p, t, ms_to_idx, t_offset)
        return self._event_cache[sequence]

    def _read_events(self, sequence: str, seq_dir: Path, timestamp_us: int) -> np.ndarray:
        _, x, y, p, t, ms_to_idx, t_offset = self._get_event_handle(sequence, seq_dir)
        start_us = timestamp_us - self.time_window_us
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

    def _encode_targets(self, annotations: list[Annotation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feat_h = EVENT_HEIGHT // self.feature_stride
        feat_w = EVENT_WIDTH // self.feature_stride
        cls_targets = np.zeros((feat_h, feat_w), dtype=np.int64)
        bbox_targets = np.zeros((4, feat_h, feat_w), dtype=np.float32)
        pos_mask = np.zeros((feat_h, feat_w), dtype=bool)
        assignment_distance = np.full((feat_h, feat_w), np.inf, dtype=np.float32)

        for ann in annotations:
            x0 = max(0.0, ann.left)
            y0 = max(0.0, ann.top)
            x1 = min(float(EVENT_WIDTH - 1), ann.left + ann.width)
            y1 = min(float(EVENT_HEIGHT - 1), ann.top + ann.height)
            if x1 <= x0 or y1 <= y0:
                continue

            center_x = 0.5 * (x0 + x1)
            center_y = 0.5 * (y0 + y1)
            center_gx = center_x / self.feature_stride - 0.5
            center_gy = center_y / self.feature_stride - 0.5
            gx = int(round(center_gx))
            gy = int(round(center_gy))
            if gx < 0 or gx >= feat_w or gy < 0 or gy >= feat_h:
                continue

            # Keep assignment radius explicit and small. Broad adaptive positives
            # encourage foreground blobs instead of precise detections.
            radius = self.positive_radius

            x_start = max(gx - radius, 0)
            x_end = min(gx + radius, feat_w - 1)
            y_start = max(gy - radius, 0)
            y_end = min(gy + radius, feat_h - 1)

            for gy_idx in range(y_start, y_end + 1):
                for gx_idx in range(x_start, x_end + 1):
                    dx = gx_idx - center_gx
                    dy = gy_idx - center_gy
                    distance = float(dx * dx + dy * dy)
                    if distance >= assignment_distance[gy_idx, gx_idx]:
                        continue

                    cls_targets[gy_idx, gx_idx] = ann.class_id + self.class_offset
                    pos_mask[gy_idx, gx_idx] = True
                    assignment_distance[gy_idx, gx_idx] = distance

                    cell_cx = (gx_idx + 0.5) * self.feature_stride
                    cell_cy = (gy_idx + 0.5) * self.feature_stride
                    bbox_targets[0, gy_idx, gx_idx] = (cell_cx - x0) / self.feature_stride
                    bbox_targets[1, gy_idx, gx_idx] = (cell_cy - y0) / self.feature_stride
                    bbox_targets[2, gy_idx, gx_idx] = (x1 - cell_cx) / self.feature_stride
                    bbox_targets[3, gy_idx, gx_idx] = (y1 - cell_cy) / self.feature_stride

        return cls_targets, bbox_targets, pos_mask

    def __getitem__(self, idx: int) -> dict:
        sample_info = self._samples[idx]
        events = self._read_events(
            sequence=sample_info["sequence"],
            seq_dir=sample_info["seq_dir"],
            timestamp_us=sample_info["timestamp"],
        )
        label = self._encode_targets(sample_info["annotations"])

        sample = {
            "events": events,
            "label": label,
            "meta": {
                "sequence": sample_info["sequence"],
                "timestamp": sample_info["timestamp"],
                "frame_index": sample_info["frame_index"],
                "num_annotations": len(sample_info["annotations"]),
            },
        }

        if self.transform is not None:
            sample["events"] = self.transform(sample["events"])
        if self.target_transform is not None:
            sample["label"] = self.target_transform(sample["label"])
        return sample
