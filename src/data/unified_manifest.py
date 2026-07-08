"""Manifest-backed dense dataset for multi-dataset event pretraining.

The manifest format is intentionally small and dataset-agnostic. Converters for
external datasets should write one JSON object per timestamp with dense
representation paths and boxes in absolute coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.data.dense_targets import DenseBox, encode_dense_targets
from src.data.representations import representation_components

DEFAULT_CLASS_MAP = {
    "vehicle": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "train": 0,
    "pedestrian": 1,
    "person": 1,
    "bicycle": 2,
    "motorcycle": 2,
    "two_wheeler": 2,
    "object": 3,
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records from a manifest."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record in {path}:{line_number}") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL manifest."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        archive = np.load(path)
        if "array" in archive:
            return np.asarray(archive["array"], dtype=np.float32)
        if "events" in archive:
            return np.asarray(archive["events"], dtype=np.float32)
        first_key = archive.files[0]
        return np.asarray(archive[first_key], dtype=np.float32)
    return np.asarray(np.load(path), dtype=np.float32)


def _resize_chw(array: np.ndarray, height: int, width: int) -> np.ndarray:
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected representation with shape CxHxW or HxW, got {array.shape}.")
    if array.shape[-2:] == (height, width):
        return array.astype(np.float32, copy=False)
    tensor = torch.from_numpy(array).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy().astype(np.float32, copy=False)


class UnifiedDenseRepresentationDataset(Dataset):
    """Dense detector dataset built from a multi-dataset JSONL manifest.

    Required manifest fields:
    - ``width`` and ``height``: source representation coordinate system.
    - ``boxes``: ``[x1, y1, x2, y2]`` boxes in source coordinates.
    - ``labels``: strings from ``class_map`` or integer zero-based class ids.

    Representation input can be provided in one of two ways:
    - ``representation_path``: one pre-concatenated ``.npy``/``.npz`` tensor.
    - ``representation_paths``: mapping from component name, e.g.
      ``event_frame`` or ``voxel_grid``, to ``.npy``/``.npz`` tensors.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        representation: str,
        class_map: dict[str, int] | None = None,
        feature_stride: int = 8,
        positive_radius: int = 1,
        image_width: int = EVENT_WIDTH,
        image_height: int = EVENT_HEIGHT,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.rows = read_jsonl(self.manifest_path)
        self.representation = representation
        self.components = representation_components(representation)
        self.class_map = dict(DEFAULT_CLASS_MAP if class_map is None else class_map)
        self.feature_stride = int(feature_stride)
        self.positive_radius = int(positive_radius)
        self.image_width = int(image_width)
        self.image_height = int(image_height)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_representation(self, row: dict[str, Any]) -> np.ndarray:
        if "representation_path" in row:
            array = _load_array(_resolve_path(self.base_dir, row["representation_path"]))
            return _resize_chw(array, self.image_height, self.image_width)

        paths = row.get("representation_paths")
        if not isinstance(paths, dict):
            raise ValueError(
                "Manifest row must contain representation_path or representation_paths."
            )
        arrays = []
        for component in self.components:
            if component not in paths:
                raise ValueError(f"Missing component '{component}' in representation_paths.")
            component_path = _resolve_path(self.base_dir, paths[component])
            component_array = _load_array(component_path)
            arrays.append(_resize_chw(component_array, self.image_height, self.image_width))
        return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)

    def _label_to_class_id(self, label: str | int) -> int:
        if isinstance(label, int):
            return label
        if label not in self.class_map:
            raise ValueError(f"Unknown label '{label}'. Extend class_map before training.")
        return int(self.class_map[label])

    def _boxes(self, row: dict[str, Any]) -> list[DenseBox]:
        source_width = float(row.get("width", self.image_width))
        source_height = float(row.get("height", self.image_height))
        sx = self.image_width / source_width
        sy = self.image_height / source_height
        labels = row.get("labels", [])
        boxes = []
        for raw_box, label in zip(row.get("boxes", []), labels):
            x1, y1, x2, y2 = [float(value) for value in raw_box]
            boxes.append(
                DenseBox(
                    left=x1 * sx,
                    top=y1 * sy,
                    width=(x2 - x1) * sx,
                    height=(y2 - y1) * sy,
                    class_id=self._label_to_class_id(label),
                )
            )
        return boxes

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        events = self._load_representation(row)
        label = encode_dense_targets(
            boxes=self._boxes(row),
            image_width=self.image_width,
            image_height=self.image_height,
            feature_stride=self.feature_stride,
            positive_radius=self.positive_radius,
            class_offset=1,
        )
        return {
            "events": events,
            "label": label,
            "meta": {
                "dataset": row.get("dataset", "unknown"),
                "sequence": row.get("sequence", "unknown"),
                "timestamp": int(row.get("timestamp_us", row.get("timestamp", index))),
                "frame_index": int(row.get("frame_index", index)),
                "num_annotations": len(row.get("boxes", [])),
            },
        }
