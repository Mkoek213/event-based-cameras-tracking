"""Shared helpers for external dataset converters.

Converters should be deliberately thin: parse one source dataset, write dense
representation tensors, and emit rows compatible with UnifiedDenseRepresentationDataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data.unified_manifest import write_jsonl

TRAFFIC_CLASS_MAP = {
    "car": "vehicle",
    "cars": "vehicle",
    "vehicle": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "bike": "two_wheeler",
}


def normalise_label(label: str) -> str:
    """Map dataset-specific traffic labels to the shared coarse taxonomy."""

    return TRAFFIC_CLASS_MAP.get(label.strip().lower(), "object")


def save_representation(path: str | Path, array: np.ndarray) -> str:
    """Save one dense representation tensor and return its path as a string."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array.astype(np.float32, copy=False))
    return str(output)


def write_split_manifest(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a converter-produced split manifest."""

    write_jsonl(path, list(rows))
