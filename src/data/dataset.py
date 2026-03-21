"""Event-based camera dataset class."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np


class EventDataset:
    """PyTorch-style dataset for event-based camera recordings.

    Each sample is a dict with keys:
        - ``events``: structured numpy array with fields (x, y, t, p)
        - ``label``:  target annotation (bounding boxes / track IDs)
        - ``meta``:   recording metadata (sensor size, file path, …)
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self._samples: list[Path] = self._load_manifest()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[Path]:
        manifest = self.root / "datasets" / f"{self.split}.txt"
        if not manifest.exists():
            return []
        return [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip()]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample_path = self._samples[idx]
        events = self._read_events(sample_path)
        sample = {"events": events, "label": None, "meta": {"path": str(sample_path)}}
        if self.transform is not None:
            sample["events"] = self.transform(sample["events"])
        if self.target_transform is not None and sample["label"] is not None:
            sample["label"] = self.target_transform(sample["label"])
        return sample

    @staticmethod
    def _read_events(path: Path) -> np.ndarray:
        """Stub: load events from *path* and return a structured array."""
        dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.bool_)])
        return np.empty(0, dtype=dtype)
