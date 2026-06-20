"""DSEC-MOT dataset adapter for torchvision object detectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH, EventDataset
from src.data.representations import BenchmarkRepresentation, representation_components


class ErosSnapshotStore:
    """Read memory-mapped EROS snapshots generated for DSEC-MOT sequences."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        self._metadata: dict[tuple[str, str], dict] = {}

    def _load(self, split: str, sequence: str) -> tuple[np.ndarray, dict]:
        key = (split, sequence)
        if key not in self._arrays:
            sequence_root = self.root / split / sequence
            array_path = sequence_root / "snapshots.npy"
            metadata_path = sequence_root / "metadata.json"
            if not array_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(
                    f"Missing EROS cache for {split}/{sequence}. "
                    "Run python -m src.data.eros_precompute first."
                )
            self._arrays[key] = np.load(array_path, mmap_mode="r")
            self._metadata[key] = json.loads(metadata_path.read_text(encoding="utf-8"))
        return self._arrays[key], self._metadata[key]

    def get(self, split: str, sequence: str, frame_index: int, timestamp: int) -> np.ndarray:
        snapshots, metadata = self._load(split, sequence)
        timestamps = metadata["timestamps"]
        if frame_index >= len(timestamps) or int(timestamps[frame_index]) != int(timestamp):
            raise ValueError(
                f"EROS cache is not aligned with {split}/{sequence} frame {frame_index}."
            )
        return snapshots[frame_index]


class DSECPretrainedDetectionDataset(Dataset):
    """Return dense event representations and torchvision detection targets."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        sequences: Sequence[str],
        representation: str,
        num_bins: int = 5,
        time_window_us: int = 50_000,
        include_unannotated: bool = False,
        eros_cache_root: str | Path = "data/cache/dsec_mot_eros",
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self.representation = representation
        self.transform = BenchmarkRepresentation(
            representation, num_bins, EVENT_HEIGHT, EVENT_WIDTH
        )
        self.needs_eros = "eros" in representation_components(representation)
        self.eros_store = ErosSnapshotStore(eros_cache_root) if self.needs_eros else None
        self.dataset = EventDataset(
            root=root,
            split=split,
            sequences=sequences,
            time_window_us=time_window_us,
            include_unannotated=include_unannotated,
            class_ids=class_ids,
        )
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def sequence_names(self) -> list[str]:
        return self.dataset.sequence_names

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
        sample = self.dataset._samples[index]
        events = self.dataset._read_events(
            sample["sequence"], sample["seq_dir"], sample["timestamp"]
        )
        eros = None
        if self.eros_store is not None:
            eros = self.eros_store.get(
                self.split, sample["sequence"], sample["frame_index"], sample["timestamp"]
            )
        image = torch.from_numpy(self.transform(events, eros=eros)).float()

        boxes: list[list[float]] = []
        labels: list[int] = []
        for annotation in sample["annotations"]:
            x1 = max(0.0, annotation.left)
            y1 = max(0.0, annotation.top)
            x2 = min(float(EVENT_WIDTH), annotation.left + annotation.width)
            y2 = min(float(EVENT_HEIGHT), annotation.top + annotation.height)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(annotation.class_id + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        metadata = {
            "sequence": sample["sequence"],
            "timestamp": sample["timestamp"],
            "frame_index": sample["frame_index"],
        }
        return image, target, metadata


class DSECDenseRepresentationDataset(Dataset):
    """Return dense representations and grid targets for SimpleDenseDetector."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        sequences: Sequence[str],
        representation: str,
        num_bins: int = 5,
        time_window_us: int = 50_000,
        feature_stride: int = 8,
        positive_radius: int = 1,
        include_unannotated: bool = False,
        eros_cache_root: str | Path = "data/cache/dsec_mot_eros",
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self.representation = representation
        self.transform = BenchmarkRepresentation(
            representation, num_bins, EVENT_HEIGHT, EVENT_WIDTH
        )
        self.needs_eros = "eros" in representation_components(representation)
        self.eros_store = ErosSnapshotStore(eros_cache_root) if self.needs_eros else None
        self.dataset = EventDataset(
            root=root,
            split=split,
            sequences=sequences,
            time_window_us=time_window_us,
            feature_stride=feature_stride,
            positive_radius=positive_radius,
            include_unannotated=include_unannotated,
            class_ids=class_ids,
        )
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def sequence_names(self) -> list[str]:
        return self.dataset.sequence_names

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset._samples[index]
        events = self.dataset._read_events(
            sample["sequence"], sample["seq_dir"], sample["timestamp"]
        )
        eros = None
        if self.eros_store is not None:
            eros = self.eros_store.get(
                self.split, sample["sequence"], sample["frame_index"], sample["timestamp"]
            )
        return {
            "events": self.transform(events, eros=eros),
            "label": self.dataset._encode_targets(sample["annotations"]),
            "meta": {
                "sequence": sample["sequence"],
                "timestamp": sample["timestamp"],
                "frame_index": sample["frame_index"],
                "num_annotations": len(sample["annotations"]),
            },
        }


def collate_detection_batch(samples):
    images, targets, metadata = zip(*samples)
    return list(images), list(targets), list(metadata)
