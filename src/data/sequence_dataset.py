"""Ordered clip dataset for recurrent embedding training on DSEC-MOT."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH, EventDataset
from src.data.dense_targets import (
    IDENTITY_IGNORE_INDEX,
    DenseBox,
    encode_dense_targets_with_identity,
)
from src.data.pretrained_detection_dataset import ErosSnapshotStore
from src.data.representations import BenchmarkRepresentation, representation_components


class IdentityVocabulary:
    """Contiguous identity classes over ``(sequence, track_id)`` pairs.

    Unknown pairs (e.g. validation-only tracks) map to ``IDENTITY_IGNORE_INDEX``
    so they never contribute to the identity loss.
    """

    def __init__(self, mapping: dict[tuple[str, int], int]) -> None:
        self._mapping = dict(mapping)

    @property
    def num_identities(self) -> int:
        return len(self._mapping)

    def lookup(self, sequence: str, track_id: int) -> int:
        return self._mapping.get((sequence, int(track_id)), IDENTITY_IGNORE_INDEX)

    @classmethod
    def from_samples(cls, samples: list[dict]) -> IdentityVocabulary:
        keys = sorted(
            {
                (sample["sequence"], int(annotation.track_id))
                for sample in samples
                for annotation in sample["annotations"]
            }
        )
        return cls({key: index for index, key in enumerate(keys)})


class DSECClipDataset(Dataset):
    """Yield clips of consecutive annotated frames from single DSEC-MOT sequences.

    Each item covers ``clip_length`` frames of one sequence and provides, per
    frame, the dense event representation, detection targets and an identity map
    aligned with ``pos_mask`` (see ``encode_dense_targets_with_identity``).
    Clips never cross sequence boundaries; tails shorter than ``clip_length``
    are dropped.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        sequences: Sequence[str] | None,
        representation: str,
        num_bins: int = 5,
        time_window_us: int = 50_000,
        clip_length: int = 8,
        clip_stride: int | None = None,
        feature_stride: int = 8,
        positive_radius: int = 1,
        eros_cache_root: str | Path = "data/cache/dsec_mot_eros",
        class_ids: Sequence[int] | None = None,
        identity_vocabulary: IdentityVocabulary | None = None,
    ) -> None:
        if clip_length <= 0:
            raise ValueError("clip_length must be positive.")
        if clip_stride is not None and clip_stride <= 0:
            raise ValueError("clip_stride must be positive.")
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride) if clip_stride is not None else self.clip_length
        self.feature_stride = int(feature_stride)
        self.positive_radius = int(positive_radius)
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
            class_ids=class_ids,
        )
        self.split = split
        self.identity_vocabulary = identity_vocabulary or IdentityVocabulary.from_samples(
            self.dataset._samples
        )
        self._clips = self._build_clips()

    def _build_clips(self) -> list[list[int]]:
        by_sequence: dict[str, list[int]] = {}
        for index, sample in enumerate(self.dataset._samples):
            by_sequence.setdefault(sample["sequence"], []).append(index)

        clips: list[list[int]] = []
        for sequence in sorted(by_sequence):
            indices = by_sequence[sequence]
            for start in range(0, len(indices) - self.clip_length + 1, self.clip_stride):
                clips.append(indices[start : start + self.clip_length])
        return clips

    def __len__(self) -> int:
        return len(self._clips)

    @property
    def num_identities(self) -> int:
        return self.identity_vocabulary.num_identities

    @property
    def sequence_names(self) -> list[str]:
        return self.dataset.sequence_names

    def _frame(self, index: int) -> dict:
        sample = self.dataset._samples[index]
        events = self.dataset._read_events(
            sample["sequence"], sample["seq_dir"], sample["timestamp"]
        )
        eros = None
        if self.eros_store is not None:
            eros = self.eros_store.get(
                self.split, sample["sequence"], sample["frame_index"], sample["timestamp"]
            )
        boxes = [
            DenseBox(
                left=annotation.left,
                top=annotation.top,
                width=annotation.width,
                height=annotation.height,
                class_id=annotation.class_id,
                identity=self.identity_vocabulary.lookup(sample["sequence"], annotation.track_id),
            )
            for annotation in sample["annotations"]
        ]
        cls_targets, bbox_targets, pos_mask, identity_targets = encode_dense_targets_with_identity(
            boxes=boxes,
            image_width=EVENT_WIDTH,
            image_height=EVENT_HEIGHT,
            feature_stride=self.feature_stride,
            positive_radius=self.positive_radius,
        )
        return {
            "events": self.transform(events, eros=eros),
            "cls": cls_targets,
            "bbox": bbox_targets,
            "pos_mask": pos_mask,
            "identity": identity_targets,
            "meta": {
                "sequence": sample["sequence"],
                "timestamp": sample["timestamp"],
                "frame_index": sample["frame_index"],
                "num_annotations": len(sample["annotations"]),
            },
        }

    def __getitem__(self, index: int) -> dict:
        frames = [self._frame(sample_index) for sample_index in self._clips[index]]
        return {
            "events": np.stack([frame["events"] for frame in frames]),
            "cls": np.stack([frame["cls"] for frame in frames]),
            "bbox": np.stack([frame["bbox"] for frame in frames]),
            "pos_mask": np.stack([frame["pos_mask"] for frame in frames]),
            "identity": np.stack([frame["identity"] for frame in frames]),
            "meta": [frame["meta"] for frame in frames],
        }


def collate_clip_batch(samples: list[dict]) -> dict[str, object]:
    """Stack clip items to ``(B, T, ...)`` tensors plus a per-clip meta list."""

    return {
        "events": torch.stack([torch.from_numpy(sample["events"]).float() for sample in samples]),
        "cls_targets": torch.stack([torch.from_numpy(sample["cls"]).long() for sample in samples]),
        "bbox_targets": torch.stack(
            [torch.from_numpy(sample["bbox"]).float() for sample in samples]
        ),
        "pos_mask": torch.stack(
            [torch.from_numpy(sample["pos_mask"]).bool() for sample in samples]
        ),
        "identity_targets": torch.stack(
            [torch.from_numpy(sample["identity"]).long() for sample in samples]
        ),
        "meta": [sample["meta"] for sample in samples],
    }
