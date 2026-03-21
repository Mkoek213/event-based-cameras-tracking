"""Unit tests for model sub-package."""

import pytest
import torch

from src.models.backbone import EventBackbone
from src.models.detector import EventDetector
from src.models.tracker import EventTracker


class TestEventBackbone:
    def test_output_shape(self):
        backbone = EventBackbone(in_channels=2, base_channels=16)
        x = torch.zeros(2, 2, 64, 64)
        out = backbone(x)
        # 3 × stride-2 layers → spatial size /8
        assert out.shape == (2, 64, 8, 8)

    def test_out_channels_attribute(self):
        backbone = EventBackbone(in_channels=2, base_channels=32)
        assert backbone.out_channels == 128


class TestEventDetector:
    def test_forward(self):
        backbone = EventBackbone(in_channels=2, base_channels=8)
        model = EventDetector(backbone=backbone, num_classes=5)
        x = torch.zeros(1, 2, 32, 32)
        cls_logits, bbox_offsets = model(x)
        assert cls_logits.shape[1] == 5
        assert bbox_offsets.shape[1] == 4


class TestEventTracker:
    def _make_boxes(self, n: int):
        import numpy as np

        boxes = []
        for i in range(n):
            x1, y1 = float(i * 20), float(i * 20)
            boxes.append([x1, y1, x1 + 10, y1 + 10])
        return np.array(boxes, dtype=float)

    def test_new_tracks_created(self):
        import numpy as np

        tracker = EventTracker(min_hits=1)
        dets = self._make_boxes(3)
        ids = np.zeros(3, dtype=int)
        scores = np.ones(3, dtype=float)
        tracks = tracker.update(dets, ids, scores)
        assert len(tracks) == 3

    def test_empty_detections(self):
        import numpy as np

        tracker = EventTracker(min_hits=1)
        dets = np.empty((0, 4), dtype=float)
        ids = np.empty(0, dtype=int)
        scores = np.empty(0, dtype=float)
        tracks = tracker.update(dets, ids, scores)
        assert tracks == []
