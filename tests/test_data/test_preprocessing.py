"""Unit tests for EventPreprocessor."""

import numpy as np
import pytest

from src.data.preprocessing import EventPreprocessor


def _make_events(n: int = 100, height: int = 32, width: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.bool_)])
    events = np.empty(n, dtype=dtype)
    events["x"] = rng.integers(0, width, size=n).astype(np.uint16)
    events["y"] = rng.integers(0, height, size=n).astype(np.uint16)
    events["t"] = rng.integers(0, 1_000_000, size=n)
    events["p"] = rng.integers(0, 2, size=n).astype(bool)
    return events


class TestEventPreprocessor:
    H, W = 32, 32

    def test_event_frame_shape(self):
        proc = EventPreprocessor(self.H, self.W, representation="event_frame")
        events = _make_events(height=self.H, width=self.W)
        out = proc(events)
        assert out.shape == (2, self.H, self.W)

    def test_event_frame_empty_input(self):
        proc = EventPreprocessor(self.H, self.W, representation="event_frame")
        dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.bool_)])
        out = proc(np.empty(0, dtype=dtype))
        assert out.shape == (2, self.H, self.W)
        assert out.sum() == 0

    def test_time_surface_shape(self):
        proc = EventPreprocessor(self.H, self.W, representation="time_surface")
        events = _make_events(height=self.H, width=self.W)
        out = proc(events)
        assert out.shape == (2, self.H, self.W)
        assert 0.0 <= out.max() <= 1.0

    def test_voxel_grid_shape(self):
        num_bins = 5
        proc = EventPreprocessor(self.H, self.W, representation="voxel_grid", num_bins=num_bins)
        events = _make_events(height=self.H, width=self.W)
        out = proc(events)
        assert out.shape == (num_bins, self.H, self.W)

    def test_invalid_representation(self):
        with pytest.raises(ValueError, match="Unknown representation"):
            EventPreprocessor(self.H, self.W, representation="bad_rep")


class TestEventDataset:
    def test_empty_dataset_length(self, tmp_path):
        from src.data.dataset import EventDataset

        ds = EventDataset(root=tmp_path)
        assert len(ds) == 0

    def test_encode_targets_uses_center_cell_only_when_radius_zero(self, tmp_path):
        from src.data.dataset import Annotation, EventDataset

        ds = EventDataset(root=tmp_path, feature_stride=8, positive_radius=0)
        annotation = Annotation(
            timestamp=0,
            track_id=1,
            left=100.0,
            top=120.0,
            width=80.0,
            height=64.0,
            class_id=2,
        )

        cls_targets, _, pos_mask = ds._encode_targets([annotation])

        assert int(pos_mask.sum()) == 1
        gy, gx = np.argwhere(pos_mask)[0]
        assert cls_targets[gy, gx] == annotation.class_id + ds.class_offset

    def test_encode_targets_respects_explicit_positive_radius(self, tmp_path):
        from src.data.dataset import Annotation, EventDataset

        ds = EventDataset(root=tmp_path, feature_stride=8, positive_radius=1)
        annotation = Annotation(
            timestamp=0,
            track_id=1,
            left=100.0,
            top=120.0,
            width=80.0,
            height=64.0,
            class_id=2,
        )

        _, _, pos_mask = ds._encode_targets([annotation])

        assert int(pos_mask.sum()) == 9
