"""Unit tests for utility functions."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.metrics import compute_iou, compute_mota


class TestComputeIou:
    def test_perfect_overlap(self):
        box = np.array([0, 0, 10, 10], dtype=float)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0, 0, 5, 5], dtype=float)
        b = np.array([10, 10, 20, 20], dtype=float)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        b = np.array([5, 5, 15, 15], dtype=float)
        # intersection 5×5=25, union=100+100-25=175
        assert compute_iou(a, b) == pytest.approx(25 / 175)


class TestComputeMota:
    def test_perfect_tracking(self):
        assert compute_mota(100, 0, 0, 0) == pytest.approx(1.0)

    def test_zero_gt(self):
        assert compute_mota(0, 0, 0, 0) == pytest.approx(0.0)

    def test_all_misses(self):
        assert compute_mota(100, 100, 0, 0) == pytest.approx(0.0)


class TestIo:
    def test_save_and_reload(self, tmp_path):
        from src.utils.io import save_results

        data = {"mota": 0.85, "idf1": 0.90}
        out = tmp_path / "results.json"
        save_results(data, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["mota"] == pytest.approx(0.85)
