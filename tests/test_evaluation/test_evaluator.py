"""Unit tests for evaluation pipeline."""

import numpy as np
import pytest

from src.evaluation.evaluator import Evaluator


class TestEvaluator:
    def _box(self, x1, y1, x2, y2):
        return np.array([[x1, y1, x2, y2]], dtype=float)

    def test_perfect_single_frame(self):
        ev = Evaluator(iou_threshold=0.5)
        gt = self._box(0, 0, 10, 10)
        pred = self._box(0, 0, 10, 10)
        ev.update(gt, pred, [1])
        result = ev.compute()
        assert result.mota == pytest.approx(1.0)
        assert result.num_fp == 0
        assert result.num_fn == 0

    def test_all_misses(self):
        ev = Evaluator(iou_threshold=0.5)
        gt = self._box(0, 0, 10, 10)
        ev.update(gt, np.empty((0, 4)), [])
        result = ev.compute()
        assert result.recall == pytest.approx(0.0)
        assert result.num_fn == 1

    def test_false_positive(self):
        ev = Evaluator(iou_threshold=0.5)
        gt = np.empty((0, 4))
        pred = self._box(0, 0, 10, 10)
        ev.update(gt, pred, [1])
        result = ev.compute()
        assert result.num_fp == 1

    def test_reset_clears_state(self):
        ev = Evaluator()
        gt = self._box(0, 0, 10, 10)
        pred = self._box(0, 0, 10, 10)
        ev.update(gt, pred, [1])
        ev.reset()
        result = ev.compute()
        assert result.num_gt == 0
