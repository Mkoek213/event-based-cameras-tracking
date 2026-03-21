"""Evaluation pipeline for detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.utils.metrics import compute_iou, compute_mota


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    mota: float = 0.0
    idf1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    num_gt: int = 0
    num_fp: int = 0
    num_fn: int = 0
    num_id_switches: int = 0
    per_class: dict = field(default_factory=dict)


class Evaluator:
    """Accumulates per-frame predictions and computes tracking metrics.

    Usage::

        evaluator = Evaluator(iou_threshold=0.5)
        for frame_gt, frame_pred, frame_ids in sequence:
            evaluator.update(frame_gt, frame_pred, frame_ids)
        result = evaluator.compute()
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._num_gt = 0
        self._num_fp = 0
        self._num_fn = 0
        self._num_id_switches = 0
        self._prev_gt_to_id: dict[int, int] = {}

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._num_gt = 0
        self._num_fp = 0
        self._num_fn = 0
        self._num_id_switches = 0
        self._prev_gt_to_id = {}

    def update(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        pred_ids: Sequence[int],
    ) -> None:
        """Accumulate statistics for one frame.

        Args:
            gt_boxes: ``(M, 4)`` ground-truth boxes ``[x1, y1, x2, y2]``.
            pred_boxes: ``(N, 4)`` predicted boxes.
            pred_ids: Track IDs corresponding to *pred_boxes*.
        """
        self._num_gt += len(gt_boxes)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        for gt_idx, gt_box in enumerate(gt_boxes):
            best_iou, best_pred_idx = 0.0, -1
            for pred_idx, pred_box in enumerate(pred_boxes):
                if pred_idx in matched_pred:
                    continue
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_pred_idx = iou, pred_idx
            if best_iou >= self.iou_threshold:
                matched_gt.add(gt_idx)
                matched_pred.add(best_pred_idx)
                new_id = pred_ids[best_pred_idx]
                if gt_idx in self._prev_gt_to_id and self._prev_gt_to_id[gt_idx] != new_id:
                    self._num_id_switches += 1
                self._prev_gt_to_id[gt_idx] = new_id

        self._num_fn += len(gt_boxes) - len(matched_gt)
        self._num_fp += len(pred_boxes) - len(matched_pred)

    def compute(self) -> EvalResult:
        """Return an :class:`EvalResult` with accumulated metrics."""
        mota = compute_mota(self._num_gt, self._num_fn, self._num_fp, self._num_id_switches)
        precision = (
            (self._num_gt - self._num_fn) / max(self._num_gt - self._num_fn + self._num_fp, 1)
        )
        recall = (self._num_gt - self._num_fn) / max(self._num_gt, 1)
        idf1 = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )
        return EvalResult(
            mota=mota,
            idf1=idf1,
            precision=precision,
            recall=recall,
            num_gt=self._num_gt,
            num_fp=self._num_fp,
            num_fn=self._num_fn,
            num_id_switches=self._num_id_switches,
        )
