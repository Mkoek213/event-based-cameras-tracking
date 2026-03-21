"""Tracking and detection metrics."""

from __future__ import annotations

import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union for two bounding boxes.

    Args:
        box_a: ``[x1, y1, x2, y2]``
        box_b: ``[x1, y1, x2, y2]``

    Returns:
        IoU score in ``[0, 1]``.
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def compute_mota(
    num_gt: int,
    num_misses: int,
    num_false_positives: int,
    num_id_switches: int,
) -> float:
    """Compute Multi-Object Tracking Accuracy (MOTA).

    Args:
        num_gt: Total number of ground-truth object-frame pairs.
        num_misses: Number of missed detections.
        num_false_positives: Number of false positive detections.
        num_id_switches: Number of identity switches.

    Returns:
        MOTA score (can be negative).
    """
    if num_gt == 0:
        return 0.0
    return 1.0 - (num_misses + num_false_positives + num_id_switches) / num_gt
