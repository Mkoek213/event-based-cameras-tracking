"""Dense-grid target encoding shared by event detection datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IDENTITY_IGNORE_INDEX = -1


@dataclass(frozen=True)
class DenseBox:
    """One object box in absolute image coordinates."""

    left: float
    top: float
    width: float
    height: float
    class_id: int
    identity: int = IDENTITY_IGNORE_INDEX


def encode_dense_targets(
    boxes: list[DenseBox],
    image_width: int,
    image_height: int,
    feature_stride: int = 8,
    positive_radius: int = 1,
    class_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode object boxes into class, bbox-distance and positive-cell tensors."""

    cls_targets, bbox_targets, pos_mask, _ = encode_dense_targets_with_identity(
        boxes=boxes,
        image_width=image_width,
        image_height=image_height,
        feature_stride=feature_stride,
        positive_radius=positive_radius,
        class_offset=class_offset,
    )
    return cls_targets, bbox_targets, pos_mask


def encode_dense_targets_with_identity(
    boxes: list[DenseBox],
    image_width: int,
    image_height: int,
    feature_stride: int = 8,
    positive_radius: int = 1,
    class_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode boxes like :func:`encode_dense_targets` plus a per-cell identity map.

    The identity map holds ``box.identity`` at every positive cell assigned to that
    box and ``IDENTITY_IGNORE_INDEX`` elsewhere, so it lines up cell-for-cell with
    ``pos_mask``.
    """

    feat_h = image_height // feature_stride
    feat_w = image_width // feature_stride
    cls_targets = np.zeros((feat_h, feat_w), dtype=np.int64)
    bbox_targets = np.zeros((4, feat_h, feat_w), dtype=np.float32)
    pos_mask = np.zeros((feat_h, feat_w), dtype=bool)
    identity_targets = np.full((feat_h, feat_w), IDENTITY_IGNORE_INDEX, dtype=np.int64)
    assignment_distance = np.full((feat_h, feat_w), np.inf, dtype=np.float32)

    for box in boxes:
        x0 = max(0.0, box.left)
        y0 = max(0.0, box.top)
        x1 = min(float(image_width - 1), box.left + box.width)
        y1 = min(float(image_height - 1), box.top + box.height)
        if x1 <= x0 or y1 <= y0:
            continue

        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)
        center_gx = center_x / feature_stride - 0.5
        center_gy = center_y / feature_stride - 0.5
        gx = int(round(center_gx))
        gy = int(round(center_gy))
        if gx < 0 or gx >= feat_w or gy < 0 or gy >= feat_h:
            continue

        radius = max(int(positive_radius), 0)
        x_start = max(gx - radius, 0)
        x_end = min(gx + radius, feat_w - 1)
        y_start = max(gy - radius, 0)
        y_end = min(gy + radius, feat_h - 1)

        for gy_idx in range(y_start, y_end + 1):
            for gx_idx in range(x_start, x_end + 1):
                dx = gx_idx - center_gx
                dy = gy_idx - center_gy
                distance = float(dx * dx + dy * dy)
                if distance >= assignment_distance[gy_idx, gx_idx]:
                    continue

                cls_targets[gy_idx, gx_idx] = box.class_id + class_offset
                pos_mask[gy_idx, gx_idx] = True
                identity_targets[gy_idx, gx_idx] = box.identity
                assignment_distance[gy_idx, gx_idx] = distance

                cell_cx = (gx_idx + 0.5) * feature_stride
                cell_cy = (gy_idx + 0.5) * feature_stride
                bbox_targets[0, gy_idx, gx_idx] = (cell_cx - x0) / feature_stride
                bbox_targets[1, gy_idx, gx_idx] = (cell_cy - y0) / feature_stride
                bbox_targets[2, gy_idx, gx_idx] = (x1 - cell_cx) / feature_stride
                bbox_targets[3, gy_idx, gx_idx] = (y1 - cell_cy) / feature_stride

    return cls_targets, bbox_targets, pos_mask, identity_targets
