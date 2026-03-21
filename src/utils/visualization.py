"""Visualization helpers for event frames and tracking results."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def draw_tracks(
    frame: np.ndarray,
    bboxes: np.ndarray,
    track_ids: Sequence[int],
    class_labels: Optional[Sequence[str]] = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding boxes and track IDs onto a BGR image.

    Args:
        frame: ``(H, W, 3)`` uint8 BGR image.
        bboxes: ``(N, 4)`` array of boxes ``[x1, y1, x2, y2]`` in pixel coords.
        track_ids: Sequence of integer track IDs corresponding to *bboxes*.
        class_labels: Optional sequence of string class labels.
        color: BGR colour for all boxes.

    Returns:
        Annotated copy of *frame*.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for visualisation. pip install opencv-python")

    out = frame.copy()
    for i, (box, tid) in enumerate(zip(bboxes, track_ids)):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid}"
        if class_labels is not None and i < len(class_labels):
            label = f"{class_labels[i]} {label}"
        cv2.putText(out, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def events_to_rgb(events: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert an event array to a colour-coded RGB image for display.

    Positive events are rendered in blue, negative events in red.

    Args:
        events: Structured array with fields ``(x, y, t, p)``.
        height: Sensor height in pixels.
        width: Sensor width in pixels.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    if events.size == 0:
        return img
    x = events["x"].astype(int)
    y = events["y"].astype(int)
    p = events["p"].astype(bool)
    mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    img[y[mask & p], x[mask & p]] = [0, 0, 255]    # positive → blue
    img[y[mask & ~p], x[mask & ~p]] = [255, 0, 0]  # negative → red
    return img
