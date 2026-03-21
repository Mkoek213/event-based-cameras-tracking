"""Multi-object tracker using detection output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """Represents a single tracked object."""

    track_id: int
    bbox: np.ndarray           # [x1, y1, x2, y2]
    class_id: int
    score: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0


class EventTracker:
    """Simple IoU-based multi-object tracker (SORT-style).

    Args:
        iou_threshold: Minimum IoU to match a detection with an existing track.
        max_age: Maximum frames a track can survive without a detection match.
        min_hits: Minimum confirmed detections before a track is output.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: np.ndarray, class_ids: np.ndarray, scores: np.ndarray) -> list[Track]:
        """Update tracker with new detections and return active tracks.

        Args:
            detections: ``(N, 4)`` array of bounding boxes ``[x1, y1, x2, y2]``.
            class_ids: ``(N,)`` array of integer class IDs.
            scores: ``(N,)`` confidence scores.

        Returns:
            List of confirmed :class:`Track` objects.
        """
        self._predict()
        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        for trk_idx, det_idx in matched:
            self._tracks[trk_idx].bbox = detections[det_idx]
            self._tracks[trk_idx].time_since_update = 0
            self._tracks[trk_idx].hits += 1
            self._tracks[trk_idx].age += 1

        for det_idx in unmatched_dets:
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox=detections[det_idx],
                    class_id=int(class_ids[det_idx]),
                    score=float(scores[det_idx]),
                )
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        return [t for t in self._tracks if t.hits >= self.min_hits]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict(self) -> None:
        for track in self._tracks:
            track.time_since_update += 1

    def _associate(
        self, detections: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(self._tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self._tracks)))

        iou_matrix = np.zeros((len(self._tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(self._tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._iou(track.bbox, det)

        matched_indices: list[tuple[int, int]] = []
        used_trks: set[int] = set()
        used_dets: set[int] = set()

        for _ in range(min(len(self._tracks), len(detections))):
            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                break
            matched_indices.append((int(t_idx), int(d_idx)))
            used_trks.add(int(t_idx))
            used_dets.add(int(d_idx))
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        unmatched_dets = [d for d in range(len(detections)) if d not in used_dets]
        unmatched_trks = [t for t in range(len(self._tracks)) if t not in used_trks]
        return matched_indices, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)
