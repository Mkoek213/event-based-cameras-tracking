"""Tracking-by-detection backends used by DSEC-MOT benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.detection_export import load_detection_export
from src.utils.metrics import compute_iou

TrackerBackend = Literal[
    "iou",
    "sort",
    "bytetrack",
    "boxmot_bytetrack",
    "boxmot_ocsort",
    "boxmot_botsort",
]


@dataclass(frozen=True)
class DetectionObservation:
    """One detector output assigned to a dataset frame."""

    frame_index: int
    timestamp: int
    class_id: int
    score: float
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    embedding: np.ndarray | None = None

    @property
    def bbox_xyxy(self) -> np.ndarray:
        return np.asarray(
            [
                self.bbox_left,
                self.bbox_top,
                self.bbox_left + self.bbox_width,
                self.bbox_top + self.bbox_height,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class TrackedRecord:
    """One tracker output row in MOTChallenge/TrackEval format."""

    frame_index: int
    timestamp: int
    track_id: int
    class_id: int
    score: float
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float

    def to_trackeval_line(self) -> str:
        frame = self.frame_index + 1
        return (
            f"{frame},{self.track_id},{self.bbox_left:.3f},{self.bbox_top:.3f},"
            f"{self.bbox_width:.3f},{self.bbox_height:.3f},{self.score:.6f},{self.class_id + 1}"
        )


@dataclass(frozen=True)
class TrackingConfig:
    """Common tracker configuration."""

    backend: TrackerBackend = "iou"
    iou_threshold: float = 0.5
    max_missed_frames: int = 2
    min_hits: int = 1
    high_threshold: float = 0.6
    low_threshold: float = 0.1
    with_reid: bool = False
    appearance_thresh: float = 0.25
    proximity_thresh: float = 0.5


@dataclass
class _IoUTrack:
    track_id: int
    class_id: int
    bbox_xyxy: np.ndarray
    score: float
    hits: int = 1
    missed_frames: int = 0


class IoUTracker:
    """Greedy class-aware IoU tracker used as the simplest baseline."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_missed_frames: int = 2,
        min_hits: int = 1,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.min_hits = int(min_hits)
        self._tracks: list[_IoUTrack] = []
        self._next_id = 1

    def update(self, detections: list[DetectionObservation]) -> list[TrackedRecord]:
        for track in self._tracks:
            track.missed_frames += 1

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for det_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                iou = compute_iou(track.bbox_xyxy.tolist(), detection.bbox_xyxy.tolist())
                if iou >= self.iou_threshold:
                    candidates.append((float(iou), track_index, det_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        outputs: list[TrackedRecord] = []
        for _, track_index, det_index in sorted(candidates, reverse=True):
            if track_index in matched_tracks or det_index in matched_detections:
                continue
            track = self._tracks[track_index]
            detection = detections[det_index]
            track.bbox_xyxy = detection.bbox_xyxy
            track.score = detection.score
            track.missed_frames = 0
            track.hits += 1
            matched_tracks.add(track_index)
            matched_detections.add(det_index)
            if track.hits >= self.min_hits:
                outputs.append(_record_from_detection(detection, track.track_id))

        for det_index, detection in enumerate(detections):
            if det_index in matched_detections:
                continue
            track = _IoUTrack(
                track_id=self._next_id,
                class_id=detection.class_id,
                bbox_xyxy=detection.bbox_xyxy,
                score=detection.score,
            )
            self._tracks.append(track)
            if track.hits >= self.min_hits:
                outputs.append(_record_from_detection(detection, track.track_id))
            self._next_id += 1

        self._tracks = [
            track for track in self._tracks if track.missed_frames <= self.max_missed_frames
        ]
        return sorted(outputs, key=lambda item: item.track_id)


class _KalmanBoxTrack:
    """Constant-velocity Kalman track over bbox corners."""

    def __init__(self, detection: DetectionObservation, track_id: int) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.score = detection.score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.last_frame_index = detection.frame_index
        self.last_timestamp = detection.timestamp

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4, 0] = detection.bbox_xyxy
        self.p = np.eye(8, dtype=np.float32) * 10.0
        self.p[4:, 4:] *= 1000.0
        self.f = np.eye(8, dtype=np.float32)
        for index in range(4):
            self.f[index, index + 4] = 1.0
        self.h = np.zeros((4, 8), dtype=np.float32)
        self.h[:4, :4] = np.eye(4, dtype=np.float32)
        self.q = np.eye(8, dtype=np.float32) * 0.01
        self.r = np.eye(4, dtype=np.float32)

    @property
    def bbox_xyxy(self) -> np.ndarray:
        bbox = self.x[:4, 0].copy()
        bbox[2] = max(bbox[2], bbox[0] + 1.0)
        bbox[3] = max(bbox[3], bbox[1] + 1.0)
        return bbox

    def predict(self) -> np.ndarray:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        self.age += 1
        self.time_since_update += 1
        return self.bbox_xyxy

    def update(self, detection: DetectionObservation) -> None:
        z = detection.bbox_xyxy.reshape(4, 1)
        y = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(8, dtype=np.float32) - k @ self.h) @ self.p
        self.score = detection.score
        self.class_id = detection.class_id
        self.hits += 1
        self.time_since_update = 0
        self.last_frame_index = detection.frame_index
        self.last_timestamp = detection.timestamp


class SortTracker:
    """SORT-style tracker with Kalman prediction and Hungarian IoU matching."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.min_hits = int(min_hits)
        self._tracks: list[_KalmanBoxTrack] = []
        self._next_id = 1

    def update(self, detections: list[DetectionObservation]) -> list[TrackedRecord]:
        for track in self._tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_detections = _match_tracks_to_detections(
            tracks=self._tracks,
            detections=detections,
            iou_threshold=self.iou_threshold,
        )

        for track_index, detection_index in matches:
            self._tracks[track_index].update(detections[detection_index])

        for detection_index in unmatched_detections:
            track = _KalmanBoxTrack(detections[detection_index], self._next_id)
            self._tracks.append(track)
            self._next_id += 1

        outputs = [
            _record_from_track(track)
            for track in self._tracks
            if track.time_since_update == 0 and track.hits >= self.min_hits
        ]
        self._tracks = [
            track for track in self._tracks if track.time_since_update <= self.max_missed_frames
        ]
        return sorted(outputs, key=lambda item: item.track_id)


class ByteTrackStyleTracker:
    """ByteTrack-style two-stage association for high and low score boxes.

    This is an in-repo implementation of the key ByteTrack idea: keep
    low-confidence detections for matching existing tracks, but initialise new
    tracks only from high-confidence detections.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 30,
        min_hits: int = 1,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.min_hits = int(min_hits)
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self._tracks: list[_KalmanBoxTrack] = []
        self._next_id = 1

    def update(self, detections: list[DetectionObservation]) -> list[TrackedRecord]:
        for track in self._tracks:
            track.predict()

        high = [detection for detection in detections if detection.score >= self.high_threshold]
        low = [
            detection
            for detection in detections
            if self.low_threshold <= detection.score < self.high_threshold
        ]

        first_matches, unmatched_tracks, unmatched_high = _match_tracks_to_detections(
            tracks=self._tracks,
            detections=high,
            iou_threshold=self.iou_threshold,
        )
        for track_index, detection_index in first_matches:
            self._tracks[track_index].update(high[detection_index])

        second_tracks = [self._tracks[index] for index in unmatched_tracks]
        second_matches, _, _ = _match_tracks_to_detections(
            tracks=second_tracks,
            detections=low,
            iou_threshold=self.iou_threshold,
        )
        matched_second_track_indices: set[int] = set()
        for local_track_index, detection_index in second_matches:
            global_track_index = unmatched_tracks[local_track_index]
            self._tracks[global_track_index].update(low[detection_index])
            matched_second_track_indices.add(global_track_index)

        for detection_index in unmatched_high:
            track = _KalmanBoxTrack(high[detection_index], self._next_id)
            self._tracks.append(track)
            self._next_id += 1

        outputs = [
            _record_from_track(track)
            for track in self._tracks
            if track.time_since_update == 0 and track.hits >= self.min_hits
        ]
        self._tracks = [
            track for track in self._tracks if track.time_since_update <= self.max_missed_frames
        ]
        return sorted(outputs, key=lambda item: item.track_id)


class BoxMotTracker:
    """Adapter for external BoxMOT tracker implementations."""

    def __init__(self, backend: TrackerBackend, config: TrackingConfig) -> None:
        self.backend = backend
        self._tracker = _build_boxmot_tracker(backend, config)
        self._dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def update(
        self,
        detections: list[DetectionObservation],
        frame_index: int,
        timestamp: int,
        embeddings: np.ndarray | None = None,
    ) -> list[TrackedRecord]:
        if detections:
            dets = np.asarray(
                [
                    [
                        detection.bbox_left,
                        detection.bbox_top,
                        detection.bbox_left + detection.bbox_width,
                        detection.bbox_top + detection.bbox_height,
                        detection.score,
                        detection.class_id,
                    ]
                    for detection in detections
                ],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        if embeddings is None:
            results = self._tracker.update(dets, self._dummy_image)
        else:
            if embeddings.shape[0] != dets.shape[0]:
                raise ValueError(
                    f"Embedding rows ({embeddings.shape[0]}) do not match "
                    f"detections ({dets.shape[0]}) at frame {frame_index}."
                )
            results = self._tracker.update(dets, self._dummy_image, embeddings)
        records: list[TrackedRecord] = []
        for row in np.asarray(results, dtype=np.float32):
            x1, y1, x2, y2, track_id, score, class_id, _ = row[:8]
            records.append(
                TrackedRecord(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    track_id=int(track_id),
                    class_id=int(class_id),
                    score=float(score),
                    bbox_left=float(x1),
                    bbox_top=float(y1),
                    bbox_width=float(x2 - x1),
                    bbox_height=float(y2 - y1),
                )
            )
        return sorted(records, key=lambda item: item.track_id)


def load_detections_by_frame(path: Path) -> list[tuple[int, int, list[DetectionObservation]]]:
    """Load detector JSON and group detections by frame."""

    payload = load_detection_export(path)
    grouped: dict[int, list[DetectionObservation]] = {
        int(frame["frame_index"]): [] for frame in payload.get("frames", [])
    }
    timestamps = {
        int(frame["frame_index"]): int(frame["timestamp"]) for frame in payload.get("frames", [])
    }
    for row in payload.get("detections", []):
        embedding = row.get("embedding")
        detection = DetectionObservation(
            frame_index=int(row["frame_index"]),
            timestamp=int(row["timestamp"]),
            class_id=int(row["class_id"]),
            score=float(row["score"]),
            bbox_left=float(row["bbox_left"]),
            bbox_top=float(row["bbox_top"]),
            bbox_width=float(row["bbox_width"]),
            bbox_height=float(row["bbox_height"]),
            embedding=np.asarray(embedding, dtype=np.float32) if embedding is not None else None,
        )
        grouped.setdefault(detection.frame_index, []).append(detection)
        timestamps[detection.frame_index] = detection.timestamp

    return [
        (frame_index, timestamps[frame_index], grouped[frame_index])
        for frame_index in sorted(grouped)
    ]


def track_detections(
    detection_export_path: Path,
    output_path: Path,
    config: TrackingConfig | None = None,
) -> dict:
    """Track exported detections and write TrackEval-compatible MOT rows."""

    config = config or TrackingConfig()
    frames = load_detections_by_frame(detection_export_path)
    tracker = build_tracker(config)

    embedding_dim = 0
    if config.with_reid:
        embedding_dim = _find_embedding_dim(frames)
        if embedding_dim == 0:
            raise ValueError(
                f"with_reid tracking requires embeddings in {detection_export_path}, "
                "but no detection carries one. Export with an embedding-head checkpoint."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracked_records: list[TrackedRecord] = []
    for frame_index, timestamp, detections in frames:
        if isinstance(tracker, BoxMotTracker):
            embeddings = _stack_frame_embeddings(detections, embedding_dim, config.with_reid)
            tracked_records.extend(tracker.update(detections, frame_index, timestamp, embeddings))
        else:
            tracked_records.extend(tracker.update(detections))

    output_path.write_text(
        "\n".join(record.to_trackeval_line() for record in tracked_records)
        + ("\n" if tracked_records else ""),
        encoding="utf-8",
    )

    return {
        "input": str(detection_export_path),
        "output": str(output_path),
        "frames": len(frames),
        "tracks_written": len(tracked_records),
        "backend": config.backend,
        "iou_threshold": config.iou_threshold,
        "max_missed_frames": config.max_missed_frames,
        "min_hits": config.min_hits,
        "high_threshold": config.high_threshold,
        "low_threshold": config.low_threshold,
        "with_reid": config.with_reid,
        "appearance_thresh": config.appearance_thresh,
        "proximity_thresh": config.proximity_thresh,
        "embedding_dim": embedding_dim,
    }


def _find_embedding_dim(frames: list[tuple[int, int, list[DetectionObservation]]]) -> int:
    for _, _, detections in frames:
        for detection in detections:
            if detection.embedding is not None:
                return int(detection.embedding.shape[0])
    return 0


def _stack_frame_embeddings(
    detections: list[DetectionObservation],
    embedding_dim: int,
    with_reid: bool,
) -> np.ndarray | None:
    """Stack per-detection embeddings aligned 1:1 with the dets row order.

    BoT-SORT with ``with_reid=True`` needs an embedding array on every update,
    including empty frames, so this returns a ``(0, D)`` array rather than None
    whenever reid is active.
    """

    if not with_reid:
        return None
    if not detections:
        return np.empty((0, embedding_dim), dtype=np.float32)
    rows: list[np.ndarray] = []
    for detection in detections:
        if detection.embedding is None:
            raise ValueError(
                f"Detection at frame {detection.frame_index} has no embedding; "
                "cannot run with_reid tracking on a partial export."
            )
        rows.append(np.asarray(detection.embedding, dtype=np.float32))
    return np.stack(rows)


def build_tracker(config: TrackingConfig):
    """Create a tracker backend from config."""

    if config.backend == "iou":
        return IoUTracker(
            iou_threshold=config.iou_threshold,
            max_missed_frames=config.max_missed_frames,
            min_hits=config.min_hits,
        )
    if config.backend == "sort":
        return SortTracker(
            iou_threshold=config.iou_threshold,
            max_missed_frames=config.max_missed_frames,
            min_hits=config.min_hits,
        )
    if config.backend == "bytetrack":
        return ByteTrackStyleTracker(
            iou_threshold=config.iou_threshold,
            max_missed_frames=config.max_missed_frames,
            min_hits=config.min_hits,
            high_threshold=config.high_threshold,
            low_threshold=config.low_threshold,
        )
    if config.backend.startswith("boxmot_"):
        return BoxMotTracker(config.backend, config)
    raise ValueError(f"Unsupported tracker backend: {config.backend}")


def _build_boxmot_tracker(backend: TrackerBackend, config: TrackingConfig):
    try:
        from boxmot.trackers import BotSort, ByteTrack, OcSort
    except ImportError as exc:
        raise ImportError(
            "BoxMOT tracker backend requested, but boxmot is not installed. "
            "Install it with: .venv/bin/python -m pip install boxmot==19.0.0"
        ) from exc

    frame_rate = 20
    if backend == "boxmot_bytetrack":
        return ByteTrack(
            min_conf=config.low_threshold,
            track_thresh=config.high_threshold,
            match_thresh=config.iou_threshold,
            track_buffer=config.max_missed_frames,
            frame_rate=frame_rate,
        )
    if backend == "boxmot_ocsort":
        return OcSort(
            min_conf=config.low_threshold,
            use_byte=True,
            delta_t=3,
        )
    if backend == "boxmot_botsort":
        if config.with_reid:
            # reid_model=None + external embeddings via update(..., embs=...):
            # BoT-SORT then never instantiates its own ReID model.
            return BotSort(
                reid_model=None,
                track_high_thresh=config.high_threshold,
                track_low_thresh=config.low_threshold,
                new_track_thresh=config.high_threshold,
                track_buffer=config.max_missed_frames,
                match_thresh=config.iou_threshold,
                proximity_thresh=config.proximity_thresh,
                appearance_thresh=config.appearance_thresh,
                frame_rate=frame_rate,
                with_reid=True,
            )
        return BotSort(
            track_high_thresh=config.high_threshold,
            track_low_thresh=config.low_threshold,
            new_track_thresh=config.high_threshold,
            track_buffer=config.max_missed_frames,
            match_thresh=config.iou_threshold,
            frame_rate=frame_rate,
            with_reid=False,
        )
    raise ValueError(f"Unsupported BoxMOT backend: {backend}")


def _record_from_detection(detection: DetectionObservation, track_id: int) -> TrackedRecord:
    return TrackedRecord(
        frame_index=detection.frame_index,
        timestamp=detection.timestamp,
        track_id=track_id,
        class_id=detection.class_id,
        score=detection.score,
        bbox_left=detection.bbox_left,
        bbox_top=detection.bbox_top,
        bbox_width=detection.bbox_width,
        bbox_height=detection.bbox_height,
    )


def _record_from_track(track: _KalmanBoxTrack) -> TrackedRecord:
    x1, y1, x2, y2 = track.bbox_xyxy
    return TrackedRecord(
        frame_index=track.last_frame_index,
        timestamp=track.last_timestamp,
        track_id=track.track_id,
        class_id=track.class_id,
        score=track.score,
        bbox_left=float(x1),
        bbox_top=float(y1),
        bbox_width=float(x2 - x1),
        bbox_height=float(y2 - y1),
    )


def _match_tracks_to_detections(
    tracks: list[_KalmanBoxTrack],
    detections: list[DetectionObservation],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks:
        return [], [], list(range(len(detections)))
    if not detections:
        return [], list(range(len(tracks))), []

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            if track.class_id != detection.class_id:
                continue
            iou_matrix[track_index, detection_index] = compute_iou(
                track.bbox_xyxy.tolist(), detection.bbox_xyxy.tolist()
            )

    row_indices, col_indices = linear_sum_assignment(1.0 - iou_matrix)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] < iou_threshold:
            continue
        matches.append((int(row), int(col)))
        matched_tracks.add(int(row))
        matched_detections.add(int(col))

    unmatched_tracks = [index for index in range(len(tracks)) if index not in matched_tracks]
    unmatched_detections = [
        index for index in range(len(detections)) if index not in matched_detections
    ]
    return matches, unmatched_tracks, unmatched_detections
