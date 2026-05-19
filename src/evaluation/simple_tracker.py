"""Simple class-aware IoU tracking-by-detection baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.evaluation.detection_export import load_detection_export
from src.utils.metrics import compute_iou


@dataclass(frozen=True)
class DetectionObservation:
    frame_index: int
    timestamp: int
    class_id: int
    score: float
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float

    @property
    def bbox_xyxy(self) -> list[float]:
        return [
            self.bbox_left,
            self.bbox_top,
            self.bbox_left + self.bbox_width,
            self.bbox_top + self.bbox_height,
        ]


@dataclass
class TrackState:
    track_id: int
    class_id: int
    bbox_xyxy: list[float]
    score: float
    hits: int = 1
    missed_frames: int = 0


@dataclass(frozen=True)
class TrackedRecord:
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


class SimpleIoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_missed_frames: int = 2,
        min_hits: int = 1,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.min_hits = int(min_hits)
        self._tracks: list[TrackState] = []
        self._next_id = 1

    def update(self, detections: list[DetectionObservation]) -> list[TrackedRecord]:
        for track in self._tracks:
            track.missed_frames += 1

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for det_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                iou = compute_iou(track.bbox_xyxy, detection.bbox_xyxy)
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
                outputs.append(
                    TrackedRecord(
                        frame_index=detection.frame_index,
                        timestamp=detection.timestamp,
                        track_id=track.track_id,
                        class_id=track.class_id,
                        score=track.score,
                        bbox_left=detection.bbox_left,
                        bbox_top=detection.bbox_top,
                        bbox_width=detection.bbox_width,
                        bbox_height=detection.bbox_height,
                    )
                )

        for det_index, detection in enumerate(detections):
            if det_index in matched_detections:
                continue
            track = TrackState(
                track_id=self._next_id,
                class_id=detection.class_id,
                bbox_xyxy=detection.bbox_xyxy,
                score=detection.score,
            )
            self._tracks.append(track)
            if track.hits >= self.min_hits:
                outputs.append(
                    TrackedRecord(
                        frame_index=detection.frame_index,
                        timestamp=detection.timestamp,
                        track_id=track.track_id,
                        class_id=track.class_id,
                        score=track.score,
                        bbox_left=detection.bbox_left,
                        bbox_top=detection.bbox_top,
                        bbox_width=detection.bbox_width,
                        bbox_height=detection.bbox_height,
                    )
                )
            self._next_id += 1

        self._tracks = [track for track in self._tracks if track.missed_frames <= self.max_missed_frames]
        outputs.sort(key=lambda item: item.track_id)
        return outputs


def load_detections_by_frame(path: Path) -> list[tuple[int, int, list[DetectionObservation]]]:
    payload = load_detection_export(path)
    grouped: dict[int, list[DetectionObservation]] = {
        int(frame["frame_index"]): [] for frame in payload.get("frames", [])
    }
    timestamps = {int(frame["frame_index"]): int(frame["timestamp"]) for frame in payload.get("frames", [])}
    for row in payload.get("detections", []):
        detection = DetectionObservation(
            frame_index=int(row["frame_index"]),
            timestamp=int(row["timestamp"]),
            class_id=int(row["class_id"]),
            score=float(row["score"]),
            bbox_left=float(row["bbox_left"]),
            bbox_top=float(row["bbox_top"]),
            bbox_width=float(row["bbox_width"]),
            bbox_height=float(row["bbox_height"]),
        )
        grouped.setdefault(detection.frame_index, []).append(detection)
        timestamps[detection.frame_index] = detection.timestamp

    frames: list[tuple[int, int, list[DetectionObservation]]] = []
    for frame_index in sorted(grouped):
        frames.append((frame_index, timestamps[frame_index], grouped[frame_index]))
    return frames


def track_detections(
    detection_export_path: Path,
    output_path: Path,
    iou_threshold: float = 0.5,
    max_missed_frames: int = 2,
    min_hits: int = 1,
) -> dict:
    frames = load_detections_by_frame(detection_export_path)
    tracker = SimpleIoUTracker(
        iou_threshold=iou_threshold,
        max_missed_frames=max_missed_frames,
        min_hits=min_hits,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracked_records: list[TrackedRecord] = []
    for _, _, detections in frames:
        tracked_records.extend(tracker.update(detections))

    output_path.write_text(
        "\n".join(record.to_trackeval_line() for record in tracked_records) + ("\n" if tracked_records else ""),
        encoding="utf-8",
    )

    return {
        "input": str(detection_export_path),
        "output": str(output_path),
        "frames": len(frames),
        "tracks_written": len(tracked_records),
        "iou_threshold": iou_threshold,
        "max_missed_frames": max_missed_frames,
        "min_hits": min_hits,
    }
