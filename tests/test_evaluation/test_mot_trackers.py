import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.mot_trackers import (
    ByteTrackStyleTracker,
    DetectionObservation,
    IoUTracker,
    TrackingConfig,
    load_detections_by_frame,
    track_detections,
)


def detection(
    frame_index: int,
    score: float = 0.9,
    left: float = 10.0,
    class_id: int = 0,
) -> DetectionObservation:
    return DetectionObservation(
        frame_index=frame_index,
        timestamp=frame_index * 50_000,
        class_id=class_id,
        score=score,
        bbox_left=left,
        bbox_top=20.0,
        bbox_width=30.0,
        bbox_height=40.0,
    )


def test_iou_tracker_keeps_id_for_overlapping_detections() -> None:
    tracker = IoUTracker(iou_threshold=0.3)

    first = tracker.update([detection(0)])
    second = tracker.update([detection(1, left=12.0)])

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_bytetrack_style_does_not_start_track_from_low_score_detection() -> None:
    tracker = ByteTrackStyleTracker(high_threshold=0.6, low_threshold=0.1)

    assert tracker.update([detection(0, score=0.3)]) == []
    assert tracker.update([detection(1, score=0.8)])


def test_track_detections_writes_mot_rows(tmp_path: Path) -> None:
    detection_export = tmp_path / "detections.json"
    detection_export.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_index": 0, "timestamp": 0},
                    {"frame_index": 1, "timestamp": 50_000},
                ],
                "detections": [
                    {
                        "frame_index": 0,
                        "timestamp": 0,
                        "class_id": 0,
                        "score": 0.9,
                        "bbox_left": 10,
                        "bbox_top": 20,
                        "bbox_width": 30,
                        "bbox_height": 40,
                    },
                    {
                        "frame_index": 1,
                        "timestamp": 50_000,
                        "class_id": 0,
                        "score": 0.8,
                        "bbox_left": 12,
                        "bbox_top": 20,
                        "bbox_width": 30,
                        "bbox_height": 40,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "tracks.txt"

    summary = track_detections(detection_export, output, TrackingConfig(backend="iou"))

    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert summary["tracks_written"] == 2
    assert rows[0].startswith("1,1,")
    assert rows[1].startswith("2,1,")


def _write_embedding_export(path: Path, embeddings: dict[int, list[float]]) -> None:
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_index": 0, "timestamp": 0},
                    {"frame_index": 1, "timestamp": 50_000},
                ],
                "detections": [
                    {
                        "frame_index": 0,
                        "timestamp": 0,
                        "class_id": 0,
                        "score": 0.9,
                        "bbox_left": 100,
                        "bbox_top": 100,
                        "bbox_width": 60,
                        "bbox_height": 80,
                        "embedding": embeddings[0],
                    },
                    {
                        "frame_index": 1,
                        "timestamp": 50_000,
                        "class_id": 0,
                        "score": 0.85,
                        "bbox_left": 104,
                        "bbox_top": 102,
                        "bbox_width": 60,
                        "bbox_height": 80,
                        "embedding": embeddings[1],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_embedding_round_trips_through_export_json(tmp_path: Path) -> None:
    detection_export = tmp_path / "detections.json"
    vector = (np.arange(8, dtype=np.float32) / 8.0).tolist()
    _write_embedding_export(detection_export, {0: vector, 1: vector})

    frames = load_detections_by_frame(detection_export)

    observation = frames[0][2][0]
    assert observation.embedding is not None
    assert observation.embedding.dtype == np.float32
    assert observation.embedding.tolist() == pytest.approx(vector)


def test_botsort_with_reid_tracks_from_external_embeddings(tmp_path: Path) -> None:
    detection_export = tmp_path / "detections.json"
    rng = np.random.default_rng(0)
    vector = rng.normal(size=16).astype(np.float32)
    vector /= np.linalg.norm(vector)
    _write_embedding_export(detection_export, {0: vector.tolist(), 1: vector.tolist()})
    output = tmp_path / "tracks.txt"

    summary = track_detections(
        detection_export,
        output,
        TrackingConfig(backend="boxmot_botsort", high_threshold=0.5, with_reid=True),
    )

    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert summary["tracks_written"] == 2
    assert summary["with_reid"] is True
    assert summary["embedding_dim"] == 16
    track_ids = {row.split(",")[1] for row in rows}
    assert len(track_ids) == 1


def test_with_reid_requires_embeddings_in_export(tmp_path: Path) -> None:
    detection_export = tmp_path / "detections.json"
    detection_export.write_text(
        json.dumps(
            {
                "frames": [{"frame_index": 0, "timestamp": 0}],
                "detections": [
                    {
                        "frame_index": 0,
                        "timestamp": 0,
                        "class_id": 0,
                        "score": 0.9,
                        "bbox_left": 10,
                        "bbox_top": 20,
                        "bbox_width": 30,
                        "bbox_height": 40,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires embeddings"):
        track_detections(
            detection_export,
            tmp_path / "tracks.txt",
            TrackingConfig(backend="boxmot_botsort", with_reid=True),
        )
