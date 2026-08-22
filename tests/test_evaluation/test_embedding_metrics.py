"""Tests for tracker-independent association-embedding diagnostics."""

import pytest
import torch

from src.evaluation.detection_export import Annotation, DetectionRecord
from src.evaluation.embedding_metrics import (
    match_detections_to_annotations,
    summarise_embeddings,
)
from src.evaluation.embedding_metrics_cli import matched_prediction_collection


def detection(left: float, class_id: int, score: float) -> DetectionRecord:
    return DetectionRecord(
        frame_index=0,
        timestamp=100,
        class_id=class_id,
        score=score,
        bbox_left=left,
        bbox_top=0.0,
        bbox_width=10.0,
        bbox_height=10.0,
        embedding=(1.0, 0.0),
    )


def annotation(left: float, class_id: int, track_id: int) -> Annotation:
    return Annotation(
        timestamp=100,
        track_id=track_id,
        left=left,
        top=0.0,
        width=10.0,
        height=10.0,
        class_id=class_id,
    )


def test_embedding_summary_reports_retrieval_separation_and_temporal_gaps() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [-1.0, 0.0],
            [-0.99, 0.01],
        ]
    )
    classes = torch.zeros(5, dtype=torch.long)
    tracks = torch.tensor([1, 1, 1, 2, 2])
    frames = torch.tensor([0, 1, 5, 0, 2])

    summary = summarise_embeddings(
        embeddings,
        classes,
        ["sequence"] * 5,
        tracks,
        frames,
        max_pairs_per_group=100,
    )

    assert summary["valid_queries"] == 5
    assert summary["retrieval_map"] == pytest.approx(1.0)
    assert summary["retrieval_rank1"] == pytest.approx(1.0)
    assert summary["mean_cosine_margin"] > 1.9
    assert summary["same_identity_cosine"]["count"] == 4
    assert summary["different_identity_same_class_cosine"]["count"] == 6
    assert summary["temporal_positive_cosine"]["1"]["count"] == 1
    assert summary["temporal_positive_cosine"]["2-4"]["count"] == 2
    assert summary["temporal_positive_cosine"]["5-8"]["count"] == 1


def test_matching_is_one_to_one_score_ordered_and_class_aware() -> None:
    detections = [
        detection(0.0, class_id=0, score=0.8),
        detection(1.0, class_id=0, score=0.9),
        detection(20.0, class_id=1, score=0.7),
    ]
    annotations = [
        annotation(0.0, class_id=0, track_id=10),
        annotation(20.0, class_id=0, track_id=20),
        annotation(20.0, class_id=1, track_id=30),
    ]

    matches = match_detections_to_annotations(detections, annotations, iou_threshold=0.5)

    assert {
        (detection_index, annotation_index) for detection_index, annotation_index, _ in matches
    } == {
        (1, 0),
        (2, 2),
    }
    assert all(iou >= 0.5 for _, _, iou in matches)


def test_empty_embedding_collection_has_explicit_zero_metrics() -> None:
    summary = summarise_embeddings(
        torch.empty((0, 8)),
        torch.empty(0, dtype=torch.long),
        [],
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )

    assert summary["embedding_count"] == 0
    assert summary["valid_queries"] == 0
    assert summary["same_identity_cosine"]["count"] == 0


def test_prediction_coverage_counts_only_frames_present_in_payload() -> None:
    prediction = detection(0.0, class_id=0, score=0.9)
    payload = {
        "embedding_dim": 2,
        "frames": [{"frame_index": 0, "timestamp": 100}],
        "detections": [prediction.to_dict()],
    }
    annotations = [
        annotation(0.0, class_id=0, track_id=10),
        Annotation(
            timestamp=200,
            track_id=20,
            left=0.0,
            top=0.0,
            width=10.0,
            height=10.0,
            class_id=0,
        ),
    ]

    collection, counts = matched_prediction_collection(
        payload, annotations, "sequence", {0}, iou_threshold=0.5
    )

    assert counts["prediction_count"] == 1
