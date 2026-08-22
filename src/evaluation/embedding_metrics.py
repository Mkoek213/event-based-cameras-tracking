"""Tracker-independent diagnostics for association embeddings."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.detection_export import Annotation, DetectionRecord

TEMPORAL_GAP_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17+", 17, None),
)


def _validate_inputs(
    embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    sequence_ids: Sequence[str],
    track_ids: torch.Tensor,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    embeddings = embeddings.detach().float().cpu()
    class_ids = class_ids.detach().long().cpu()
    track_ids = track_ids.detach().long().cpu()
    frame_indices = frame_indices.detach().long().cpu()
    count = embeddings.shape[0]
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D).")
    if not (
        class_ids.shape == track_ids.shape == frame_indices.shape == (count,)
        and len(sequence_ids) == count
    ):
        raise ValueError("Embedding metadata does not align with the embedding rows.")
    return F.normalize(embeddings, dim=1) if count else embeddings


def class_aware_retrieval_metrics(
    embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    sequence_ids: Sequence[str],
    track_ids: torch.Tensor,
) -> dict[str, float | int]:
    """Compute leave-one-out retrieval mAP and Rank-1 within each class."""

    class_ids = class_ids.detach().long().cpu()
    track_ids = track_ids.detach().long().cpu()
    frame_indices = torch.zeros_like(track_ids)
    embeddings = _validate_inputs(embeddings, class_ids, sequence_ids, track_ids, frame_indices)
    count = embeddings.shape[0]
    if count < 2:
        return {"retrieval_map": 0.0, "retrieval_rank1": 0.0, "valid_queries": 0}
    similarities = embeddings @ embeddings.T
    average_precisions: list[float] = []
    rank1_hits: list[float] = []
    for query in range(count):
        gallery = class_ids == class_ids[query]
        gallery[query] = False
        positives = gallery & (track_ids == track_ids[query])
        positives &= torch.tensor(
            [sequence == sequence_ids[query] for sequence in sequence_ids], dtype=torch.bool
        )
        if not positives.any():
            continue
        gallery_indices = torch.nonzero(gallery, as_tuple=False).flatten()
        order = gallery_indices[similarities[query, gallery_indices].argsort(descending=True)]
        ranked_positive = positives[order]
        positive_ranks = torch.nonzero(ranked_positive, as_tuple=False).flatten() + 1
        precisions = torch.arange(1, positive_ranks.numel() + 1).float() / positive_ranks.float()
        average_precisions.append(float(precisions.mean()))
        rank1_hits.append(float(ranked_positive[0]))
    if not average_precisions:
        return {"retrieval_map": 0.0, "retrieval_rank1": 0.0, "valid_queries": 0}
    return {
        "retrieval_map": float(np.mean(average_precisions)),
        "retrieval_rank1": float(np.mean(rank1_hits)),
        "valid_queries": len(average_precisions),
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def cosine_pair_diagnostics(
    embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    sequence_ids: Sequence[str],
    track_ids: torch.Tensor,
    frame_indices: torch.Tensor,
    max_pairs_per_group: int = 200_000,
) -> dict[str, object]:
    """Summarise positive and class-matched negative cosine similarities."""

    class_ids = class_ids.detach().long().cpu()
    track_ids = track_ids.detach().long().cpu()
    frame_indices = frame_indices.detach().long().cpu()
    embeddings = _validate_inputs(embeddings, class_ids, sequence_ids, track_ids, frame_indices)
    positive: list[float] = []
    negative: list[float] = []
    temporal: defaultdict[str, list[float]] = defaultdict(list)
    count = embeddings.shape[0]
    for first in range(count):
        if len(positive) >= max_pairs_per_group and len(negative) >= max_pairs_per_group:
            break
        similarities = embeddings[first + 1 :] @ embeddings[first]
        for offset, similarity in enumerate(similarities.tolist(), start=first + 1):
            if class_ids[offset] != class_ids[first]:
                continue
            same_identity = bool(
                sequence_ids[offset] == sequence_ids[first]
                and track_ids[offset] == track_ids[first]
            )
            target = positive if same_identity else negative
            if len(target) < max_pairs_per_group:
                target.append(float(similarity))
            if same_identity:
                gap = abs(int(frame_indices[offset]) - int(frame_indices[first]))
                for label, minimum, maximum in TEMPORAL_GAP_BUCKETS:
                    if gap >= minimum and (maximum is None or gap <= maximum):
                        if len(temporal[label]) < max_pairs_per_group:
                            temporal[label].append(float(similarity))
                        break
    positive_stats = _distribution(positive)
    negative_stats = _distribution(negative)
    margin = (
        float(positive_stats["mean"] - negative_stats["mean"])
        if positive_stats["count"] and negative_stats["count"]
        else 0.0
    )
    return {
        "same_identity_cosine": positive_stats,
        "different_identity_same_class_cosine": negative_stats,
        "mean_cosine_margin": margin,
        "temporal_positive_cosine": {
            label: _distribution(temporal[label]) for label, _, _ in TEMPORAL_GAP_BUCKETS
        },
        "pair_sampling": "deterministic prefix capped independently per group",
        "max_pairs_per_group": max_pairs_per_group,
    }


def summarise_embeddings(
    embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    sequence_ids: Sequence[str],
    track_ids: torch.Tensor,
    frame_indices: torch.Tensor,
    max_pairs_per_group: int = 200_000,
) -> dict[str, object]:
    """Return retrieval and pair-separation metrics for one descriptor source."""

    result: dict[str, object] = {
        "embedding_count": int(embeddings.shape[0]),
        **class_aware_retrieval_metrics(embeddings, class_ids, sequence_ids, track_ids),
    }
    result.update(
        cosine_pair_diagnostics(
            embeddings,
            class_ids,
            sequence_ids,
            track_ids,
            frame_indices,
            max_pairs_per_group=max_pairs_per_group,
        )
    )
    return result


def box_iou_xyxy(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for two XYXY box tensors."""

    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != (4,) or second.shape[1:] != (4,):
        raise ValueError("Boxes must have shape (N, 4) and (M, 4).")
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=2)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp_min(1e-9)


def match_detections_to_annotations(
    detections: Sequence[DetectionRecord],
    annotations: Sequence[Annotation],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedily match scored detections to same-class GT boxes at one timestamp."""

    if not detections or not annotations:
        return []
    detection_boxes = torch.tensor(
        [
            [d.bbox_left, d.bbox_top, d.bbox_left + d.bbox_width, d.bbox_top + d.bbox_height]
            for d in detections
        ],
        dtype=torch.float32,
    )
    annotation_boxes = torch.tensor(
        [[a.left, a.top, a.left + a.width, a.top + a.height] for a in annotations],
        dtype=torch.float32,
    )
    ious = box_iou_xyxy(detection_boxes, annotation_boxes)
    candidates: list[tuple[float, float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        for annotation_index, annotation in enumerate(annotations):
            iou = float(ious[detection_index, annotation_index])
            if detection.class_id == annotation.class_id and iou >= iou_threshold:
                candidates.append((detection.score, iou, detection_index, annotation_index))
    matched_detections: set[int] = set()
    matched_annotations: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for _, iou, detection_index, annotation_index in sorted(candidates, reverse=True):
        if detection_index in matched_detections or annotation_index in matched_annotations:
            continue
        matched_detections.add(detection_index)
        matched_annotations.add(annotation_index)
        matches.append((detection_index, annotation_index, iou))
    return matches
