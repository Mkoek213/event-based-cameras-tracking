"""Streamed COCO detection mAP for the large event MOT detector.

Two uses during training (``configs/large_event_mot/training.json``):

- 1Mpx source validation (pretraining selection): COCO mAP@[0.5:0.95] over pedestrian,
  two-wheeler, car, truck, and bus, plus the 3-class subset (pedestrian, two-wheeler,
  car) for comparison with RVT, AP50, and per-class AP.
- DSEC-Detection validation monitor (fine-tuning): the six target classes without
  ``train``; a logged monitor, never a selection input.

Protocol: every sequence starts from a zero recurrent state and is streamed causally
with the state carried through all of its steps; the model runs in eval mode in fp32
(the caller passes the EMA weights). Only supervised (annotated) steps are evaluated, so
unannotated windows never count as empty ground truth. Ground truth is every box of an
evaluated class at those steps that passes the dataset's model-frame size rule (1Mpx:
60 px diagonal and 20 px side at native resolution; DSEC-Detection: 614.4 px² rectified
area), including boxes that training ignores only because their window has no events.
Predictions use YOLOX decoding with the configured score floor, class-aware NMS, and at
most ``max_detections`` per step, and the same size rule.

Several sequences run side by side in ``lanes`` (one batch row each); a lane that
finishes a sequence starts its next one with a reset, so lanes never share state.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch

from src.data.large_event_dataset import (
    ClipSpec,
    LargeEventClipDataset,
    SequenceData,
    collate_clips,
    network_input,
    sequence_clips,
)
from src.models.large_event_detector import LargeEventDetector, decode_detections
from src.models.large_event_modules import RecurrentState


class DetectionMapError(ValueError):
    """The evaluation request is inconsistent with the model or the sequences."""


@dataclass(frozen=True)
class MapSettings:
    """Classes, size rule, decoding, and streaming settings of one mAP evaluation."""

    classes: tuple[int, ...]
    class_names: tuple[str, ...]
    subset_classes: tuple[int, ...]
    small_box: Mapping[str, Optional[float]]
    score_floor: float
    nms_iou: float
    max_detections: int
    clip_steps: int
    lanes: int

    def __post_init__(self) -> None:
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise DetectionMapError("evaluation classes must be distinct and non-empty")
        if not set(self.subset_classes) <= set(self.classes):
            raise DetectionMapError("subset classes must be evaluation classes")
        if self.clip_steps < 1 or self.lanes < 1 or self.max_detections < 1:
            raise DetectionMapError("clip_steps, lanes, and max_detections must be positive")


def map_settings(
    validation: Mapping[str, Any],
    head_classes: Sequence[str],
    small_box: Mapping[str, Optional[float]],
) -> MapSettings:
    """Settings from a ``validation`` (or ``monitor``) block of ``training.json``."""

    classes = tuple(int(label) for label in validation["classes"])
    if any(not 0 <= label < len(head_classes) for label in classes):
        raise DetectionMapError(f"evaluation classes {classes} are not head rows")
    return MapSettings(
        classes=classes,
        class_names=tuple(head_classes[label] for label in classes),
        subset_classes=tuple(int(label) for label in validation.get("subset_classes", [])),
        small_box=dict(small_box),
        score_floor=float(validation["score_floor"]),
        nms_iou=float(validation["nms_iou"]),
        max_detections=int(validation["max_detections"]),
        clip_steps=int(validation["clip_steps"]),
        lanes=int(validation["lanes"]),
    )


def passes_size_rule(xyxy: np.ndarray, rule: Mapping[str, Optional[float]]) -> np.ndarray:
    """Boxes that meet every configured model-frame threshold."""

    boxes = np.asarray(xyxy, np.float64).reshape(-1, 4)
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    keep = (width > 0) & (height > 0)
    if rule.get("min_diagonal") is not None:
        keep &= np.hypot(width, height) >= rule["min_diagonal"]
    if rule.get("min_side") is not None:
        keep &= np.minimum(width, height) >= rule["min_side"]
    if rule.get("min_area") is not None:
        keep &= width * height >= rule["min_area"]
    return keep


# --------------------------------------------------------------------------- streaming


def lane_schedule(
    sequences: Sequence[SequenceData], clip_steps: int, lanes: int
) -> list[list[ClipSpec]]:
    """Ordered evaluation clips per lane; sequences are dealt round-robin to lanes."""

    schedule: list[list[ClipSpec]] = [[] for _ in range(min(lanes, len(sequences)))]
    for index, sequence in enumerate(sequences):
        schedule[index % len(schedule)].extend(sequence_clips(sequence, clip_steps))
    return schedule


def _iterate_lanes(
    dataset: LargeEventClipDataset, schedule: Sequence[Sequence[ClipSpec]]
) -> Iterator[tuple[list[int], dict[str, Any]]]:
    for position in range(max((len(specs) for specs in schedule), default=0)):
        active = [lane for lane, specs in enumerate(schedule) if position < len(specs)]
        yield active, collate_clips([dataset[schedule[lane][position]] for lane in active])


@dataclass
class _Collected:
    images: list[dict[str, Any]]
    ground_truth: list[tuple[int, np.ndarray, int]]
    predictions: list[tuple[int, np.ndarray, float, int]]
    frames: int = 0


def collect_detections(
    model: LargeEventDetector,
    dataset: LargeEventClipDataset,
    sequences: Sequence[SequenceData],
    settings: MapSettings,
    device: torch.device | str,
) -> _Collected:
    """Stream ``sequences`` and gather ground truth and predictions at supervised steps."""

    if not sequences:
        raise DetectionMapError("no sequences to evaluate")
    for sequence in sequences:
        if sequence.split == "test":
            raise DetectionMapError(f"{sequence.sequence}: test data is never evaluated here")
    schedule = lane_schedule(sequences, settings.clip_steps, settings.lanes)
    states: list[Optional[RecurrentState]] = [None] * len(schedule)
    collected = _Collected(images=[], ground_truth=[], predictions=[])
    evaluated = set(settings.classes)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            for active, batch in _iterate_lanes(dataset, schedule):
                event_frame, voxel_grid = network_input(batch, device)
                rows = [
                    states[lane] if states[lane] is not None else model.initial_state(1, device)
                    for lane in active
                ]
                state = RecurrentState.cat(rows).reset(batch["reset"].to(device))
                valid = batch["valid"]
                for step in range(valid.shape[1]):
                    output = model(
                        event_frame[:, step],
                        voxel_grid[:, step],
                        state=state,
                        valid=valid[:, step].to(device),
                        with_embeddings=False,
                    )
                    state = output.state
                    collected.frames += int(valid[:, step].sum())
                    rows_to_decode = [
                        row
                        for row in range(len(active))
                        if valid[row, step] and batch["supervised"][row, step]
                    ]
                    if not rows_to_decode:
                        continue
                    detections = decode_detections(
                        output.head,
                        model.geometry.valid_hw,
                        settings.score_floor,
                        settings.nms_iou,
                        settings.max_detections,
                    )
                    for row in rows_to_decode:
                        image_id = len(collected.images) + 1
                        meta = batch["meta"][row]
                        collected.images.append(
                            {
                                "id": image_id,
                                "dataset": meta["dataset"],
                                "sequence": meta["sequence"],
                                "step": int(batch["step_index"][row, step]),
                            }
                        )
                        target = batch["targets"][row][step]
                        boxes = target["boxes_xyxy"].numpy()
                        labels = target["labels"].numpy()
                        keep = passes_size_rule(boxes, settings.small_box) & np.isin(
                            labels, list(evaluated)
                        )
                        collected.ground_truth.extend(
                            (image_id, box, int(label))
                            for box, label in zip(boxes[keep], labels[keep])
                        )
                        found = detections[row]
                        boxes = found.boxes.float().cpu().numpy()
                        labels = found.labels.cpu().numpy()
                        scores = found.scores.float().cpu().numpy()
                        keep = passes_size_rule(boxes, settings.small_box) & np.isin(
                            labels, list(evaluated)
                        )
                        collected.predictions.extend(
                            (image_id, box, float(score), int(label))
                            for box, score, label in zip(boxes[keep], scores[keep], labels[keep])
                        )
                for row, lane in enumerate(active):
                    states[lane] = state.rows([row])
    finally:
        model.train(was_training)
    return collected


# --------------------------------------------------------------------------- COCO


def _xywh(box: np.ndarray) -> list[float]:
    x0, y0, x1, y1 = (float(value) for value in box)
    return [x0, y0, x1 - x0, y1 - y0]


def coco_metrics(
    collected: _Collected, classes: Sequence[int], max_detections: int
) -> dict[str, Any]:
    """COCO bbox metrics over ``classes`` (head rows); category id is ``row + 1``."""

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    class_ids = [int(label) + 1 for label in classes]
    gt_classes = {label for _, _, label in collected.ground_truth if label in classes}
    result: dict[str, Any] = {
        "map": None,
        "ap50": None,
        "ap75": None,
        "per_class_ap": {int(label): None for label in classes},
    }
    if not gt_classes:
        return result
    annotations = []
    for index, (image_id, box, label) in enumerate(collected.ground_truth, start=1):
        if label not in classes:
            continue
        xywh = _xywh(box)
        annotations.append(
            {
                "id": index,
                "image_id": image_id,
                "category_id": label + 1,
                "bbox": xywh,
                "area": xywh[2] * xywh[3],
                "iscrowd": 0,
            }
        )
    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {"description": "large event MOT detection validation"},
        "images": [{"id": image["id"]} for image in collected.images],
        "annotations": annotations,
        "categories": [{"id": class_id, "name": str(class_id)} for class_id in class_ids],
    }
    predictions = [
        {"image_id": image_id, "category_id": label + 1, "bbox": _xywh(box), "score": score}
        for image_id, box, score, label in collected.predictions
        if label in classes
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
        if not predictions:
            result.update({"map": 0.0, "ap50": 0.0, "ap75": 0.0})
            result["per_class_ap"] = {
                int(label): (0.0 if label in gt_classes else None) for label in classes
            }
            return result
        detections = ground_truth.loadRes(predictions)
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.catIds = class_ids
        evaluator.params.maxDets = [1, 10, int(max_detections)]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]  # [IoU, recall, class, area, maxDets]
    for position, label in enumerate(classes):
        values = precision[:, :, position, 0, -1]
        values = values[values > -1]
        result["per_class_ap"][int(label)] = float(values.mean()) if values.size else None
    result.update(
        {
            "map": float(evaluator.stats[0]),
            "ap50": float(evaluator.stats[1]),
            "ap75": float(evaluator.stats[2]),
        }
    )
    return result


def evaluate_detection_map(
    model: LargeEventDetector,
    dataset: LargeEventClipDataset,
    sequences: Sequence[SequenceData],
    settings: MapSettings,
    device: torch.device | str,
) -> dict[str, Any]:
    """Stream ``sequences`` with ``model`` (EMA weights) and return COCO metrics.

    ``map`` and ``ap50`` cover ``settings.classes``; ``subset_map`` and ``subset_ap50``
    cover ``settings.subset_classes`` when configured. ``None`` marks a metric without
    ground truth. Counts describe the evaluated steps and boxes.
    """

    started = time.perf_counter()
    collected = collect_detections(model, dataset, sequences, settings, device)
    metrics = coco_metrics(collected, settings.classes, settings.max_detections)
    names = dict(zip(settings.classes, settings.class_names))
    summary: dict[str, Any] = {
        "map": metrics["map"],
        "ap50": metrics["ap50"],
        "ap75": metrics["ap75"],
        "per_class_ap": {names[label]: ap for label, ap in metrics["per_class_ap"].items()},
        "classes": list(settings.class_names),
    }
    if settings.subset_classes:
        subset = coco_metrics(collected, settings.subset_classes, settings.max_detections)
        summary.update(
            {
                "subset_map": subset["map"],
                "subset_ap50": subset["ap50"],
                "subset_classes": [names[label] for label in settings.subset_classes],
            }
        )
    gt_per_class = {name: 0 for name in settings.class_names}
    for _, _, label in collected.ground_truth:
        gt_per_class[names[label]] += 1
    summary["counts"] = {
        "sequences": len(sequences),
        "frames": collected.frames,
        "evaluated_steps": len(collected.images),
        "ground_truth_boxes": len(collected.ground_truth),
        "ground_truth_per_class": gt_per_class,
        "predictions": len(collected.predictions),
    }
    summary["settings"] = {
        "score_floor": settings.score_floor,
        "nms_iou": settings.nms_iou,
        "max_detections": settings.max_detections,
        "size_rule": dict(settings.small_box),
        "precision": "fp32",
        "state": "reset per sequence",
    }
    summary["seconds"] = time.perf_counter() - started
    return summary
