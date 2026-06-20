"""Detection-only metrics for DSEC-MOT detection exports."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.detection_export import CLASS_NAMES, load_annotations, load_detection_export


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x1 - x0, 0.0) * np.maximum(y1 - y0, 0.0)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    return inter / np.maximum(box_area + boxes_area - inter, 1e-9)


def _xywh_to_xyxy(row: dict) -> np.ndarray:
    left = float(row["bbox_left"])
    top = float(row["bbox_top"])
    return np.asarray(
        [left, top, left + float(row["bbox_width"]), top + float(row["bbox_height"])],
        dtype=np.float32,
    )


def _annotation_xyxy(annotation) -> np.ndarray:
    return np.asarray(
        [
            annotation.left,
            annotation.top,
            annotation.left + annotation.width,
            annotation.top + annotation.height,
        ],
        dtype=np.float32,
    )


def _average_precision(tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    if total_gt == 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    for index in range(precisions.size - 1, 0, -1):
        precisions[index - 1] = max(precisions[index - 1], precisions[index])
    changing = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[changing + 1] - recalls[changing]) * precisions[changing + 1]))


def evaluate_detection_export(
    detection_export_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate class-aware detections against DSEC-MOT annotations.

    The AP value is computed from the detections present in the export. If the export
    was created with a high score threshold, AP is correspondingly threshold-limited.
    """
    payload = load_detection_export(Path(detection_export_path))
    dataset_root = Path(payload["dataset_root"])
    split = payload["split"]
    sequence = payload["sequence"]
    annotations = load_annotations(dataset_root / "annotations" / split / f"{sequence}.txt")
    selected_timestamps = {int(frame["timestamp"]) for frame in payload["frames"]}

    gt_by_key: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for annotation in annotations:
        if annotation.timestamp not in selected_timestamps:
            continue
        gt_by_key[(annotation.timestamp, annotation.class_id)].append(_annotation_xyxy(annotation))

    detections_by_class: dict[int, list[dict]] = defaultdict(list)
    for row in payload["detections"]:
        detections_by_class[int(row["class_id"])].append(row)

    per_class: dict[str, dict] = {}
    total_tp = total_fp = total_gt = 0
    ap_values: list[float] = []
    for class_id, class_name in CLASS_NAMES.items():
        class_detections = sorted(
            detections_by_class.get(class_id, []),
            key=lambda row: float(row["score"]),
            reverse=True,
        )
        class_gt = sum(len(boxes) for key, boxes in gt_by_key.items() if key[1] == class_id)
        matched: dict[tuple[int, int], set[int]] = defaultdict(set)
        tp_flags: list[float] = []
        fp_flags: list[float] = []

        for detection in class_detections:
            key = (int(detection["timestamp"]), class_id)
            gt_boxes = np.asarray(gt_by_key.get(key, []), dtype=np.float32)
            det_box = _xywh_to_xyxy(detection)
            ious = box_iou_xyxy(det_box, gt_boxes)
            best_index = int(np.argmax(ious)) if ious.size else -1
            best_iou = float(ious[best_index]) if best_index >= 0 else 0.0
            if best_iou >= iou_threshold and best_index not in matched[key]:
                matched[key].add(best_index)
                tp_flags.append(1.0)
                fp_flags.append(0.0)
            else:
                tp_flags.append(0.0)
                fp_flags.append(1.0)

        tp = int(sum(tp_flags))
        fp = int(sum(fp_flags))
        fn = max(class_gt - tp, 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(class_gt, 1) if class_gt else float("nan")
        f1 = 2 * precision * recall / max(precision + recall, 1e-9) if class_gt else float("nan")
        ap50 = _average_precision(np.asarray(tp_flags), np.asarray(fp_flags), class_gt)
        if class_gt > 0:
            ap_values.append(ap50)
        per_class[class_name] = {
            "gt": class_gt,
            "detections": len(class_detections),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "AP50": ap50,
        }
        total_tp += tp
        total_fp += fp
        total_gt += class_gt

    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_gt, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-9)
    return {
        "detection_export": str(detection_export_path),
        "split": split,
        "sequence": sequence,
        "score_threshold": payload.get("score_threshold"),
        "iou_threshold": iou_threshold,
        "aggregate": {
            "mAP50": float(np.mean(ap_values)) if ap_values else float("nan"),
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": max(total_gt - total_tp, 0),
            "gt": total_gt,
            "detections": total_tp + total_fp,
        },
        "per_class": per_class,
    }
