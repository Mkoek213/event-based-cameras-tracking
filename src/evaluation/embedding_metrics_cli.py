#!/usr/bin/env python3
"""Evaluate event association embeddings independently of the tracker."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.data.representations import (
    BenchmarkRepresentation,
    representation_channel_splits,
    representation_components,
)
from src.evaluation.detection_export import (
    Annotation,
    DetectionRecord,
    load_annotations,
    load_event_file,
    load_image_timestamps,
    read_events,
)
from src.evaluation.embedding_metrics import (
    match_detections_to_annotations,
    summarise_embeddings,
)
from src.evaluation.simple_detector_trackeval_cli import (
    export_simple_detector_detections_for_sequence,
    load_model,
)
from src.models.simple_detector import normalise_event_tensor, normalise_representation_tensor


def parse_int_list(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def extract_gt_embeddings_for_sequence(
    model,
    root: Path,
    split: str,
    sequence: str,
    representation: str,
    num_bins: int,
    time_window_us: int,
    device: torch.device,
    input_normalisation: str,
    selected_classes: set[int] | None,
    max_frames: int = 0,
) -> dict[str, object]:
    """Extract one normalised descriptor for every selected ground-truth box."""

    if "eros" in representation_components(representation):
        raise ValueError("Embedding diagnostics currently support non-EROS representations only.")
    seq_dir = root / split / sequence
    timestamps = load_image_timestamps(seq_dir / f"{sequence}_image_timestamps.txt")
    if max_frames > 0:
        timestamps = timestamps[:max_frames]
    annotations = load_annotations(root / "annotations" / split / f"{sequence}.txt")
    annotations_by_timestamp: defaultdict[int, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        if selected_classes is not None and annotation.class_id not in selected_classes:
            continue
        if annotation.width > 0 and annotation.height > 0:
            annotations_by_timestamp[annotation.timestamp].append(annotation)

    transform = BenchmarkRepresentation(
        representation=representation,
        num_bins=num_bins,
        height=EVENT_HEIGHT,
        width=EVENT_WIDTH,
    )
    component_splits = (
        representation_channel_splits(representation, num_bins)
        if input_normalisation == "component"
        else ()
    )
    events_h5 = seq_dir / "events_left" / "events.h5"
    handle, x, y, p, t, ms_to_idx, t_offset, _ = load_event_file(events_h5)
    embedding_rows: list[torch.Tensor] = []
    class_ids: list[int] = []
    track_ids: list[int] = []
    frame_indices: list[int] = []
    embedding_state = None
    temporal_state = None
    try:
        for frame_index, timestamp in enumerate(timestamps):
            events = read_events(
                x=x,
                y=y,
                p=p,
                t=t,
                ms_to_idx=ms_to_idx,
                t_offset=t_offset,
                timestamp_us=timestamp,
                window_us=time_window_us,
            )
            dense = transform(events)
            tensor = torch.from_numpy(dense).float().unsqueeze(0).to(device)
            tensor = (
                normalise_representation_tensor(tensor, component_splits)
                if input_normalisation == "component"
                else normalise_event_tensor(tensor)
            )
            outputs = model(
                tensor,
                embedding_state=embedding_state,
                temporal_state=temporal_state,
            )
            embedding_state = outputs.get("embedding_state")
            temporal_state = outputs.get("temporal_state")
            frame_annotations = annotations_by_timestamp.get(timestamp, [])
            if not frame_annotations:
                continue
            boxes = tensor.new_tensor(
                [
                    [
                        max(0.0, annotation.left),
                        max(0.0, annotation.top),
                        min(float(EVENT_WIDTH), annotation.left + annotation.width),
                        min(float(EVENT_HEIGHT), annotation.top + annotation.height),
                    ]
                    for annotation in frame_annotations
                ]
            )
            if model.config.embedding_head_type == "dense":
                embedding_map = outputs.get("embeddings")
                if not isinstance(embedding_map, torch.Tensor):
                    raise RuntimeError("Dense checkpoint did not return embeddings.")
                descriptors = model.extract_dense_embeddings_at_boxes(embedding_map, [boxes])
            else:
                feature_map = outputs.get("embedding_feature_map")
                if not isinstance(feature_map, torch.Tensor):
                    raise RuntimeError("RoI checkpoint did not return embedding_feature_map.")
                descriptors = model.extract_roi_embeddings(feature_map, [boxes])
            embedding_rows.append(descriptors.detach().float().cpu())
            class_ids.extend(annotation.class_id for annotation in frame_annotations)
            track_ids.extend(annotation.track_id for annotation in frame_annotations)
            frame_indices.extend([frame_index] * len(frame_annotations))
    finally:
        handle.close()
    embeddings = (
        torch.cat(embedding_rows, dim=0)
        if embedding_rows
        else torch.empty((0, model.config.embedding_dim))
    )
    return {
        "embeddings": embeddings,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "track_ids": torch.tensor(track_ids, dtype=torch.long),
        "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
        "sequence_ids": [sequence] * len(class_ids),
    }


def append_collection(target: dict[str, object], source: dict[str, object]) -> None:
    for key in ("embeddings", "class_ids", "track_ids", "frame_indices"):
        target[key].append(source[key])
    target["sequence_ids"].extend(source["sequence_ids"])


def concatenate_collection(collection: dict[str, object], embedding_dim: int) -> dict[str, object]:
    result: dict[str, object] = {"sequence_ids": list(collection["sequence_ids"])}
    for key in ("class_ids", "track_ids", "frame_indices"):
        rows = collection[key]
        result[key] = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long)
    rows = collection["embeddings"]
    result["embeddings"] = torch.cat(rows, dim=0) if rows else torch.empty((0, embedding_dim))
    return result


def matched_prediction_collection(
    payload: dict,
    annotations: list[Annotation],
    sequence: str,
    selected_classes: set[int] | None,
    iou_threshold: float,
) -> tuple[dict[str, object], dict[str, int]]:
    """Assign predicted descriptors GT identities using class-aware IoU matching."""

    detections_by_timestamp: defaultdict[int, list[DetectionRecord]] = defaultdict(list)
    for row in payload.get("detections", []):
        detection = DetectionRecord(**row)
        if selected_classes is None or detection.class_id in selected_classes:
            detections_by_timestamp[detection.timestamp].append(detection)
    frame_by_timestamp = {
        int(frame["timestamp"]): int(frame["frame_index"]) for frame in payload.get("frames", [])
    }
    annotations_by_timestamp: defaultdict[int, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        if annotation.timestamp in frame_by_timestamp and (
            selected_classes is None or annotation.class_id in selected_classes
        ):
            annotations_by_timestamp[annotation.timestamp].append(annotation)
    embeddings: list[torch.Tensor] = []
    class_ids: list[int] = []
    track_ids: list[int] = []
    frame_indices: list[int] = []
    matched = 0
    matched_iou_sum = 0.0
    for timestamp, detections in detections_by_timestamp.items():
        frame_annotations = annotations_by_timestamp.get(timestamp, [])
        for detection_index, annotation_index, iou in match_detections_to_annotations(
            detections, frame_annotations, iou_threshold=iou_threshold
        ):
            detection = detections[detection_index]
            annotation = frame_annotations[annotation_index]
            if detection.embedding is None:
                continue
            embeddings.append(torch.tensor(detection.embedding, dtype=torch.float32))
            class_ids.append(annotation.class_id)
            track_ids.append(annotation.track_id)
            frame_indices.append(frame_by_timestamp[timestamp])
            matched += 1
            matched_iou_sum += iou
    embedding_dim = int(payload.get("embedding_dim", 0))
    collection = {
        "embeddings": torch.stack(embeddings) if embeddings else torch.empty((0, embedding_dim)),
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "track_ids": torch.tensor(track_ids, dtype=torch.long),
        "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
        "sequence_ids": [sequence] * len(class_ids),
    }
    counts = {
        "prediction_count": sum(len(rows) for rows in detections_by_timestamp.values()),
        "gt_count": sum(len(rows) for rows in annotations_by_timestamp.values()),
        "matched_count": matched,
        "matched_iou_sum_milli": int(round(matched_iou_sum * 1000.0)),
    }
    return collection, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--class-ids", default=None)
    parser.add_argument("--representation", default="event_frame_voxel_grid")
    parser.add_argument("--num-bins", type=int, default=3)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument(
        "--input-normalisation", choices=("whole", "component"), default="component"
    )
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-pairs-per-group", type=int, default=200_000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    if model.config.embedding_dim <= 0:
        raise SystemExit("The selected checkpoint has no association embedding head.")
    checkpoint["checkpoint_path"] = str(args.checkpoint.resolve())
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    selected_classes = parse_int_list(args.class_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_parts: dict[str, object] = {
        "embeddings": [],
        "class_ids": [],
        "track_ids": [],
        "frame_indices": [],
        "sequence_ids": [],
    }
    predicted_parts: dict[str, object] = {
        "embeddings": [],
        "class_ids": [],
        "track_ids": [],
        "frame_indices": [],
        "sequence_ids": [],
    }
    total_counts: defaultdict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for sequence in sequences:
            prediction_path = args.output_dir / "detections" / f"{sequence}.json"
            payload = export_simple_detector_detections_for_sequence(
                model=model,
                checkpoint=checkpoint,
                root=args.root,
                split=args.split,
                sequence=sequence,
                output_path=prediction_path,
                score_threshold=args.score_threshold,
                nms_iou_threshold=args.nms_iou_threshold,
                max_detections=args.max_detections,
                representation=args.representation,
                num_bins=args.num_bins,
                time_window_us=args.time_window_us,
                device=device,
                max_frames=args.max_frames,
                input_normalisation=args.input_normalisation,
            )
            gt_part = extract_gt_embeddings_for_sequence(
                model=model,
                root=args.root,
                split=args.split,
                sequence=sequence,
                representation=args.representation,
                num_bins=args.num_bins,
                time_window_us=args.time_window_us,
                device=device,
                input_normalisation=args.input_normalisation,
                selected_classes=selected_classes,
                max_frames=args.max_frames,
            )
            annotations = load_annotations(
                args.root / "annotations" / args.split / f"{sequence}.txt"
            )
            predicted_part, counts = matched_prediction_collection(
                payload, annotations, sequence, selected_classes, args.match_iou_threshold
            )
            append_collection(gt_parts, gt_part)
            append_collection(predicted_parts, predicted_part)
            for key, value in counts.items():
                total_counts[key] += value

    gt = concatenate_collection(gt_parts, model.config.embedding_dim)
    predicted = concatenate_collection(predicted_parts, model.config.embedding_dim)
    gt_summary = summarise_embeddings(**gt, max_pairs_per_group=args.max_pairs_per_group)
    predicted_summary = summarise_embeddings(
        **predicted, max_pairs_per_group=args.max_pairs_per_group
    )
    matched = total_counts["matched_count"]
    coverage = {
        **dict(total_counts),
        "matched_prediction_fraction": matched / max(total_counts["prediction_count"], 1),
        "matched_gt_fraction": matched / max(total_counts["gt_count"], 1),
        "mean_matched_iou": total_counts["matched_iou_sum_milli"] / max(matched, 1) / 1000.0,
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sequences": sequences,
        "class_ids": sorted(selected_classes) if selected_classes is not None else None,
        "score_threshold": args.score_threshold,
        "match_iou_threshold": args.match_iou_threshold,
        "ground_truth_boxes": gt_summary,
        "matched_predicted_boxes": predicted_summary,
        "prediction_matching": coverage,
    }
    (args.output_dir / "embedding_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "embedding_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source",
                "embedding_count",
                "retrieval_map",
                "retrieval_rank1",
                "valid_queries",
                "mean_cosine_margin",
            ),
        )
        writer.writeheader()
        for source, values in (
            ("ground_truth_boxes", gt_summary),
            ("matched_predicted_boxes", predicted_summary),
        ):
            writer.writerow(
                {
                    "source": source,
                    **{key: values[key] for key in writer.fieldnames if key != "source"},
                }
            )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
