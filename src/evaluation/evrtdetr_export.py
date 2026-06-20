"""Detection export helpers for external EvRT-DETR checkpoints on DSEC-MOT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.evaluation.detection_export import (
    CLASS_NAMES,
    DetectionRecord,
    load_event_file,
    load_image_timestamps,
)
from src.evaluation.evrtdetr_runtime import (
    build_stacked_histogram,
    load_evrtdetr_model,
    run_evrtdetr_inference,
)

EVRTDETR_TO_DSEC_CLASS_ID = {
    "car": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "bike": 2,
    "cyclist": 2,
    "two-wheeler": 2,
    "two_wheeler": 2,
    "motorcycle": 3,
    "motorbike": 3,
    "bus": 4,
    "truck": 5,
    "train": 6,
}


def _normalise_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def build_dsec_class_id_by_model_label(class_names: list[str]) -> list[int]:
    """Map model label indices to zero-based DSEC-MOT class ids."""
    mapped: list[int] = []
    for label, class_name in enumerate(class_names):
        normalised = _normalise_class_name(class_name)
        if normalised not in EVRTDETR_TO_DSEC_CLASS_ID:
            valid = ", ".join(sorted(EVRTDETR_TO_DSEC_CLASS_ID))
            raise ValueError(
                f"Cannot map model class label {label} ('{class_name}') to DSEC-MOT. "
                f"Known class names: {valid}."
            )
        mapped.append(EVRTDETR_TO_DSEC_CLASS_ID[normalised])
    return mapped


def export_evrtdetr_detections_for_sequence(
    model_dir: Path,
    root: Path,
    split: str,
    sequence: str,
    output_path: Path,
    score_threshold: float = 0.35,
    device: str = "cpu",
    window_ms: float = 50.0,
    n_bins: int = 0,
    start_frame: int = 0,
    max_frames: int = 0,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> dict:
    seq_dir = root / split / sequence
    events_h5 = seq_dir / "events_left" / "events.h5"

    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory does not exist: {seq_dir}")
    if not events_h5.exists():
        raise FileNotFoundError(f"Missing events file: {events_h5}")

    timestamps = load_image_timestamps(seq_dir / f"{sequence}_image_timestamps.txt")
    frame_entries = list(enumerate(timestamps))
    if start_frame:
        frame_entries = frame_entries[start_frame:]
    if max_frames > 0:
        frame_entries = frame_entries[:max_frames]
    if not frame_entries:
        raise ValueError("No frames selected for export.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_evrtdetr_model(model_dir, device_name=device, forced_n_bins=n_bins)
    dsec_class_id_by_model_label = build_dsec_class_id_by_model_label(model.class_names)
    window_us = int(round(window_ms * 1000.0))

    all_detections: list[DetectionRecord] = []
    frames_payload = [
        {"frame_index": frame_index, "timestamp": timestamp}
        for frame_index, timestamp in frame_entries
    ]
    handle, x, y, p, t, ms_to_idx, t_offset, np_h5 = load_event_file(events_h5)
    try:
        total = len(frame_entries)
        for position, (frame_index, timestamp_us) in enumerate(frame_entries, start=1):
            hist = build_stacked_histogram(
                x=x,
                y=y,
                p=p,
                t=t,
                ms_to_idx=ms_to_idx,
                t_offset=t_offset,
                timestamp_us=timestamp_us,
                window_us=window_us,
                n_bins=model.n_bins,
                np_mod=np_h5,
            )
            predictions = run_evrtdetr_inference(
                model=model,
                hist=hist,
                score_threshold=score_threshold,
            )

            for prediction in predictions:
                if prediction.label < 0 or prediction.label >= len(dsec_class_id_by_model_label):
                    raise ValueError(
                        f"Prediction label {prediction.label} is outside model class range "
                        f"0..{len(dsec_class_id_by_model_label) - 1}."
                    )
                left, top, right, bottom = prediction.box
                all_detections.append(
                    DetectionRecord(
                        frame_index=frame_index,
                        timestamp=timestamp_us,
                        class_id=dsec_class_id_by_model_label[prediction.label],
                        score=prediction.score,
                        bbox_left=left,
                        bbox_top=top,
                        bbox_width=max(0.0, right - left),
                        bbox_height=max(0.0, bottom - top),
                    )
                )

            if progress_callback is not None:
                progress_callback(position, total, timestamp_us, len(predictions))
    finally:
        handle.close()

    payload = {
        "split": split,
        "sequence": sequence,
        "dataset_root": str(root),
        "model_dir": str(model.model_dir),
        "frame_count": len(frame_entries),
        "frame_count_total": len(timestamps),
        "score_threshold": score_threshold,
        "window_us": window_us,
        "n_bins": model.n_bins,
        "model_class_names": model.class_names,
        "dsec_class_names": CLASS_NAMES,
        "model_label_to_dsec_class_id": dsec_class_id_by_model_label,
        "class_mapping_note": (
            "Model classes are mapped to DSEC-MOT class ids before tracking. "
            "The public EvRT-DETR 'two-wheeler' class is evaluated as DSEC 'bicycle' "
            "because the current TrackEval adapter has no two-wheeler supercategory."
        ),
        "frames": frames_payload,
        "detections": [record.to_dict() for record in all_detections],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
