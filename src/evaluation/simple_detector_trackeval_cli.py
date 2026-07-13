#!/usr/bin/env python3
"""Export detections from the lightweight detector, track them, and run TrackEval.

Appearance-based association with an embedding-head checkpoint
(trained via ``src.training.recurrent_embedding_detector``):

    .venv/bin/python -m src.evaluation.simple_detector_trackeval_cli \\
      --checkpoint runs/recurrent_embedding/<run>/best.pt \\
      --root data/datasets/dsec_mot \\
      --split test --sequences interlaken_00_d \\
      --tracker-backend boxmot_botsort --track-with-reid \\
      --device cuda \\
      --run-name gated_recurrent_embed_botsort_reid

Motion-only BoT-SORT ablation on the same checkpoint: drop ``--track-with-reid``.
IoU baseline on a plain gated checkpoint: ``--tracker-backend iou``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.data.pretrained_detection_dataset import ErosSnapshotStore
from src.data.representations import (
    REPRESENTATION_CHOICES,
    BenchmarkRepresentation,
    representation_channel_splits,
    representation_components,
)
from src.evaluation.detection_export import (
    DetectionRecord,
    load_event_file,
    load_image_timestamps,
    read_events,
)
from src.evaluation.mot_trackers import TrackingConfig, track_detections
from src.evaluation.trackeval_adapter import (
    TRACKEVAL_CLASS_NAMES,
    export_trackeval_bundle,
    run_trackeval,
    summarise_trackeval_results,
    write_summary_csv,
    write_summary_json,
)
from src.models.simple_detector import (
    SimpleDenseDetector,
    SimpleDetectorConfig,
    decode_dense_detections,
    normalise_event_tensor,
    normalise_representation_tensor,
)

DEFAULT_TEST_SEQUENCES = ["interlaken_00_d", "zurich_city_00_b"]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SimpleDenseDetector, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = SimpleDetectorConfig(**checkpoint["model_config"])
    model = SimpleDenseDetector(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def export_simple_detector_detections_for_sequence(
    model: SimpleDenseDetector,
    checkpoint: dict,
    root: Path,
    split: str,
    sequence: str,
    output_path: Path,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    representation: str,
    num_bins: int,
    time_window_us: int,
    device: torch.device,
    eros_store: ErosSnapshotStore | None = None,
    start_frame: int = 0,
    max_frames: int = 0,
    input_normalisation: str = "whole",
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
    transform = BenchmarkRepresentation(
        representation=representation,
        num_bins=num_bins,
        height=EVENT_HEIGHT,
        width=EVENT_WIDTH,
    )

    if input_normalisation not in ("whole", "component"):
        raise ValueError(f"Unknown input_normalisation '{input_normalisation}'.")
    component_splits = (
        representation_channel_splits(representation, num_bins)
        if input_normalisation == "component"
        else ()
    )
    has_embedding_head = model.config.embedding_dim > 0

    all_detections: list[DetectionRecord] = []
    frames_payload = [
        {"frame_index": frame_index, "timestamp": timestamp}
        for frame_index, timestamp in frame_entries
    ]
    handle, x, y, p, t, ms_to_idx, t_offset, _ = load_event_file(events_h5)
    started = time.perf_counter()
    embedding_state = None
    try:
        total = len(frame_entries)
        for position, (frame_index, timestamp_us) in enumerate(frame_entries, start=1):
            events = read_events(
                x=x,
                y=y,
                p=p,
                t=t,
                ms_to_idx=ms_to_idx,
                t_offset=t_offset,
                timestamp_us=timestamp_us,
                window_us=time_window_us,
            )
            eros = (
                eros_store.get(split, sequence, frame_index, timestamp_us)
                if eros_store is not None
                else None
            )
            dense = transform(events, eros=eros)
            tensor = torch.from_numpy(dense).float().unsqueeze(0).to(device, non_blocking=True)
            if input_normalisation == "component":
                tensor = normalise_representation_tensor(tensor, component_splits)
            else:
                tensor = normalise_event_tensor(tensor)
            with torch.inference_mode():
                outputs = model(tensor, embedding_state)
                embedding_state = outputs.get("embedding_state")
                detections = decode_dense_detections(
                    outputs=outputs,
                    frame_index=frame_index,
                    timestamp=timestamp_us,
                    score_threshold=score_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                    max_detections=max_detections,
                    feature_stride=model.config.feature_stride,
                    embeddings=outputs.get("embeddings") if has_embedding_head else None,
                )
            all_detections.extend(detections)
            if position == 1 or position == total or position % 100 == 0:
                print(
                    f"[{sequence}] {position}/{total}: "
                    f"t={timestamp_us} detections={len(detections)}"
                )
    finally:
        handle.close()
    elapsed_s = time.perf_counter() - started

    payload = {
        "split": split,
        "sequence": sequence,
        "dataset_root": str(root),
        "checkpoint": str(checkpoint.get("checkpoint_path", "")),
        "frame_count": len(frame_entries),
        "frame_count_total": len(timestamps),
        "score_threshold": score_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "max_detections": max_detections,
        "representation": representation,
        "num_bins": num_bins,
        "time_window_us": time_window_us,
        "input_normalisation": input_normalisation,
        "embedding_dim": model.config.embedding_dim,
        "model_config": checkpoint["model_config"],
        "benchmark_config": checkpoint.get("benchmark_config", {}),
        "elapsed_s": elapsed_s,
        "fps": len(frame_entries) / elapsed_s if elapsed_s > 0 else 0.0,
        "frames": frames_payload,
        "detections": [record.to_dict() for record in all_detections],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--representation", choices=REPRESENTATION_CHOICES, default=None)
    parser.add_argument("--num-bins", type=int, default=None)
    parser.add_argument("--time-window-us", type=int, default=None)
    parser.add_argument("--eros-cache-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--track-iou-threshold", type=float, default=0.5)
    parser.add_argument("--track-max-missed-frames", type=int, default=2)
    parser.add_argument("--track-min-hits", type=int, default=1)
    parser.add_argument(
        "--tracker-backend",
        choices=[
            "iou",
            "sort",
            "bytetrack",
            "boxmot_bytetrack",
            "boxmot_ocsort",
            "boxmot_botsort",
        ],
        default="iou",
        help="Tracking backend used after detector export.",
    )
    parser.add_argument("--track-high-threshold", type=float, default=0.6)
    parser.add_argument("--track-low-threshold", type=float, default=0.1)
    parser.add_argument(
        "--track-with-reid",
        action="store_true",
        help=(
            "Feed exported detection embeddings into BoT-SORT appearance matching. "
            "Requires --tracker-backend boxmot_botsort and an embedding-head checkpoint."
        ),
    )
    parser.add_argument("--track-appearance-thresh", type=float, default=0.25)
    parser.add_argument("--track-proximity-thresh", type=float, default=0.5)
    parser.add_argument(
        "--input-normalisation",
        choices=("auto", "whole", "component"),
        default="auto",
        help=(
            "Input normalisation during export. 'component' matches training for fused "
            "representations; 'whole' is the historical export behaviour. 'auto' picks "
            "'component' for embedding-head checkpoints and 'whole' otherwise, keeping "
            "previously reported baselines unchanged."
        ),
    )
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--classes-to-eval",
        nargs="+",
        choices=TRACKEVAL_CLASS_NAMES,
        default=None,
        help="Optional subset of TrackEval classes. Use 'car' for car-only evaluation.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tracker-name", default=None)
    parser.add_argument("--trackeval-root", type=Path, default=Path("external/TrackEval"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/dsec_mot_trackeval_simple_detector")
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")

    model, checkpoint = load_model(args.checkpoint, device)
    checkpoint["checkpoint_path"] = str(args.checkpoint)
    if args.track_with_reid and args.tracker_backend != "boxmot_botsort":
        raise SystemExit("--track-with-reid requires --tracker-backend boxmot_botsort.")
    if args.track_with_reid and model.config.embedding_dim <= 0:
        raise SystemExit(
            "--track-with-reid requires a checkpoint with an embedding head "
            "(train with src.training.recurrent_embedding_detector)."
        )
    input_normalisation = args.input_normalisation
    if input_normalisation == "auto":
        input_normalisation = "component" if model.config.embedding_dim > 0 else "whole"
    benchmark_config = checkpoint.get("benchmark_config", {})
    representation = args.representation or benchmark_config.get("representation", "voxel_grid")
    num_bins = (
        args.num_bins if args.num_bins is not None else int(benchmark_config.get("num_bins", 5))
    )
    time_window_us = (
        args.time_window_us
        if args.time_window_us is not None
        else int(benchmark_config.get("time_window_us", 50_000))
    )
    sequences = args.sequences or (DEFAULT_TEST_SEQUENCES if args.split == "test" else [])
    if not sequences:
        raise SystemExit("No sequences selected. Pass --sequences explicitly for non-test splits.")
    eros_store = (
        ErosSnapshotStore(args.eros_cache_root)
        if "eros" in representation_components(representation)
        else None
    )

    tracker_suffix = (
        f"{args.tracker_backend}_reid" if args.track_with_reid else args.tracker_backend
    )
    tracker_name = args.tracker_name or f"simple_dense_detector_{tracker_suffix}"
    run_name = args.run_name or f"{args.checkpoint.stem}_{representation}_{args.split}"
    output_root = args.output_root / run_name
    detections_dir = output_root / "detections"
    tracks_dir = output_root / "tracks"
    trackeval_dir = output_root / "trackeval"
    output_root.mkdir(parents=True, exist_ok=True)

    detection_runs: dict[str, dict] = {}
    tracker_runs: dict[str, dict] = {}
    for sequence in sequences:
        detection_path = detections_dir / f"{sequence}.json"
        print(f"Exporting detections for {sequence} -> {detection_path}")
        detection_runs[sequence] = export_simple_detector_detections_for_sequence(
            model=model,
            checkpoint=checkpoint,
            root=args.root,
            split=args.split,
            sequence=sequence,
            output_path=detection_path,
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            max_detections=args.max_detections,
            representation=representation,
            num_bins=num_bins,
            time_window_us=time_window_us,
            device=device,
            eros_store=eros_store,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            input_normalisation=input_normalisation,
        )

        track_path = tracks_dir / f"{sequence}.txt"
        print(f"Tracking {sequence} -> {track_path}")
        tracker_runs[sequence] = track_detections(
            detection_export_path=detection_path,
            output_path=track_path,
            config=TrackingConfig(
                backend=args.tracker_backend,
                iou_threshold=args.track_iou_threshold,
                max_missed_frames=args.track_max_missed_frames,
                min_hits=args.track_min_hits,
                high_threshold=args.track_high_threshold,
                low_threshold=args.track_low_threshold,
                with_reid=args.track_with_reid,
                appearance_thresh=args.track_appearance_thresh,
                proximity_thresh=args.track_proximity_thresh,
            ),
        )

    (output_root / "detection_runs.json").write_text(
        json.dumps(detection_runs, indent=2), encoding="utf-8"
    )
    (output_root / "tracker_runs.json").write_text(
        json.dumps(tracker_runs, indent=2), encoding="utf-8"
    )

    print(f"Preparing TrackEval bundle in {trackeval_dir}")
    bundle = export_trackeval_bundle(
        dataset_root=args.root,
        split=args.split,
        sequences=sequences,
        tracker_name=tracker_name,
        tracker_results_dir=tracks_dir,
        output_root=trackeval_dir,
    )

    print("Running TrackEval")
    raw_results, raw_messages = run_trackeval(
        bundle=bundle,
        output_root=trackeval_dir / "reports",
        trackeval_root=args.trackeval_root,
        eval_iou_threshold=args.eval_iou_threshold,
        classes_to_eval=args.classes_to_eval,
    )
    summary = summarise_trackeval_results(
        results=raw_results,
        tracker_name=tracker_name,
        eval_iou_threshold=args.eval_iou_threshold,
        trackeval_root=args.trackeval_root,
        classes_to_eval=args.classes_to_eval,
    )
    (output_root / "trackeval_messages.json").write_text(
        json.dumps(raw_messages, indent=2), encoding="utf-8"
    )
    summary_json = output_root / "metrics_summary.json"
    summary_csv = output_root / "metrics_summary.csv"
    write_summary_json(summary, summary_json)
    write_summary_csv(summary, summary_csv)

    print("Per-sequence results:")
    for sequence, sequence_summary in summary["per_sequence"].items():
        metrics = sequence_summary["metrics"]
        print(
            f"  {sequence}: HOTA={metrics['HOTA']:.4f} MOTA={metrics['MOTA']:.4f} "
            f"IDF1={metrics['IDF1']:.4f} IDS={metrics['IDS']} FP={metrics['FP']} FN={metrics['FN']}"
        )
    aggregate = summary["aggregate"]
    print(
        f"  COMBINED: HOTA={aggregate['HOTA']:.4f} MOTA={aggregate['MOTA']:.4f} "
        f"IDF1={aggregate['IDF1']:.4f} IDS={aggregate['IDS']} "
        f"FP={aggregate['FP']} FN={aggregate['FN']}"
    )
    print(f"Saved summary JSON to {summary_json}")
    print(f"Saved summary CSV to {summary_csv}")
    if args.max_frames:
        print(
            "WARNING: --max-frames was used, so TrackEval quality metrics are only "
            "a format smoke test."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
