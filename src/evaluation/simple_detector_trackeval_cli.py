#!/usr/bin/env python3
"""Export detections from the lightweight detector, track them, and run TrackEval."""

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
    representation_components,
)
from src.evaluation.detection_export import (
    DetectionRecord,
    load_event_file,
    load_image_timestamps,
    read_events,
)
from src.evaluation.simple_tracker import track_detections
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

    all_detections: list[DetectionRecord] = []
    frames_payload = [
        {"frame_index": frame_index, "timestamp": timestamp}
        for frame_index, timestamp in frame_entries
    ]
    handle, x, y, p, t, ms_to_idx, t_offset, _ = load_event_file(events_h5)
    started = time.perf_counter()
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
            tensor = normalise_event_tensor(tensor)
            with torch.inference_mode():
                outputs = model(tensor)
                detections = decode_dense_detections(
                    outputs=outputs,
                    frame_index=frame_index,
                    timestamp=timestamp_us,
                    score_threshold=score_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                    max_detections=max_detections,
                    feature_stride=model.config.feature_stride,
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
    parser.add_argument("--tracker-name", default="simple_dense_detector_iou")
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
        )

        track_path = tracks_dir / f"{sequence}.txt"
        print(f"Tracking {sequence} -> {track_path}")
        tracker_runs[sequence] = track_detections(
            detection_export_path=detection_path,
            output_path=track_path,
            iou_threshold=args.track_iou_threshold,
            max_missed_frames=args.track_max_missed_frames,
            min_hits=args.track_min_hits,
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
        tracker_name=args.tracker_name,
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
        tracker_name=args.tracker_name,
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
