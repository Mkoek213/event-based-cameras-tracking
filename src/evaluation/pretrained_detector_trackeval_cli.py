#!/usr/bin/env python3
"""Export, track, and evaluate an RGB-pretrained event detector on DSEC-MOT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.pretrained_detection_dataset import DSECPretrainedDetectionDataset
from src.evaluation.detection_export import DetectionRecord
from src.evaluation.simple_tracker import track_detections
from src.evaluation.trackeval_adapter import (
    export_trackeval_bundle,
    run_trackeval,
    summarise_trackeval_results,
    write_summary_csv,
    write_summary_json,
)
from src.models.pretrained_detector import PretrainedDetectorConfig, PretrainedEventDetector


def load_model(path: Path, device: torch.device) -> tuple[PretrainedEventDetector, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config_dict = dict(checkpoint["model_config"])
    config_dict["pretrained_weights"] = False
    model = PretrainedEventDetector(PretrainedDetectorConfig(**config_dict)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def export_sequence(
    model: PretrainedEventDetector,
    checkpoint: dict,
    root: Path,
    split: str,
    sequence: str,
    output_path: Path,
    score_threshold: float,
    max_detections: int,
    device: torch.device,
    start_frame: int,
    max_frames: int,
) -> dict:
    config = checkpoint["benchmark_config"]
    dataset = DSECPretrainedDetectionDataset(
        root=root,
        split=split,
        sequences=[sequence],
        representation=config["representation"],
        num_bins=int(config["num_bins"]),
        time_window_us=int(config["time_window_us"]),
        include_unannotated=True,
        eros_cache_root=config["eros_cache_root"],
    )
    indices = list(range(len(dataset)))[start_frame:]
    if max_frames:
        indices = indices[:max_frames]
    records: list[DetectionRecord] = []
    frames: list[dict] = []
    started = time.perf_counter()
    for position, index in enumerate(indices, start=1):
        image, _, metadata = dataset[index]
        with torch.inference_mode():
            prediction = model([image.to(device)])[0]
        keep = prediction["scores"] >= score_threshold
        boxes = prediction["boxes"][keep][:max_detections].cpu()
        labels = prediction["labels"][keep][:max_detections].cpu()
        scores = prediction["scores"][keep][:max_detections].cpu()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()
            records.append(
                DetectionRecord(
                    frame_index=metadata["frame_index"],
                    timestamp=metadata["timestamp"],
                    class_id=int(label) - 1,
                    score=float(score),
                    bbox_left=x1,
                    bbox_top=y1,
                    bbox_width=x2 - x1,
                    bbox_height=y2 - y1,
                )
            )
        frames.append({"frame_index": metadata["frame_index"], "timestamp": metadata["timestamp"]})
        if position == 1 or position % 100 == 0 or position == len(indices):
            print(f"[{sequence}] {position}/{len(indices)} detections={len(boxes)}", flush=True)
    elapsed_s = time.perf_counter() - started
    payload = {
        "split": split,
        "sequence": sequence,
        "frame_count": len(indices),
        "score_threshold": score_threshold,
        "max_detections": max_detections,
        "representation": config["representation"],
        "adapter_mode": config["adapter_mode"],
        "elapsed_s": elapsed_s,
        "fps": len(indices) / elapsed_s if elapsed_s else 0.0,
        "frames": frames,
        "detections": [record.to_dict() for record in records],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--track-iou-threshold", type=float, default=0.5)
    parser.add_argument("--track-max-missed-frames", type=int, default=2)
    parser.add_argument("--track-min-hits", type=int, default=1)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tracker-name", default="pretrained_fasterrcnn_iou")
    parser.add_argument("--trackeval-root", type=Path, default=Path("external/TrackEval"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/dsec_mot_trackeval_pretrained_detector")
    )
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    output_root = args.output_root / args.run_name
    detections_dir, tracks_dir = output_root / "detections", output_root / "tracks"
    detection_runs, tracker_runs = {}, {}
    for sequence in args.sequences:
        detection_path = detections_dir / f"{sequence}.json"
        detection_runs[sequence] = export_sequence(
            model,
            checkpoint,
            args.root,
            args.split,
            sequence,
            detection_path,
            args.score_threshold,
            args.max_detections,
            device,
            args.start_frame,
            args.max_frames,
        )
        track_path = tracks_dir / f"{sequence}.txt"
        tracker_runs[sequence] = track_detections(
            detection_path,
            track_path,
            args.track_iou_threshold,
            args.track_max_missed_frames,
            args.track_min_hits,
        )
    (output_root / "detection_runs.json").write_text(
        json.dumps(detection_runs, indent=2), encoding="utf-8"
    )
    (output_root / "tracker_runs.json").write_text(
        json.dumps(tracker_runs, indent=2), encoding="utf-8"
    )

    bundle = export_trackeval_bundle(
        args.root,
        args.split,
        args.sequences,
        args.tracker_name,
        tracks_dir,
        output_root / "trackeval",
    )
    results, messages = run_trackeval(
        bundle, output_root / "trackeval" / "reports", args.trackeval_root, args.eval_iou_threshold
    )
    summary = summarise_trackeval_results(
        results, args.tracker_name, args.eval_iou_threshold, args.trackeval_root
    )
    (output_root / "trackeval_messages.json").write_text(
        json.dumps(messages, indent=2), encoding="utf-8"
    )
    write_summary_json(summary, output_root / "metrics_summary.json")
    write_summary_csv(summary, output_root / "metrics_summary.csv")
    aggregate = summary["aggregate"]
    print(
        f"COMBINED: HOTA={aggregate['HOTA']:.4f} MOTA={aggregate['MOTA']:.4f} "
        f"IDF1={aggregate['IDF1']:.4f} IDS={aggregate['IDS']} "
        f"FP={aggregate['FP']} FN={aggregate['FN']}"
    )
    if args.max_frames:
        print("WARNING: partial sequence metrics are only a smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
