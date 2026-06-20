"""Tests for detection-only metrics."""

import json

from src.evaluation.detection_metrics import evaluate_detection_export


def test_detection_metrics_counts_tp_fp_fn(tmp_path):
    root = tmp_path / "dataset"
    ann_dir = root / "annotations" / "train"
    ann_dir.mkdir(parents=True)
    (ann_dir / "seq.txt").write_text(
        "100,1,10,10,20,20,0\n100,2,100,100,20,20,0\n",
        encoding="utf-8",
    )
    detection_path = tmp_path / "detections.json"
    detection_path.write_text(
        json.dumps(
            {
                "dataset_root": str(root),
                "split": "train",
                "sequence": "seq",
                "score_threshold": 0.5,
                "frames": [{"frame_index": 0, "timestamp": 100}],
                "detections": [
                    {
                        "frame_index": 0,
                        "timestamp": 100,
                        "class_id": 0,
                        "score": 0.9,
                        "bbox_left": 10,
                        "bbox_top": 10,
                        "bbox_width": 20,
                        "bbox_height": 20,
                    },
                    {
                        "frame_index": 0,
                        "timestamp": 100,
                        "class_id": 0,
                        "score": 0.8,
                        "bbox_left": 200,
                        "bbox_top": 200,
                        "bbox_width": 20,
                        "bbox_height": 20,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_detection_export(detection_path)

    assert summary["aggregate"]["tp"] == 1
    assert summary["aggregate"]["fp"] == 1
    assert summary["aggregate"]["fn"] == 1
    assert summary["aggregate"]["precision"] == 0.5
    assert summary["aggregate"]["recall"] == 0.5
