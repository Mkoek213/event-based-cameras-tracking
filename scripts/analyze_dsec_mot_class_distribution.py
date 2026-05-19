#!/usr/bin/env python3
"""Report DSEC-MOT class distribution from annotation files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.detection_export import CLASS_NAMES, load_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/datasets/dsec_mot"),
        help="Path to the local DSEC-MOT root.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the summary as JSON.",
    )
    return parser.parse_args()


def percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * count / total


def analyse_annotations(root: Path) -> dict:
    splits = ("train", "test")

    total_annotation_counts: Counter[int] = Counter()
    total_track_sets: dict[int, set[tuple[str, str, int]]] = defaultdict(set)
    split_summaries: dict[str, dict] = {}

    for split in splits:
        ann_dir = root / "annotations" / split
        sequence_files = sorted(path for path in ann_dir.glob("*.txt") if path.is_file())
        split_annotation_counts: Counter[int] = Counter()
        split_track_sets: dict[int, set[tuple[str, int]]] = defaultdict(set)
        sequence_summaries: dict[str, dict] = {}

        for ann_path in sequence_files:
            sequence = ann_path.stem
            annotations = load_annotations(ann_path)
            seq_annotation_counts: Counter[int] = Counter()
            seq_track_sets: dict[int, set[int]] = defaultdict(set)

            for ann in annotations:
                seq_annotation_counts[ann.class_id] += 1
                seq_track_sets[ann.class_id].add(ann.track_id)

                split_annotation_counts[ann.class_id] += 1
                split_track_sets[ann.class_id].add((sequence, ann.track_id))

                total_annotation_counts[ann.class_id] += 1
                total_track_sets[ann.class_id].add((split, sequence, ann.track_id))

            sequence_summaries[sequence] = {
                "annotation_total": len(annotations),
                "track_total": len({ann.track_id for ann in annotations}),
                "classes": {
                    CLASS_NAMES[class_id]: {
                        "annotation_count": seq_annotation_counts[class_id],
                        "track_count": len(seq_track_sets[class_id]),
                    }
                    for class_id in sorted(CLASS_NAMES)
                },
            }

        split_annotation_total = sum(split_annotation_counts.values())
        split_track_total = sum(len(track_ids) for track_ids in split_track_sets.values())
        split_summaries[split] = {
            "annotation_total": split_annotation_total,
            "track_total": split_track_total,
            "classes": {
                CLASS_NAMES[class_id]: {
                    "annotation_count": split_annotation_counts[class_id],
                    "annotation_percent": percent(split_annotation_counts[class_id], split_annotation_total),
                    "track_count": len(split_track_sets[class_id]),
                    "track_percent": percent(len(split_track_sets[class_id]), split_track_total),
                }
                for class_id in sorted(CLASS_NAMES)
            },
            "sequences": sequence_summaries,
        }

    total_annotation_total = sum(total_annotation_counts.values())
    total_track_total = sum(len(track_ids) for track_ids in total_track_sets.values())
    return {
        "dataset_root": str(root),
        "annotation_total": total_annotation_total,
        "track_total": total_track_total,
        "classes": {
            CLASS_NAMES[class_id]: {
                "annotation_count": total_annotation_counts[class_id],
                "annotation_percent": percent(total_annotation_counts[class_id], total_annotation_total),
                "track_count": len(total_track_sets[class_id]),
                "track_percent": percent(len(total_track_sets[class_id]), total_track_total),
            }
            for class_id in sorted(CLASS_NAMES)
        },
        "splits": split_summaries,
    }


def print_summary(summary: dict) -> None:
    print(f"Dataset root: {summary['dataset_root']}")
    print(f"Total annotations: {summary['annotation_total']}")
    print(f"Total unique tracks: {summary['track_total']}")
    print()
    print("Overall class distribution:")
    for class_name, stats in summary["classes"].items():
        print(
            f"  {class_name:10s}  "
            f"annotations={stats['annotation_count']:6d} ({stats['annotation_percent']:6.2f}%)  "
            f"tracks={stats['track_count']:4d} ({stats['track_percent']:6.2f}%)"
        )

    for split_name, split_summary in summary["splits"].items():
        print()
        print(f"{split_name.capitalize()} split:")
        print(f"  annotations={split_summary['annotation_total']} tracks={split_summary['track_total']}")
        for class_name, stats in split_summary["classes"].items():
            print(
                f"    {class_name:10s}  "
                f"annotations={stats['annotation_count']:6d} ({stats['annotation_percent']:6.2f}%)  "
                f"tracks={stats['track_count']:4d} ({stats['track_percent']:6.2f}%)"
            )


def main() -> int:
    args = parse_args()
    summary = analyse_annotations(args.root)
    print_summary(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print()
        print(f"Saved JSON summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
