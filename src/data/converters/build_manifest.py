"""CLI for converting external event datasets into unified pretraining manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.converters.event_manifest import (
    BoxRecord,
    RepresentationConfig,
    csv_summary,
    group_boxes_by_timestamp,
    inspect_h5,
    inspect_numpy,
    iter_sampled_timestamps,
    load_structured_boxes,
    make_manifest_row,
    parse_class_id_map,
    print_json,
    read_h5_event_file,
    read_metavision_dat,
    save_dense_representations,
    select_event_window,
    write_train_val_manifests,
)
from src.data.unified_manifest import read_jsonl, write_jsonl


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--time-window-us", type=int, default=50_000)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)


def _representation_config(args: argparse.Namespace) -> RepresentationConfig:
    return RepresentationConfig(
        width=args.width,
        height=args.height,
        num_bins=args.num_bins,
        time_window_us=args.time_window_us,
        dtype=args.dtype,
    )


def _row_output_prefix(output_root: Path, dataset: str, sequence: str, frame_index: int) -> Path:
    return output_root / "cache" / dataset / sequence / f"{frame_index:08d}"


def _convert_one_event_file(
    *,
    dataset_name: str,
    sequence: str,
    events,
    boxes: list[BoxRecord],
    output_root: Path,
    config: RepresentationConfig,
    max_samples: int,
    sample_stride: int,
) -> list[dict]:
    grouped = group_boxes_by_timestamp(boxes)
    rows: list[dict] = []
    for frame_index, (timestamp_us, timestamp_boxes) in enumerate(
        iter_sampled_timestamps(grouped, max_samples=max_samples, sample_stride=sample_stride)
    ):
        window = select_event_window(
            events,
            end_timestamp_us=timestamp_us,
            time_window_us=config.time_window_us,
        )
        if len(window) == 0:
            continue
        paths = save_dense_representations(
            window,
            _row_output_prefix(output_root, dataset_name, sequence, frame_index),
            config,
        )
        rows.append(
            make_manifest_row(
                dataset=dataset_name,
                sequence=sequence,
                timestamp_us=timestamp_us,
                frame_index=frame_index,
                width=config.width,
                height=config.height,
                representation_paths=paths,
                boxes=timestamp_boxes,
            )
        )
    return rows


def command_inspect(args: argparse.Namespace) -> int:
    for path in args.paths:
        suffix = path.suffix.lower()
        if suffix in {".npy", ".npz"}:
            print_json(inspect_numpy(path))
        elif suffix in {".h5", ".hdf5"}:
            print_json(inspect_h5(path))
        elif suffix == ".dat":
            events = read_metavision_dat(path, max_events=args.max_events)
            print_json(
                {
                    "path": str(path),
                    "events": int(len(events)),
                    "dtype": str(events.dtype),
                    "first": repr(events[: min(len(events), 3)]),
                    "last_timestamp_us": int(events["t"][-1]) if len(events) else None,
                }
            )
        else:
            print_json({"path": str(path), "error": f"Unsupported suffix {suffix}"})
    return 0


def command_dsec_detection(args: argparse.Namespace) -> int:
    config = _representation_config(args)
    class_map = parse_class_id_map(args.class_id_map)
    rows: list[dict] = []
    manifest_rows = args.manifest.read_text(encoding="utf-8").splitlines()[1:]
    for line in manifest_rows:
        sequence, event_path, tracks_path = line.split(",", maxsplit=2)
        print(f"[dsec_detection] {sequence}", flush=True)
        events = read_h5_event_file(event_path)
        boxes = load_structured_boxes(tracks_path, class_id_map=class_map)
        rows.extend(
            _convert_one_event_file(
                dataset_name="dsec_detection",
                sequence=sequence,
                events=events,
                boxes=boxes,
                output_root=args.output_root,
                config=config,
                max_samples=args.max_samples,
                sample_stride=args.sample_stride,
            )
        )
    train_path, val_path = write_train_val_manifests(
        args.output_root,
        rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    csv_summary(args.output_root / "manifests" / "dsec_detection_summary.csv", rows)
    print(f"Wrote {len(rows)} rows")
    print(f"train={train_path}")
    print(f"val={val_path}")
    return 0


def command_etram(args: argparse.Namespace) -> int:
    config = _representation_config(args)
    class_map = parse_class_id_map(args.class_id_map)
    rows: list[dict] = []
    h5_files = sorted(args.hdf5_root.glob("*_td.h5"))
    for h5_path in h5_files:
        sequence = h5_path.name.removesuffix("_td.h5")
        bbox_path = args.bbox_root / f"{sequence}_bbox.npy"
        if not bbox_path.exists():
            print(f"[etram] skipping {sequence}: missing {bbox_path}", flush=True)
            continue
        print(f"[etram] {sequence}", flush=True)
        events = read_h5_event_file(h5_path)
        boxes = load_structured_boxes(bbox_path, class_id_map=class_map)
        rows.extend(
            _convert_one_event_file(
                dataset_name="etram",
                sequence=sequence,
                events=events,
                boxes=boxes,
                output_root=args.output_root,
                config=config,
                max_samples=args.max_samples,
                sample_stride=args.sample_stride,
            )
        )
    train_path, val_path = write_train_val_manifests(
        args.output_root,
        rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    csv_summary(args.output_root / "manifests" / "etram_summary.csv", rows)
    print(f"Wrote {len(rows)} rows")
    print(f"train={train_path}")
    print(f"val={val_path}")
    return 0


def command_prophesee(args: argparse.Namespace) -> int:
    config = _representation_config(args)
    class_map = parse_class_id_map(args.class_id_map)
    rows: list[dict] = []
    event_files = sorted(args.root.rglob("*_td.dat"))
    for event_path in event_files:
        sequence = event_path.name.removesuffix("_td.dat")
        bbox_path = event_path.with_name(sequence + "_bbox.npy")
        if not bbox_path.exists():
            alt = list(event_path.parent.glob("*bbox*.npy"))
            bbox_path = alt[0] if len(alt) == 1 else bbox_path
        if not bbox_path.exists():
            print(f"[prophesee_1mp] skipping {sequence}: missing bbox npy", flush=True)
            continue
        print(f"[prophesee_1mp] {sequence}", flush=True)
        events = read_metavision_dat(event_path)
        boxes = load_structured_boxes(bbox_path, class_id_map=class_map)
        rows.extend(
            _convert_one_event_file(
                dataset_name="prophesee_1mp",
                sequence=sequence,
                events=events,
                boxes=boxes,
                output_root=args.output_root,
                config=config,
                max_samples=args.max_samples,
                sample_stride=args.sample_stride,
            )
        )
    train_path, val_path = write_train_val_manifests(
        args.output_root,
        rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    csv_summary(args.output_root / "manifests" / "prophesee_1mp_summary.csv", rows)
    print(f"Wrote {len(rows)} rows")
    print(f"train={train_path}")
    print(f"val={val_path}")
    return 0


def command_merge(args: argparse.Namespace) -> int:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for manifest_dir in args.manifest_dirs:
        train_path = manifest_dir / "manifests" / "pretrain_train.jsonl"
        val_path = manifest_dir / "manifests" / "pretrain_val.jsonl"
        if train_path.exists():
            train_rows.extend(read_jsonl(train_path))
        else:
            print(f"[merge] missing {train_path}", flush=True)
        if val_path.exists():
            val_rows.extend(read_jsonl(val_path))
        else:
            print(f"[merge] missing {val_path}", flush=True)
    output_manifest_dir = args.output_root / "manifests"
    train_output = output_manifest_dir / "pretrain_train.jsonl"
    val_output = output_manifest_dir / "pretrain_val.jsonl"
    write_jsonl(train_output, train_rows)
    write_jsonl(val_output, val_rows)
    csv_summary(output_manifest_dir / "merged_summary.csv", train_rows + val_rows)
    print(f"Wrote train={len(train_rows)} rows to {train_output}")
    print(f"Wrote val={len(val_rows)} rows to {val_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("paths", type=Path, nargs="+")
    inspect_parser.add_argument("--max-events", type=int, default=5)
    inspect_parser.set_defaults(func=command_inspect)

    dsec_parser = subparsers.add_parser("dsec-detection")
    _common_parser(dsec_parser)
    dsec_parser.add_argument("--manifest", type=Path, required=True)
    dsec_parser.add_argument(
        "--class-id-map",
        default="0:pedestrian,1:two_wheeler,2:vehicle,3:vehicle,4:vehicle,5:two_wheeler,6:two_wheeler",
    )
    dsec_parser.set_defaults(func=command_dsec_detection)

    etram_parser = subparsers.add_parser("etram")
    _common_parser(etram_parser)
    etram_parser.add_argument("--hdf5-root", type=Path, required=True)
    etram_parser.add_argument("--bbox-root", type=Path, required=True)
    etram_parser.add_argument("--class-id-map", default=None)
    etram_parser.set_defaults(func=command_etram)

    prophesee_parser = subparsers.add_parser("prophesee-1mp")
    _common_parser(prophesee_parser)
    prophesee_parser.add_argument("--root", type=Path, required=True)
    prophesee_parser.add_argument(
        "--class-id-map",
        default="0:pedestrian,1:two_wheeler,2:vehicle",
    )
    prophesee_parser.set_defaults(func=command_prophesee)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--output-root", type=Path, required=True)
    merge_parser.add_argument("manifest_dirs", type=Path, nargs="+")
    merge_parser.set_defaults(func=command_merge)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
