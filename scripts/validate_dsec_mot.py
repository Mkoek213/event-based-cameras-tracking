#!/usr/bin/env python3
"""Validate the local DSEC-MOT dataset layout and basic data consistency."""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


EXPECTED_SPLITS = {
    "train": [
        "interlaken_00_a",
        "interlaken_00_b",
        "interlaken_00_c",
        "interlaken_00_e",
        "zurich_city_01_d",
        "zurich_city_01_e",
        "zurich_city_04_b",
        "zurich_city_09_c",
        "zurich_city_09_d",
        "zurich_city_14_b",
    ],
    "test": [
        "interlaken_00_d",
        "zurich_city_00_b",
    ],
}

EXPECTED_CLASSES = set(range(7))
EVENT_WIDTH = 640
EVENT_HEIGHT = 480
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


@dataclass
class SequenceStats:
    split: str
    sequence: str
    images: int
    annotations: int
    tracks: int
    classes: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/datasets/dsec_mot"),
        help="Path to the DSEC-MOT dataset root.",
    )
    parser.add_argument(
        "--verify-all-pngs",
        action="store_true",
        help="Verify every PNG file instead of sampling first/middle/last.",
    )
    parser.add_argument(
        "--check-zip-crc",
        action="store_true",
        help="Run CRC checks on all zip archives (slower).",
    )
    return parser.parse_args()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def read_hdf5_signature(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE


def load_image_timestamps(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_exposure_timestamps(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        start, end = [part.strip() for part in line.split(",", maxsplit=1)]
        rows.append((int(start), int(end)))
    return rows


def load_annotations(path: Path) -> list[tuple[int, int, float, float, float, float, int]]:
    rows: list[tuple[int, int, float, float, float, float, int]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for index, raw_row in enumerate(reader, start=1):
            if not raw_row:
                continue
            if len(raw_row) != 7:
                raise ValueError(f"{path}:{index}: expected 7 columns, got {len(raw_row)}")
            timestamp = int(raw_row[0].strip())
            track_id = int(raw_row[1].strip())
            left = float(raw_row[2].strip())
            top = float(raw_row[3].strip())
            width = float(raw_row[4].strip())
            height = float(raw_row[5].strip())
            class_id = int(raw_row[6].strip())
            row = (timestamp, track_id, left, top, width, height, class_id)
            rows.append(row)
    return rows


def strictly_increasing(values: list[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def nondecreasing(values: list[int]) -> bool:
    return all(left <= right for left, right in zip(values, values[1:]))


def verify_png(path: Path) -> None:
    # `verify()` alone may miss truncated images; reopen and fully decode too.
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()


def png_paths_to_check(image_dir: Path, verify_all: bool) -> list[Path]:
    pngs = sorted(image_dir.glob("*.png"))
    if verify_all or len(pngs) <= 3:
        return pngs
    indices = sorted({0, len(pngs) // 2, len(pngs) - 1})
    return [pngs[index] for index in indices]


def check_zip_members(path: Path, expected_members: set[str] | None, errors: list[str], crc: bool) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            if expected_members is not None and members != expected_members:
                fail(
                    errors,
                    f"{path}: unexpected zip members {sorted(members)}; expected {sorted(expected_members)}",
                )
            if expected_members is None and not members:
                fail(errors, f"{path}: zip archive is empty")
            if crc:
                broken_member = archive.testzip()
                if broken_member is not None:
                    fail(errors, f"{path}: CRC check failed for {broken_member}")
    except zipfile.BadZipFile:
        fail(errors, f"{path}: invalid zip archive")


def validate_sequence(root: Path, split: str, sequence: str, verify_all_pngs: bool, check_zip_crc: bool) -> tuple[list[str], SequenceStats | None]:
    errors: list[str] = []
    seq_dir = root / split / sequence

    raw_zip_paths = {
        "events_zip": seq_dir / f"{sequence}_events_left.zip",
        "images_zip": seq_dir / f"{sequence}_images_rectified_left.zip",
        "calibration_zip": seq_dir / f"{sequence}_calibration.zip",
    }
    text_paths = {
        "image_timestamps": seq_dir / f"{sequence}_image_timestamps.txt",
        "image_exposure_timestamps_left": seq_dir / f"{sequence}_image_exposure_timestamps_left.txt",
        "annotation_txt": root / "annotations" / split / f"{sequence}.txt",
    }
    extracted_paths = {
        "events_h5": seq_dir / "events_left" / "events.h5",
        "rectify_map_h5": seq_dir / "events_left" / "rectify_map.h5",
        "cam_to_cam_yaml": seq_dir / "calibration" / "cam_to_cam.yaml",
        "cam_to_lidar_yaml": seq_dir / "calibration" / "cam_to_lidar.yaml",
        "images_dir": seq_dir / "images_rectified_left",
    }

    for label, path in {**raw_zip_paths, **text_paths, **extracted_paths}.items():
        if not path.exists():
            fail(errors, f"{seq_dir}: missing {label} at {path.name}")

    if errors:
        return errors, None

    check_zip_members(raw_zip_paths["events_zip"], {"events.h5", "rectify_map.h5"}, errors, check_zip_crc)
    check_zip_members(raw_zip_paths["calibration_zip"], {"cam_to_cam.yaml", "cam_to_lidar.yaml"}, errors, check_zip_crc)
    check_zip_members(raw_zip_paths["images_zip"], None, errors, check_zip_crc)

    for h5_key in ("events_h5", "rectify_map_h5"):
        if not read_hdf5_signature(extracted_paths[h5_key]):
            fail(errors, f"{extracted_paths[h5_key]}: invalid HDF5 signature")
        if extracted_paths[h5_key].stat().st_size <= len(HDF5_SIGNATURE):
            fail(errors, f"{extracted_paths[h5_key]}: file is unexpectedly small")

    for yaml_key in ("cam_to_cam_yaml", "cam_to_lidar_yaml"):
        if extracted_paths[yaml_key].stat().st_size == 0:
            fail(errors, f"{extracted_paths[yaml_key]}: empty file")

    image_timestamps = load_image_timestamps(text_paths["image_timestamps"])
    if not image_timestamps:
        fail(errors, f"{text_paths['image_timestamps']}: no timestamps")
    elif not strictly_increasing(image_timestamps):
        fail(errors, f"{text_paths['image_timestamps']}: timestamps are not strictly increasing")

    exposure_timestamps = load_exposure_timestamps(text_paths["image_exposure_timestamps_left"])
    if len(exposure_timestamps) != len(image_timestamps):
        fail(
            errors,
            f"{text_paths['image_exposure_timestamps_left']}: expected {len(image_timestamps)} rows, got {len(exposure_timestamps)}",
        )
    else:
        exposure_starts = [start for start, _ in exposure_timestamps]
        exposure_ends = [end for _, end in exposure_timestamps]
        if not strictly_increasing(exposure_starts):
            fail(errors, f"{text_paths['image_exposure_timestamps_left']}: exposure start timestamps are not strictly increasing")
        if not strictly_increasing(exposure_ends):
            fail(errors, f"{text_paths['image_exposure_timestamps_left']}: exposure end timestamps are not strictly increasing")
        for image_ts, (start, end) in zip(image_timestamps, exposure_timestamps):
            if start > end:
                fail(errors, f"{text_paths['image_exposure_timestamps_left']}: exposure start is after end")
                break
            if not (start <= image_ts <= end):
                fail(
                    errors,
                    f"{text_paths['image_exposure_timestamps_left']}: image timestamp {image_ts} outside exposure window [{start}, {end}]",
                )
                break

    image_dir = extracted_paths["images_dir"]
    png_paths = sorted(image_dir.glob("*.png"))
    if not png_paths:
        fail(errors, f"{image_dir}: no PNG files found")
    elif len(png_paths) != len(image_timestamps):
        fail(errors, f"{image_dir}: expected {len(image_timestamps)} PNG files, got {len(png_paths)}")

    for png_path in png_paths_to_check(image_dir, verify_all_pngs):
        try:
            verify_png(png_path)
        except Exception as exc:  # noqa: BLE001
            fail(errors, f"{png_path}: PNG verification failed: {exc}")

    try:
        annotations = load_annotations(text_paths["annotation_txt"])
    except ValueError as exc:
        fail(errors, str(exc))
        return errors, None

    if not annotations:
        fail(errors, f"{text_paths['annotation_txt']}: no annotations")
        return errors, None

    image_timestamp_set = set(image_timestamps)
    track_ids: set[int] = set()
    classes: set[int] = set()
    track_last_timestamp: dict[int, int] = {}
    for line_number, (timestamp, track_id, left, top, width, height, class_id) in enumerate(annotations, start=1):
        if timestamp not in image_timestamp_set:
            fail(errors, f"{text_paths['annotation_txt']}:{line_number}: timestamp {timestamp} not present in image timestamps")
            break
        previous_timestamp = track_last_timestamp.get(track_id)
        if previous_timestamp is not None and timestamp < previous_timestamp:
            fail(
                errors,
                f"{text_paths['annotation_txt']}:{line_number}: track {track_id} timestamp moved backwards "
                f"from {previous_timestamp} to {timestamp}",
            )
            break
        if track_id < 0:
            fail(errors, f"{text_paths['annotation_txt']}:{line_number}: negative track_id {track_id}")
            break
        if width <= 0 or height <= 0:
            fail(errors, f"{text_paths['annotation_txt']}:{line_number}: non-positive bbox size ({width}, {height})")
            break
        if left < 0 or top < 0:
            fail(errors, f"{text_paths['annotation_txt']}:{line_number}: negative bbox origin ({left}, {top})")
            break
        if left >= EVENT_WIDTH or top >= EVENT_HEIGHT or left + width <= 0 or top + height <= 0:
            fail(
                errors,
                f"{text_paths['annotation_txt']}:{line_number}: bbox exceeds event frame "
                f"{EVENT_WIDTH}x{EVENT_HEIGHT}: ({left}, {top}, {width}, {height})",
            )
            break
        if class_id not in EXPECTED_CLASSES:
            fail(errors, f"{text_paths['annotation_txt']}:{line_number}: invalid class_id {class_id}")
            break
        track_ids.add(track_id)
        classes.add(class_id)
        track_last_timestamp[track_id] = timestamp

    if errors:
        return errors, None

    return (
        errors,
        SequenceStats(
            split=split,
            sequence=sequence,
            images=len(png_paths),
            annotations=len(annotations),
            tracks=len(track_ids),
            classes=sorted(classes),
        ),
    )


def validate_annotation_archives(root: Path, errors: list[str], check_zip_crc: bool) -> None:
    for split, sequences in EXPECTED_SPLITS.items():
        archive_dir = root / "annotations" / split
        zip_paths = sorted(archive_dir.glob("*.zip"))
        if not zip_paths:
            fail(errors, f"{archive_dir}: no annotation zip found")
            continue
        expected_members = {f"{split}/{sequence}.txt" for sequence in sequences}
        if len(zip_paths) != 1:
            fail(errors, f"{archive_dir}: expected exactly one annotation zip, found {len(zip_paths)}")
        for zip_path in zip_paths:
            check_zip_members(zip_path, expected_members, errors, check_zip_crc)


def main() -> int:
    args = parse_args()
    root = args.root
    errors: list[str] = []
    stats: list[SequenceStats] = []

    if not root.exists():
        print(f"Dataset root does not exist: {root}", file=sys.stderr)
        return 1

    validate_annotation_archives(root, errors, args.check_zip_crc)

    for split, sequences in EXPECTED_SPLITS.items():
        for sequence in sequences:
            seq_errors, seq_stats = validate_sequence(
                root=root,
                split=split,
                sequence=sequence,
                verify_all_pngs=args.verify_all_pngs,
                check_zip_crc=args.check_zip_crc,
            )
            errors.extend(seq_errors)
            if seq_stats is not None:
                stats.append(seq_stats)

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation successful.")
    total_images = sum(item.images for item in stats)
    total_annotations = sum(item.annotations for item in stats)
    total_tracks = sum(item.tracks for item in stats)
    print(f"Sequences: {len(stats)}")
    print(f"Images: {total_images}")
    print(f"Annotations: {total_annotations}")
    print(f"Track ids: {total_tracks}")
    for item in stats:
        classes = ",".join(str(class_id) for class_id in item.classes)
        print(
            f"[{item.split}] {item.sequence}: "
            f"images={item.images}, annotations={item.annotations}, tracks={item.tracks}, classes={classes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
