"""TrackEval adapter and file export helpers for DSEC-MOT."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.evaluation.detection_export import load_annotations, load_image_timestamps

TRACKEVAL_CLASS_ID_TO_NAME = {
    1: "car",
    2: "pedestrian",
    3: "bicycle",
    4: "motorcycle",
    5: "bus",
    6: "truck",
    7: "train",
}
TRACKEVAL_CLASS_NAMES = list(TRACKEVAL_CLASS_ID_TO_NAME.values())


def _ensure_numpy_compat() -> None:
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(key): _to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_trackeval_importable(trackeval_root: Path | None = None):
    _ensure_numpy_compat()
    root = trackeval_root or Path("external/TrackEval")
    if not root.exists():
        raise FileNotFoundError(
            f"TrackEval checkout not found at {root}. Clone "
            "https://github.com/JonathonLuiten/TrackEval into external/TrackEval first."
        )
    root_resolved = str(root.resolve())
    if root_resolved not in sys.path:
        sys.path.insert(0, root_resolved)
    import trackeval  # type: ignore

    return trackeval


class DSECMOTTrackEvalDataset:
    @staticmethod
    def get_default_dataset_config():
        return {
            "GT_FOLDER": None,
            "TRACKERS_FOLDER": None,
            "OUTPUT_FOLDER": None,
            "TRACKERS_TO_EVAL": None,
            "CLASSES_TO_EVAL": TRACKEVAL_CLASS_NAMES,
            "BENCHMARK": "dsec_mot",
            "SPLIT_TO_EVAL": "test",
            "PRINT_CONFIG": True,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "TRACKER_DISPLAY_NAMES": None,
            "SEQMAP_FILE": None,
            "SKIP_SPLIT_FOL": False,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        }

    def __init__(self, config=None):
        trackeval = ensure_trackeval_importable()
        from trackeval.datasets._base_dataset import _BaseDataset  # type: ignore

        self._base = _BaseDataset
        super_class = _BaseDataset
        super_class.__init__(self)

        self.config = trackeval.utils.init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )
        self.should_classes_combine = True
        self.use_super_categories = False

        gt_set = f"{self.config['BENCHMARK']}-{self.config['SPLIT_TO_EVAL']}"
        split_folder = "" if self.config["SKIP_SPLIT_FOL"] else gt_set
        self.gt_fol = os.path.join(self.config["GT_FOLDER"], split_folder)
        self.tracker_fol = os.path.join(self.config["TRACKERS_FOLDER"], split_folder)
        self.output_fol = self.config["OUTPUT_FOLDER"] or self.tracker_fol
        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        self.valid_classes = TRACKEVAL_CLASS_NAMES
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config["CLASSES_TO_EVAL"]
        ]
        if not all(self.class_list):
            raise trackeval.utils.TrackEvalException(
                "Attempted to evaluate invalid classes. "
                f"Valid classes: {', '.join(self.valid_classes)}"
            )
        self.class_name_to_class_id = {
            name: class_id for class_id, name in TRACKEVAL_CLASS_ID_TO_NAME.items()
        }

        self.seq_list, self.seq_lengths = self._get_seq_info()
        if not self.seq_list:
            raise trackeval.utils.TrackEvalException(
                "No sequences selected for DSEC-MOT evaluation."
            )

        for seq in self.seq_list:
            gt_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
            if not os.path.isfile(gt_file):
                raise trackeval.utils.TrackEvalException(
                    f"GT file not found for sequence {seq}: {gt_file}"
                )

        if self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = sorted(os.listdir(self.tracker_fol))
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        if self.config["TRACKER_DISPLAY_NAMES"] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif len(self.config["TRACKER_DISPLAY_NAMES"]) == len(self.tracker_list):
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.config["TRACKER_DISPLAY_NAMES"])
            )
        else:
            raise trackeval.utils.TrackEvalException(
                "Tracker display names do not match trackers to eval."
            )

        for tracker in self.tracker_list:
            for seq in self.seq_list:
                tracker_file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol, f"{seq}.txt"
                )
                if not os.path.isfile(tracker_file):
                    raise trackeval.utils.TrackEvalException(
                        f"Tracker file not found: {tracker_file}"
                    )

    @classmethod
    def get_name(cls):
        return cls.__name__

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def get_output_fol(self, tracker):
        return os.path.join(self.output_fol, tracker, self.output_sub_fol)

    def get_eval_info(self):
        return self.tracker_list, self.seq_list, self.class_list

    def _get_seq_info(self):
        trackeval = ensure_trackeval_importable()
        seqmap_file = self.config["SEQMAP_FILE"]
        if seqmap_file is None:
            seqmap_file = os.path.join(
                self.config["GT_FOLDER"],
                "seqmaps",
                f"{self.config['BENCHMARK']}-{self.config['SPLIT_TO_EVAL']}.txt",
            )
        if not os.path.isfile(seqmap_file):
            raise trackeval.utils.TrackEvalException(f"Seqmap file not found: {seqmap_file}")

        seq_list: list[str] = []
        seq_lengths: dict[str, int] = {}
        with open(seqmap_file, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if index == 0 or not row or not row[0]:
                    continue
                seq = row[0].strip()
                seq_list.append(seq)
                seqinfo_path = Path(self.gt_fol) / seq / "seqinfo.ini"
                if not seqinfo_path.exists():
                    raise trackeval.utils.TrackEvalException(
                        f"seqinfo.ini not found for {seq}: {seqinfo_path}"
                    )
                seq_lengths[seq] = _read_seq_length(seqinfo_path)
        return seq_list, seq_lengths

    def _load_raw_file(self, tracker, seq, is_gt):
        trackeval = ensure_trackeval_importable()
        if is_gt:
            file_path = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
        else:
            file_path = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, f"{seq}.txt")

        read_data, _ = self._base._load_simple_text_file(
            file_path,
            time_col=0,
            id_col=1,
            remove_negative_ids=True,
        )

        num_timesteps = self.seq_lengths[seq]
        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_crowd_ignore_regions"]
        else:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        valid_time_keys = {str(t + 1) for t in range(num_timesteps)}
        extra_time_keys = [time_key for time_key in read_data if time_key not in valid_time_keys]
        if extra_time_keys:
            data_type = "Ground-truth" if is_gt else "Tracking"
            raise trackeval.utils.TrackEvalException(
                f"{data_type} data contains invalid timesteps for seq {seq}: "
                f"{', '.join(extra_time_keys)}"
            )

        for t in range(num_timesteps):
            time_key = str(t + 1)
            if time_key in read_data:
                time_data = np.asarray(read_data[time_key], dtype=np.float)
                raw_data["dets"][t] = np.atleast_2d(time_data[:, 2:6])
                raw_data["ids"][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                if time_data.shape[1] < 8:
                    raise trackeval.utils.TrackEvalException(
                        "Expected 8 columns in DSEC-MOT TrackEval file "
                        f"for seq {seq}, frame {t + 1}."
                    )
                raw_data["classes"][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                if is_gt:
                    raw_data["gt_crowd_ignore_regions"][t] = np.empty((0, 4))
                else:
                    raw_data["tracker_confidences"][t] = np.atleast_1d(time_data[:, 6])
            else:
                raw_data["dets"][t] = np.empty((0, 4))
                raw_data["ids"][t] = np.empty(0, dtype=int)
                raw_data["classes"][t] = np.empty(0, dtype=int)
                if is_gt:
                    raw_data["gt_crowd_ignore_regions"][t] = np.empty((0, 4))
                else:
                    raw_data["tracker_confidences"][t] = np.empty(0)

        key_map = {
            "ids": "gt_ids" if is_gt else "tracker_ids",
            "classes": "gt_classes" if is_gt else "tracker_classes",
            "dets": "gt_dets" if is_gt else "tracker_dets",
        }
        for old_key, new_key in key_map.items():
            raw_data[new_key] = raw_data.pop(old_key)
        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data

    def _check_unique_ids(self, data, after_preproc=False):
        self._base._check_unique_ids(data, after_preproc=after_preproc)

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        return self._base._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format="xywh")

    def get_raw_seq_data(self, tracker, seq):
        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}
        raw_data["similarity_scores"] = [
            self._calculate_similarities(gt_dets_t, tracker_dets_t)
            for gt_dets_t, tracker_dets_t in zip(raw_data["gt_dets"], raw_data["tracker_dets"])
        ]
        return raw_data

    def get_preprocessed_seq_data(self, raw_data, cls):
        self._check_unique_ids(raw_data)
        cls_id = self.class_name_to_class_id[cls]

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids: list[int] = []
        unique_tracker_ids: list[int] = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data["num_timesteps"]):
            gt_mask = np.atleast_1d(raw_data["gt_classes"][t] == cls_id).astype(np.bool)
            tracker_mask = np.atleast_1d(raw_data["tracker_classes"][t] == cls_id).astype(np.bool)

            data["gt_ids"][t] = raw_data["gt_ids"][t][gt_mask]
            data["gt_dets"][t] = raw_data["gt_dets"][t][gt_mask]
            data["tracker_ids"][t] = raw_data["tracker_ids"][t][tracker_mask]
            data["tracker_dets"][t] = raw_data["tracker_dets"][t][tracker_mask]
            data["tracker_confidences"][t] = raw_data["tracker_confidences"][t][tracker_mask]
            data["similarity_scores"][t] = raw_data["similarity_scores"][t][gt_mask][
                :, tracker_mask
            ]

            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][t]))
            num_gt_dets += len(data["gt_ids"][t])
            num_tracker_dets += len(data["tracker_ids"][t])

        if unique_gt_ids:
            unique_gt = np.unique(unique_gt_ids)
            gt_id_map = np.full((np.max(unique_gt) + 1,), np.nan)
            gt_id_map[unique_gt] = np.arange(len(unique_gt))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(np.int)

        if unique_tracker_ids:
            unique_tracker = np.unique(unique_tracker_ids)
            tracker_id_map = np.full((np.max(unique_tracker) + 1,), np.nan)
            tracker_id_map[unique_tracker] = np.arange(len(unique_tracker))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[data["tracker_ids"][t]].astype(np.int)

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(np.unique(unique_tracker_ids)) if unique_tracker_ids else 0
        data["num_gt_ids"] = len(np.unique(unique_gt_ids)) if unique_gt_ids else 0
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]
        self._check_unique_ids(data, after_preproc=True)
        return data


def _read_seq_length(seqinfo_path: Path) -> int:
    seq_length = None
    for line in seqinfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("seqLength="):
            seq_length = int(line.split("=", maxsplit=1)[1].strip())
            break
    if seq_length is None:
        raise ValueError(f"Could not parse seqLength from {seqinfo_path}")
    return seq_length


def write_seqinfo_ini(path: Path, sequence: str, seq_length: int) -> None:
    path.write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={sequence}",
                "imDir=images_rectified_left",
                "frameRate=20",
                f"seqLength={seq_length}",
                f"imWidth={EVENT_WIDTH}",
                f"imHeight={EVENT_HEIGHT}",
                "imExt=.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def export_ground_truth_for_sequence(
    dataset_root: Path,
    split: str,
    sequence: str,
    output_gt_path: Path,
    output_seqinfo_path: Path,
) -> None:
    timestamps = load_image_timestamps(
        dataset_root / split / sequence / f"{sequence}_image_timestamps.txt"
    )
    timestamp_to_frame = {
        timestamp: frame_index + 1 for frame_index, timestamp in enumerate(timestamps)
    }
    annotations = load_annotations(dataset_root / "annotations" / split / f"{sequence}.txt")

    output_gt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_gt_path.open("wt", encoding="utf-8", newline="") as handle:
        for annotation in annotations:
            if annotation.timestamp not in timestamp_to_frame:
                raise ValueError(
                    f"Annotation timestamp {annotation.timestamp} does not map "
                    f"to an image frame in {sequence}"
                )
            frame_index = timestamp_to_frame[annotation.timestamp]
            handle.write(
                f"{frame_index},{annotation.track_id},{annotation.left:.3f},{annotation.top:.3f},"
                f"{annotation.width:.3f},{annotation.height:.3f},1,{annotation.class_id + 1}\n"
            )

    output_seqinfo_path.parent.mkdir(parents=True, exist_ok=True)
    write_seqinfo_ini(output_seqinfo_path, sequence, len(timestamps))


def export_trackeval_bundle(
    dataset_root: Path,
    split: str,
    sequences: list[str],
    tracker_name: str,
    tracker_results_dir: Path,
    output_root: Path,
) -> dict:
    benchmark_name = "dsec_mot"
    gt_root = output_root / "gt"
    trackers_root = output_root / "trackers"
    split_name = f"{benchmark_name}-{split}"

    seqmap_dir = gt_root / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    seqmap_file = seqmap_dir / f"{split_name}.txt"
    seqmap_file.write_text("name\n" + "\n".join(sequences) + "\n", encoding="utf-8")

    for sequence in sequences:
        export_ground_truth_for_sequence(
            dataset_root=dataset_root,
            split=split,
            sequence=sequence,
            output_gt_path=gt_root / split_name / sequence / "gt" / "gt.txt",
            output_seqinfo_path=gt_root / split_name / sequence / "seqinfo.ini",
        )

        source_track_file = tracker_results_dir / f"{sequence}.txt"
        if not source_track_file.exists():
            raise FileNotFoundError(f"Tracker output not found for {sequence}: {source_track_file}")
        target_track_file = trackers_root / split_name / tracker_name / "data" / f"{sequence}.txt"
        target_track_file.parent.mkdir(parents=True, exist_ok=True)
        target_track_file.write_text(
            source_track_file.read_text(encoding="utf-8"), encoding="utf-8"
        )

    return {
        "benchmark": benchmark_name,
        "split": split,
        "gt_folder": str(gt_root),
        "trackers_folder": str(trackers_root),
        "seqmap_file": str(seqmap_file),
        "tracker_name": tracker_name,
    }


def run_trackeval(
    bundle: dict,
    output_root: Path,
    trackeval_root: Path | None = None,
    eval_iou_threshold: float = 0.5,
    classes_to_eval: list[str] | None = None,
) -> tuple[dict, dict]:
    trackeval = ensure_trackeval_importable(trackeval_root)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )

    dataset_config = DSECMOTTrackEvalDataset.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": bundle["gt_folder"],
            "TRACKERS_FOLDER": bundle["trackers_folder"],
            "OUTPUT_FOLDER": str(output_root),
            "TRACKERS_TO_EVAL": [bundle["tracker_name"]],
            "CLASSES_TO_EVAL": classes_to_eval or TRACKEVAL_CLASS_NAMES,
            "BENCHMARK": bundle["benchmark"],
            "SPLIT_TO_EVAL": bundle["split"],
            "PRINT_CONFIG": False,
            "SEQMAP_FILE": bundle["seqmap_file"],
            "SKIP_SPLIT_FOL": False,
        }
    )

    metrics_config = {"THRESHOLD": eval_iou_threshold, "PRINT_CONFIG": False}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = DSECMOTTrackEvalDataset(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    return evaluator.evaluate([dataset], metrics)


def summarise_trackeval_results(
    results: dict,
    tracker_name: str,
    eval_iou_threshold: float = 0.5,
    trackeval_root: Path | None = None,
    classes_to_eval: list[str] | None = None,
) -> dict:
    trackeval = ensure_trackeval_importable(trackeval_root)
    dataset_name = next(iter(results))
    tracker_results = results[dataset_name][tracker_name]

    clear_metric = trackeval.metrics.CLEAR({"THRESHOLD": eval_iou_threshold, "PRINT_CONFIG": False})
    identity_metric = trackeval.metrics.Identity(
        {"THRESHOLD": eval_iou_threshold, "PRINT_CONFIG": False}
    )
    hota_metric = trackeval.metrics.HOTA({"THRESHOLD": eval_iou_threshold, "PRINT_CONFIG": False})

    per_sequence: dict[str, dict] = {}
    class_names = classes_to_eval or TRACKEVAL_CLASS_NAMES
    for sequence, sequence_results in tracker_results.items():
        if sequence == "COMBINED_SEQ":
            continue
        hota_by_class = {
            class_name: sequence_results[class_name]["HOTA"] for class_name in class_names
        }
        clear_by_class = {
            class_name: sequence_results[class_name]["CLEAR"] for class_name in class_names
        }
        identity_by_class = {
            class_name: sequence_results[class_name]["Identity"] for class_name in class_names
        }
        combined_hota = hota_metric.combine_classes_det_averaged(hota_by_class)
        combined_clear = clear_metric.combine_classes_det_averaged(clear_by_class)
        combined_identity = identity_metric.combine_classes_det_averaged(identity_by_class)
        per_sequence[sequence] = {
            "metrics": {
                "HOTA": float(np.mean(combined_hota["HOTA"])),
                "MOTA": float(combined_clear["MOTA"]),
                "IDF1": float(combined_identity["IDF1"]),
                "IDS": int(combined_clear["IDSW"]),
                "FP": int(combined_clear["CLR_FP"]),
                "FN": int(combined_clear["CLR_FN"]),
            },
            "by_class": {
                class_name: {
                    "HOTA": float(np.mean(sequence_results[class_name]["HOTA"]["HOTA"])),
                    "MOTA": float(sequence_results[class_name]["CLEAR"]["MOTA"]),
                    "IDF1": float(sequence_results[class_name]["Identity"]["IDF1"]),
                    "IDS": int(sequence_results[class_name]["CLEAR"]["IDSW"]),
                    "FP": int(sequence_results[class_name]["CLEAR"]["CLR_FP"]),
                    "FN": int(sequence_results[class_name]["CLEAR"]["CLR_FN"]),
                }
                for class_name in TRACKEVAL_CLASS_NAMES
                if class_name in class_names
            },
        }

    combined = tracker_results["COMBINED_SEQ"]["cls_comb_det_av"]
    aggregate = {
        "HOTA": float(np.mean(combined["HOTA"]["HOTA"])),
        "MOTA": float(combined["CLEAR"]["MOTA"]),
        "IDF1": float(combined["Identity"]["IDF1"]),
        "IDS": int(combined["CLEAR"]["IDSW"]),
        "FP": int(combined["CLEAR"]["CLR_FP"]),
        "FN": int(combined["CLEAR"]["CLR_FN"]),
    }

    return {
        "tracker_name": tracker_name,
        "eval_iou_threshold": eval_iou_threshold,
        "aggregate": aggregate,
        "per_sequence": per_sequence,
        "raw_trackeval": _to_builtin(results),
    }


def write_summary_csv(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "HOTA", "MOTA", "IDF1", "IDS", "FP", "FN"])
        for sequence, sequence_summary in summary["per_sequence"].items():
            metrics = sequence_summary["metrics"]
            writer.writerow(
                [
                    sequence,
                    f"{metrics['HOTA']:.6f}",
                    f"{metrics['MOTA']:.6f}",
                    f"{metrics['IDF1']:.6f}",
                    metrics["IDS"],
                    metrics["FP"],
                    metrics["FN"],
                ]
            )
        aggregate = summary["aggregate"]
        writer.writerow(
            [
                "COMBINED",
                f"{aggregate['HOTA']:.6f}",
                f"{aggregate['MOTA']:.6f}",
                f"{aggregate['IDF1']:.6f}",
                aggregate["IDS"],
                aggregate["FP"],
                aggregate["FN"],
            ]
        )


def write_summary_json(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_builtin(summary), indent=2), encoding="utf-8")
