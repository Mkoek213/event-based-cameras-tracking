#!/usr/bin/env python3
"""Run a pretrained EvRT-DETR RT-DETR checkpoint on DSEC-MOT sequences."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from visualize_dsec_mot_samples import (
    CLASS_NAMES,
    EVENT_HEIGHT,
    EVENT_WIDTH,
    build_event_frame,
    build_timestamp_to_png,
    group_annotations_by_timestamp,
    import_or_die,
    load_annotations,
    load_event_file,
    load_image_timestamps,
    timestamp_window_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVRT_DETR_ROOT = REPO_ROOT / "external" / "evrt-detr"
DEFAULT_MODEL_ROOT = REPO_ROOT / "external" / "models" / "evrt-detr" / "gen4_frame_rtdetr_presnet50"

TOP_BAR = 34
GAP = 16
PRED_COLOR = (0, 165, 255)
GT_COLOR = (0, 255, 0)


@dataclass(frozen=True)
class LoadedEvRTDETR:
    backbone: torch.nn.Module
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    postprocessor: torch.nn.Module
    class_names: list[str]
    input_hw: tuple[int, int]
    n_bins: int
    model_dir: Path
    device: torch.device
    uses_ema: bool


@dataclass(frozen=True)
class ResizeMeta:
    scale: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int
    target_height: int
    target_width: int


@dataclass(frozen=True)
class Prediction:
    label: int
    score: float
    box: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/datasets/dsec_mot"),
        help="Path to the local DSEC-MOT root.",
    )
    parser.add_argument("--split", choices=("train", "test"), required=True, help="Dataset split.")
    parser.add_argument("--sequence", required=True, help="Sequence name, for example interlaken_00_a.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Directory with the unpacked EvRT-DETR checkpoint.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=50.0,
        help="Event accumulation window in milliseconds before each timestamp.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=0,
        help="Override the number of temporal bins. 0 uses the checkpoint default.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Output video frames per second.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Discard predictions below this confidence.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start from this image/frame index.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to render. 0 means all selected frames.",
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Render only timestamps that have ground-truth annotations.",
    )
    parser.add_argument(
        "--with-ground-truth",
        action="store_true",
        help="Overlay DSEC-MOT ground-truth boxes on the event view in green.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference, for example cuda or cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to data/processed/dsec_mot_predictions/<split>/<sequence>_evrtdetr.avi",
    )
    return parser.parse_args()


def import_evrtdetr_parts():
    if str(EVRT_DETR_ROOT) not in sys.path:
        sys.path.insert(0, str(EVRT_DETR_ROOT))

    try:
        from evlearn.nn.backbone.presnet_rtdetr import PResNetRTDETR
        from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.hybrid_encoder import HybridEncoder
        from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    except ImportError as exc:  # pragma: no cover - dependency error at runtime
        raise SystemExit(
            "Failed to import EvRT-DETR runtime pieces. "
            "Install the missing dependencies into .venv, for example: "
            "`./.venv/bin/pip install psutil tqdm`."
        ) from exc

    return PResNetRTDETR, HybridEncoder, RTDETRTransformer, RTDETRPostProcessor


def resolve_model_dir(root: Path) -> Path:
    root = root.resolve()
    if (root / "config.json").exists():
        return root

    candidates = sorted(path.parent for path in root.rglob("config.json"))
    valid = [path for path in candidates if any(path.glob("net_*_decoder.pth")) or (path / "net_decoder.pth").exists()]
    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise SystemExit(f"Could not find an EvRT-DETR model directory under {root}")
    raise SystemExit(f"Found multiple candidate model directories under {root}: {valid}")


def load_json(path: Path) -> dict:
    with path.open("rt", encoding="utf-8") as handle:
        return json.load(handle)


def strip_name(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "name"}


def choose_weight_path(model_dir: Path, ema_name: str, base_name: str) -> tuple[Path, bool]:
    ema_path = model_dir / ema_name
    if ema_path.exists():
        return ema_path, True

    base_path = model_dir / base_name
    if base_path.exists():
        return base_path, False

    raise SystemExit(f"Missing both {ema_name} and {base_name} in {model_dir}")


def load_state_dict(module: torch.nn.Module, path: Path, device: torch.device) -> None:
    state_dict = torch.load(path, map_location=device)
    try:
        module.load_state_dict(state_dict)
    except RuntimeError:
        if not any(key.startswith("module.") for key in state_dict):
            raise
        stripped = {key.removeprefix("module."): value for key, value in state_dict.items()}
        module.load_state_dict(stripped)


def load_model(model_dir: Path, device_name: str, forced_n_bins: int) -> LoadedEvRTDETR:
    PResNetRTDETR, HybridEncoder, RTDETRTransformer, RTDETRPostProcessor = import_evrtdetr_parts()

    model_dir = resolve_model_dir(model_dir)
    config = load_json(model_dir / "config.json")
    frame_shape = tuple(config["data"]["eval"]["shapes"][0])
    if len(frame_shape) != 3:
        raise SystemExit(f"Unexpected frame shape in config: {frame_shape}")

    channels, input_h, input_w = frame_shape
    if channels % 2 != 0:
        raise SystemExit(f"Expected an even number of channels, got {channels}")

    n_bins = channels // 2
    if forced_n_bins:
        if forced_n_bins * 2 != channels:
            raise SystemExit(
                f"Checkpoint expects {channels} channels, which means {n_bins} bins. "
                f"The requested override --n-bins={forced_n_bins} is incompatible."
            )
        n_bins = forced_n_bins

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False in the current environment.")

    device = torch.device(device_name)
    backbone = PResNetRTDETR(
        input_shape=frame_shape,
        **strip_name(config["nets"]["backbone"]["model"]),
    ).to(device)
    encoder = HybridEncoder(**strip_name(config["nets"]["encoder"]["model"])).to(device)
    decoder = RTDETRTransformer(**strip_name(config["nets"]["decoder"]["model"])).to(device)
    postprocessor = RTDETRPostProcessor(**config["model"]["rtdetr_postproc_kwargs"]).to(device)
    postprocessor.deploy()

    backbone_path, backbone_ema = choose_weight_path(model_dir, "net_ema_backbone.pth", "net_backbone.pth")
    encoder_path, encoder_ema = choose_weight_path(model_dir, "net_ema_encoder.pth", "net_encoder.pth")
    decoder_path, decoder_ema = choose_weight_path(model_dir, "net_ema_decoder.pth", "net_decoder.pth")

    load_state_dict(backbone, backbone_path, device)
    load_state_dict(encoder, encoder_path, device)
    load_state_dict(decoder, decoder_path, device)

    backbone.eval()
    encoder.eval()
    decoder.eval()
    postprocessor.eval()

    classes = list(config["model"]["evaluator"]["classes"])
    return LoadedEvRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        class_names=classes,
        input_hw=(input_h, input_w),
        n_bins=n_bins,
        model_dir=model_dir,
        device=device,
        uses_ema=backbone_ema and encoder_ema and decoder_ema,
    )


def build_stacked_histogram(
    x,
    y,
    p,
    t,
    ms_to_idx,
    t_offset: int,
    timestamp_us: int,
    window_us: int,
    n_bins: int,
    np_mod,
) -> np.ndarray:
    start_us = timestamp_us - window_us
    start_idx, end_idx = timestamp_window_indices(ms_to_idx, t, t_offset, start_us, timestamp_us, np_mod)

    hist = np_mod.zeros((2, n_bins, EVENT_HEIGHT, EVENT_WIDTH), dtype=np_mod.float32)
    if end_idx <= start_idx:
        return hist

    xs = x[start_idx:end_idx].astype(np_mod.int32)
    ys = y[start_idx:end_idx].astype(np_mod.int32)
    ts = t[start_idx:end_idx].astype(np_mod.int64) + int(t_offset)
    ps = p[start_idx:end_idx].astype(np_mod.bool_)

    valid = (xs >= 0) & (xs < EVENT_WIDTH) & (ys >= 0) & (ys < EVENT_HEIGHT)
    if not np_mod.any(valid):
        return hist

    xs = xs[valid]
    ys = ys[valid]
    ts = ts[valid]
    ps = ps[valid]

    time_min = int(ts.min())
    time_max = int(ts.max()) + 1
    curr_time = time_min
    time_step = (time_max - time_min) / n_bins if n_bins > 0 else 0

    for bin_idx in range(n_bins):
        next_time = time_max if bin_idx == n_bins - 1 else curr_time + time_step
        mask_bin = (ts >= curr_time) & (ts < next_time)
        if np_mod.any(mask_bin):
            pos = mask_bin & ps
            neg = mask_bin & (~ps)
            np_mod.add.at(hist, (0, bin_idx, ys[pos], xs[pos]), 1)
            np_mod.add.at(hist, (1, bin_idx, ys[neg], xs[neg]), 1)
        curr_time = next_time

    return hist


def letterbox_frame(hist: np.ndarray, target_hw: tuple[int, int]) -> tuple[torch.Tensor, ResizeMeta]:
    _, _, source_h, source_w = hist.shape
    target_h, target_w = target_hw
    scale = min(target_w / source_w, target_h / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))

    frame = torch.from_numpy(hist.reshape((-1, source_h, source_w))).unsqueeze(0)
    frame = F.interpolate(frame, size=(resized_h, resized_w), mode="bilinear", align_corners=False)

    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    padded = torch.zeros((1, frame.shape[1], target_h, target_w), dtype=frame.dtype)
    padded[:, :, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = frame

    meta = ResizeMeta(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=resized_w,
        resized_height=resized_h,
        target_height=target_h,
        target_width=target_w,
    )
    return padded, meta


def undo_letterbox(boxes: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    restored = boxes.copy()
    restored[:, [0, 2]] -= meta.pad_left
    restored[:, [1, 3]] -= meta.pad_top
    restored /= meta.scale
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, EVENT_WIDTH - 1)
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, EVENT_HEIGHT - 1)
    return restored


@torch.inference_mode()
def run_inference(model: LoadedEvRTDETR, hist: np.ndarray, score_threshold: float) -> list[Prediction]:
    frame, meta = letterbox_frame(hist, model.input_hw)
    frame = frame.to(model.device, dtype=torch.float32, non_blocking=True)

    features = model.backbone(frame)
    encoded = model.encoder(features)
    decoded = model.decoder(encoded)
    orig_sizes = torch.tensor([[model.input_hw[1], model.input_hw[0]]], dtype=torch.float32, device=model.device)
    labels, boxes, scores = model.postprocessor(decoded, orig_sizes)

    labels_np = labels[0].detach().cpu().numpy().astype(np.int32)
    boxes_np = boxes[0].detach().cpu().numpy()
    scores_np = scores[0].detach().cpu().numpy().astype(np.float32)
    boxes_np = undo_letterbox(boxes_np, meta)

    predictions: list[Prediction] = []
    for label, score, box in zip(labels_np, scores_np, boxes_np):
        if float(score) < score_threshold:
            continue
        left, top, right, bottom = [float(value) for value in box]
        if right <= left or bottom <= top:
            continue
        predictions.append(Prediction(label=int(label), score=float(score), box=(left, top, right, bottom)))
    return predictions


def draw_ground_truth(frame_bgr: np.ndarray, annotations) -> np.ndarray:
    output = frame_bgr.copy()
    for annotation in annotations:
        left = max(0, int(round(annotation.left)))
        top = max(0, int(round(annotation.top)))
        right = min(EVENT_WIDTH - 1, int(round(annotation.left + annotation.width)))
        bottom = min(EVENT_HEIGHT - 1, int(round(annotation.top + annotation.height)))
        cv2.rectangle(output, (left, top), (right, bottom), GT_COLOR, 2)
        gt_label = CLASS_NAMES.get(annotation.class_id, str(annotation.class_id))
        text = f"gt:{gt_label} id={annotation.track_id}"
        text_y = max(14, top - 6)
        cv2.putText(output, text, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GT_COLOR, 1, cv2.LINE_AA)
    return output


def draw_predictions(frame_bgr: np.ndarray, predictions: list[Prediction], class_names: list[str]) -> np.ndarray:
    output = frame_bgr.copy()
    for prediction in predictions:
        left, top, right, bottom = prediction.box
        left_i = max(0, int(round(left)))
        top_i = max(0, int(round(top)))
        right_i = min(EVENT_WIDTH - 1, int(round(right)))
        bottom_i = min(EVENT_HEIGHT - 1, int(round(bottom)))
        cv2.rectangle(output, (left_i, top_i), (right_i, bottom_i), PRED_COLOR, 2)
        class_name = class_names[prediction.label] if 0 <= prediction.label < len(class_names) else str(prediction.label)
        text = f"pred:{class_name} {prediction.score:.2f}"
        text_y = min(EVENT_HEIGHT - 8, bottom_i + 14) if top_i < 16 else top_i - 6
        cv2.putText(output, text, (left_i, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_COLOR, 1, cv2.LINE_AA)
    return output


def compose_frame(
    event_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    timestamp_us: int,
    pred_count: int,
    model_label: str,
    gt_count: int | None,
) -> np.ndarray:
    right = cv2.resize(rectified_bgr, (EVENT_WIDTH, EVENT_HEIGHT), interpolation=cv2.INTER_AREA)
    canvas = np.full((EVENT_HEIGHT + TOP_BAR, EVENT_WIDTH * 2 + GAP, 3), 24, dtype=np.uint8)
    canvas[TOP_BAR:, :EVENT_WIDTH] = event_bgr
    canvas[TOP_BAR:, EVENT_WIDTH + GAP :] = right

    gt_text = f" | gt={gt_count}" if gt_count is not None else ""
    cv2.putText(
        canvas,
        f"{model_label} | t={timestamp_us} us | pred={pred_count}{gt_text}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Rectified left frame | reference only",
        (EVENT_WIDTH + GAP + 8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def default_output_path(split: str, sequence: str) -> Path:
    return Path("data/processed/dsec_mot_predictions") / split / f"{sequence}_evrtdetr.avi"


def main() -> int:
    args = parse_args()
    h5py, np_h5, _image, _image_draw = import_or_die()

    seq_dir = args.root / args.split / args.sequence
    annotation_path = args.root / "annotations" / args.split / f"{args.sequence}.txt"
    events_h5 = seq_dir / "events_left" / "events.h5"

    if not seq_dir.exists():
        raise SystemExit(f"Sequence directory does not exist: {seq_dir}")
    if not annotation_path.exists():
        raise SystemExit(f"Annotation file does not exist: {annotation_path}")
    if not events_h5.exists():
        raise SystemExit(f"Missing events file: {events_h5}")

    model = load_model(args.model_dir, args.device, args.n_bins)

    timestamps = load_image_timestamps(seq_dir / f"{args.sequence}_image_timestamps.txt")
    grouped = group_annotations_by_timestamp(load_annotations(annotation_path))
    png_map = build_timestamp_to_png(seq_dir, args.sequence)

    if args.annotated_only:
        timestamps = [timestamp for timestamp in timestamps if timestamp in grouped]
    if args.start_frame:
        timestamps = timestamps[args.start_frame :]
    if args.max_frames > 0:
        timestamps = timestamps[: args.max_frames]
    if not timestamps:
        raise SystemExit("No frames selected for inference.")

    output_path = args.output or default_output_path(args.split, args.sequence)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (EVENT_WIDTH * 2 + GAP, EVENT_HEIGHT + TOP_BAR),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {output_path}")

    window_us = int(args.window_ms * 1000)
    handle, x, y, p, t, ms_to_idx, t_offset = load_event_file({"events_h5": events_h5}, h5py)
    model_label = model.model_dir.name
    try:
        total = len(timestamps)
        for index, timestamp_us in enumerate(timestamps, start=1):
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
            predictions = run_inference(model, hist, args.score_threshold)

            event_rgb = build_event_frame(x, y, p, t, ms_to_idx, t_offset, timestamp_us, window_us, np_h5)
            event_bgr = cv2.cvtColor(event_rgb, cv2.COLOR_RGB2BGR)
            if args.with_ground_truth:
                event_bgr = draw_ground_truth(event_bgr, grouped.get(timestamp_us, []))
            event_bgr = draw_predictions(event_bgr, predictions, model.class_names)

            rectified_path = png_map.get(timestamp_us)
            if rectified_path is None:
                raise SystemExit(f"No PNG found for timestamp {timestamp_us}")
            rectified_bgr = cv2.imread(str(rectified_path), cv2.IMREAD_COLOR)
            if rectified_bgr is None:
                raise SystemExit(f"Could not read {rectified_path}")

            frame = compose_frame(
                event_bgr=event_bgr,
                rectified_bgr=rectified_bgr,
                timestamp_us=timestamp_us,
                pred_count=len(predictions),
                model_label=model_label,
                gt_count=len(grouped.get(timestamp_us, [])) if args.with_ground_truth else None,
            )
            writer.write(frame)

            if index == 1 or index == total or index % 25 == 0:
                print(f"{index}/{total}: {timestamp_us} | preds={len(predictions)}")
    finally:
        handle.close()
        writer.release()

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
