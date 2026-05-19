"""Shared EvRT-DETR runtime helpers for DSEC-MOT inference and export."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH, _timestamp_window_indices


REPO_ROOT = Path(__file__).resolve().parents[2]
EVRT_DETR_ROOT = REPO_ROOT / "external" / "evrt-detr"
DEFAULT_PUBLIC_MODEL_ROOT = REPO_ROOT / "external" / "models" / "evrt-detr" / "gen4_frame_rtdetr_presnet50"


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


@dataclass(frozen=True)
class Prediction:
    label: int
    score: float
    box: tuple[float, float, float, float]


def import_evrtdetr_parts():
    if str(EVRT_DETR_ROOT) not in sys.path:
        sys.path.insert(0, str(EVRT_DETR_ROOT))

    try:
        from evlearn.nn.backbone.presnet_rtdetr import PResNetRTDETR
        from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.hybrid_encoder import HybridEncoder
        from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "Failed to import EvRT-DETR runtime pieces. Install the external runtime "
            "dependencies into `.venv` first."
        ) from exc

    return PResNetRTDETR, HybridEncoder, RTDETRTransformer, RTDETRPostProcessor


def resolve_model_dir(root: Path) -> Path:
    root = root.resolve()
    if (root / "config.json").exists():
        return root

    candidates = sorted(path.parent for path in root.rglob("config.json"))
    valid = [
        path for path in candidates
            if any(path.glob("net_*_decoder.pth")) or (path / "net_decoder.pth").exists()
    ]

    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise SystemExit(f"Could not find an EvRT-DETR model directory under {root}")

    raise SystemExit(f"Found multiple candidate model directories under {root}: {valid}")


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


def load_evrtdetr_model(model_dir: Path, device_name: str, forced_n_bins: int = 0) -> LoadedEvRTDETR:
    PResNetRTDETR, HybridEncoder, RTDETRTransformer, RTDETRPostProcessor = import_evrtdetr_parts()

    model_dir = resolve_model_dir(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
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
    np_mod = np,
) -> np.ndarray:
    start_us = timestamp_us - window_us
    start_idx, end_idx = _timestamp_window_indices(ms_to_idx, t, t_offset, start_us, timestamp_us)

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

    return padded, ResizeMeta(scale=scale, pad_left=pad_left, pad_top=pad_top)


def undo_letterbox(boxes: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    restored = boxes.copy()
    restored[:, [0, 2]] -= meta.pad_left
    restored[:, [1, 3]] -= meta.pad_top
    restored /= meta.scale
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, EVENT_WIDTH - 1)
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, EVENT_HEIGHT - 1)
    return restored


@torch.inference_mode()
def run_evrtdetr_inference(
    model: LoadedEvRTDETR,
    hist: np.ndarray,
    score_threshold: float,
) -> list[Prediction]:
    frame, meta = letterbox_frame(hist, model.input_hw)
    frame = frame.to(model.device, dtype=torch.float32, non_blocking=True)

    features = model.backbone(frame)
    encoded = model.encoder(features)
    decoded = model.decoder(encoded)
    orig_sizes = torch.tensor(
        [[model.input_hw[1], model.input_hw[0]]],
        dtype=torch.float32,
        device=model.device,
    )
    labels, boxes, scores = model.postprocessor(decoded, orig_sizes)

    labels_np = labels[0].detach().cpu().numpy().astype(np.int32)
    boxes_np = undo_letterbox(boxes[0].detach().cpu().numpy(), meta)
    scores_np = scores[0].detach().cpu().numpy().astype(np.float32)

    predictions: list[Prediction] = []
    for label, score, box in zip(labels_np, scores_np, boxes_np):
        if float(score) < score_threshold:
            continue
        left, top, right, bottom = [float(value) for value in box]
        if right <= left or bottom <= top:
            continue
        predictions.append(
            Prediction(
                label=int(label),
                score=float(score),
                box=(left, top, right, bottom),
            )
        )

    return predictions
