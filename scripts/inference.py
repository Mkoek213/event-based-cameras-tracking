"""Inference entry-point script.

Usage::

    python scripts/inference.py --config configs/base_config.yaml \\
        --checkpoint models/checkpoints/checkpoint_epoch_0100.pt \\
        --input data/raw/sample.dat --output results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import EventPreprocessor
from src.models import EventBackbone, EventDetector
from src.utils import load_config, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with event-based detector")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to raw event file or directory")
    parser.add_argument("--output", default="results/", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocessor = EventPreprocessor(
        height=cfg["data"]["height"],
        width=cfg["data"]["width"],
    )

    backbone = EventBackbone(in_channels=2)
    model = EventDetector(backbone=backbone, num_classes=cfg["model"]["num_classes"])
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Stub: load events and run a single forward pass
    dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.bool_)])
    events = np.empty(0, dtype=dtype)
    frame = preprocessor(events)
    tensor = torch.from_numpy(frame).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        cls_logits, bbox_offsets = model(tensor)

    results = {
        "cls_logits_shape": list(cls_logits.shape),
        "bbox_offsets_shape": list(bbox_offsets.shape),
    }
    save_results(results, output_dir / "predictions.json")
    print(f"Inference complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
