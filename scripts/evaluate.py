"""Evaluation entry-point script.

Usage::

    python scripts/evaluate.py --config configs/base_config.yaml \\
        --checkpoint models/checkpoints/checkpoint_epoch_0100.pt
"""

from __future__ import annotations

import argparse

import torch

from src.data import EventDataset, EventPreprocessor
from src.evaluation import Evaluator
from src.models import EventBackbone, EventDetector
from src.utils import load_config, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate event-based object detector")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", default="results/eval.json", help="Path to save metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocessor = EventPreprocessor(
        height=cfg["data"]["height"],
        width=cfg["data"]["width"],
    )
    val_dataset = EventDataset(root=cfg["data"]["root_dir"], split="val", transform=preprocessor)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
    )

    backbone = EventBackbone(in_channels=2)
    model = EventDetector(backbone=backbone, num_classes=cfg["model"]["num_classes"])
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    evaluator = Evaluator(iou_threshold=cfg["evaluation"]["iou_threshold"])

    with torch.no_grad():
        for batch in val_loader:
            _ = model(batch["events"].to(device))
            # TODO: decode predictions and call evaluator.update(gt, pred, ids)

    result = evaluator.compute()
    print(f"MOTA: {result.mota:.4f}  IDF1: {result.idf1:.4f}")
    save_results(vars(result), args.output)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
