"""Training entry-point script.

Usage::

    python scripts/train.py --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse

import torch

from src.data import EventDataset, EventPreprocessor
from src.models import EventBackbone, EventDetector
from src.training import DetectionLoss, Trainer
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train event-based object detector")
    parser.add_argument("--config", default="configs/base_config.yaml", help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    preprocessor = EventPreprocessor(
        height=cfg["data"]["height"],
        width=cfg["data"]["width"],
        representation="event_frame",
    )

    train_dataset = EventDataset(
        root=cfg["data"]["root_dir"],
        split="train",
        transform=preprocessor,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=True,
    )

    backbone = EventBackbone(in_channels=2)
    model = EventDetector(backbone=backbone, num_classes=cfg["model"]["num_classes"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = DetectionLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        log_dir=cfg["training"]["log_dir"],
    )
    trainer.fit(train_loader, num_epochs=cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
