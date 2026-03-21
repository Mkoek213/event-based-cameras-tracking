"""Training loop implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Manages the training loop for an event-based detection/tracking model.

    Args:
        model: The PyTorch model to train.
        optimizer: Optimiser instance.
        criterion: Loss function returning ``(total_loss, loss_dict)``.
        device: Torch device string or object.
        checkpoint_dir: Directory to save checkpoints.
        log_dir: Directory to write TensorBoard logs.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str | torch.device = "cpu",
        checkpoint_dir: str | Path = "models/checkpoints",
        log_dir: str | Path = "experiments/logs",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._epoch = 0

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, num_epochs: int, val_loader: Optional[DataLoader] = None) -> None:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            num_epochs: Number of epochs to train.
            val_loader: Optional DataLoader for validation.
        """
        for epoch in range(num_epochs):
            self._epoch = epoch + 1
            train_loss = self._train_epoch(train_loader)
            print(f"Epoch [{self._epoch}/{num_epochs}] train_loss={train_loss:.4f}", flush=True)
            if val_loader is not None:
                val_loss = self._val_epoch(val_loader)
                print(f"  val_loss={val_loss:.4f}", flush=True)
            self._save_checkpoint()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            self.optimizer.zero_grad()
            loss = self._forward_pass(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                loss = self._forward_pass(batch)
                total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _forward_pass(self, batch: dict) -> torch.Tensor:
        inputs = batch["events"].to(self.device)
        targets = batch.get("label")
        outputs = self.model(inputs)
        if targets is None or self.criterion is None:
            return outputs[0].sum() * 0.0
        loss, _ = self.criterion(*outputs, *targets)
        return loss

    def _save_checkpoint(self) -> None:
        path = self.checkpoint_dir / f"checkpoint_epoch_{self._epoch:04d}.pt"
        torch.save(
            {
                "epoch": self._epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
