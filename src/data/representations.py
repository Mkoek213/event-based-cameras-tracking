"""Shared representation helpers for controlled benchmark variants."""

from __future__ import annotations

import numpy as np

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.data.preprocessing import EventPreprocessor

REPRESENTATION_CHOICES = (
    "event_frame",
    "voxel_grid",
    "event_frame_voxel_grid",
    "eros",
    "event_frame_eros",
    "voxel_grid_eros",
    "event_frame_voxel_grid_eros",
)
PRETRAINED_REPRESENTATION_CHOICES = REPRESENTATION_CHOICES


def representation_components(representation: str) -> tuple[str, ...]:
    components = {
        "event_frame": ("event_frame",),
        "voxel_grid": ("voxel_grid",),
        "event_frame_voxel_grid": ("event_frame", "voxel_grid"),
        "eros": ("eros",),
        "event_frame_eros": ("event_frame", "eros"),
        "voxel_grid_eros": ("voxel_grid", "eros"),
        "event_frame_voxel_grid_eros": ("event_frame", "voxel_grid", "eros"),
    }
    try:
        return components[representation]
    except KeyError as exc:
        raise ValueError(f"Unknown representation '{representation}'.") from exc


def representation_channels(representation: str, num_bins: int) -> int:
    channel_counts = {"event_frame": 2, "voxel_grid": 2 * num_bins, "eros": 1}
    return sum(channel_counts[component] for component in representation_components(representation))


def representation_channel_splits(representation: str, num_bins: int) -> tuple[int, ...]:
    channel_counts = {"event_frame": 2, "voxel_grid": 2 * num_bins, "eros": 1}
    return tuple(
        channel_counts[component] for component in representation_components(representation)
    )


class BenchmarkRepresentation:
    """Callable transform returning a dense `(C, H, W)` event representation."""

    def __init__(
        self,
        representation: str,
        num_bins: int,
        height: int = EVENT_HEIGHT,
        width: int = EVENT_WIDTH,
    ) -> None:
        if representation not in PRETRAINED_REPRESENTATION_CHOICES:
            raise ValueError(
                f"Unknown benchmark representation '{representation}'. "
                f"Choose one of {PRETRAINED_REPRESENTATION_CHOICES}."
            )
        self.representation = representation
        self.num_bins = int(num_bins)
        self.height = int(height)
        self.width = int(width)
        self.event_frame = EventPreprocessor(
            height, width, representation="event_frame", num_bins=num_bins
        )
        self.voxel_grid = EventPreprocessor(
            height, width, representation="voxel_grid", num_bins=num_bins
        )

    @property
    def channels(self) -> int:
        return representation_channels(self.representation, self.num_bins)

    def __call__(self, events: np.ndarray, eros: np.ndarray | None = None) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for component in representation_components(self.representation):
            if component == "event_frame":
                outputs.append(self.event_frame(events))
            elif component == "voxel_grid":
                outputs.append(self.voxel_grid(events))
            else:
                if eros is None:
                    raise ValueError(
                        f"Representation '{self.representation}' requires an EROS snapshot."
                    )
                surface = np.asarray(eros, dtype=np.float32)
                if surface.shape == (self.height, self.width):
                    surface = surface[None]
                if surface.shape != (1, self.height, self.width):
                    raise ValueError(
                        f"Expected EROS shape {(self.height, self.width)} or "
                        f"{(1, self.height, self.width)}, got {surface.shape}."
                    )
                outputs.append(surface / 255.0 if surface.max() > 1.0 else surface)
        return outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)
