"""Event stream preprocessing and transform utilities."""

from __future__ import annotations

import numpy as np


class EventPreprocessor:
    """Converts a raw event array into a fixed-size tensor representation.

    Supported representations:
        - ``"event_frame"``: polarity-split 2-D histogram (H × W × 2)
        - ``"time_surface"``: most-recent timestamp surface (H × W × 2)
        - ``"voxel_grid"``: temporal voxel grid (B × H × W)
    """

    REPRESENTATIONS = ("event_frame", "time_surface", "voxel_grid")

    def __init__(
        self,
        height: int,
        width: int,
        representation: str = "event_frame",
        num_bins: int = 5,
    ) -> None:
        if representation not in self.REPRESENTATIONS:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                f"Choose one of {self.REPRESENTATIONS}."
            )
        self.height = height
        self.width = width
        self.representation = representation
        self.num_bins = num_bins

    # ------------------------------------------------------------------

    def __call__(self, events: np.ndarray) -> np.ndarray:
        if self.representation == "event_frame":
            return self._to_event_frame(events)
        if self.representation == "time_surface":
            return self._to_time_surface(events)
        return self._to_voxel_grid(events)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _to_event_frame(self, events: np.ndarray) -> np.ndarray:
        """Polarity-split event histogram."""
        frame = np.zeros((2, self.height, self.width), dtype=np.float32)
        if events.size == 0:
            return frame
        x = events["x"].astype(int)
        y = events["y"].astype(int)
        p = events["p"].astype(int)
        mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        np.add.at(frame[p[mask]], (y[mask], x[mask]), 1)
        return frame

    def _to_time_surface(self, events: np.ndarray) -> np.ndarray:
        """Most-recent-timestamp surface per polarity."""
        surface = np.zeros((2, self.height, self.width), dtype=np.float32)
        if events.size == 0:
            return surface
        x = events["x"].astype(int)
        y = events["y"].astype(int)
        t = events["t"].astype(np.float32)
        p = events["p"].astype(int)
        mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        for xi, yi, ti, pi in zip(x[mask], y[mask], t[mask], p[mask]):
            surface[pi, yi, xi] = max(surface[pi, yi, xi], ti)
        if surface.max() > 0:
            surface /= surface.max()
        return surface

    def _to_voxel_grid(self, events: np.ndarray) -> np.ndarray:
        """Temporal voxel grid with *num_bins* time slices."""
        grid = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        if events.size == 0:
            return grid
        t = events["t"].astype(np.float64)
        t_min, t_max = t.min(), t.max()
        dt = t_max - t_min if t_max > t_min else 1.0
        bins = ((t - t_min) / dt * (self.num_bins - 1)).astype(int).clip(0, self.num_bins - 1)
        x = events["x"].astype(int)
        y = events["y"].astype(int)
        mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        np.add.at(grid, (bins[mask], y[mask], x[mask]), 1)
        return grid
