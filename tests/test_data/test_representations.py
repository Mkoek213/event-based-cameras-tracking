import numpy as np
import pytest

from src.data.representations import (
    BenchmarkRepresentation,
    representation_channel_splits,
    representation_channels,
)


def _events():
    dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.bool_)])
    return np.array([(1, 2, 1, False), (3, 4, 2, True)], dtype=dtype)


def test_eros_fusion_channels_and_output():
    transform = BenchmarkRepresentation(
        "event_frame_voxel_grid_eros", num_bins=3, height=8, width=8
    )
    output = transform(_events(), eros=np.full((8, 8), 255, dtype=np.uint8))

    assert representation_channels("event_frame_voxel_grid_eros", 3) == 9
    assert representation_channel_splits("event_frame_voxel_grid_eros", 3) == (2, 6, 1)
    assert output.shape == (9, 8, 8)
    assert output[-1].max() == 1.0


def test_eros_representation_requires_snapshot():
    transform = BenchmarkRepresentation("eros", num_bins=3, height=8, width=8)
    with pytest.raises(ValueError, match="requires an EROS snapshot"):
        transform(_events())
