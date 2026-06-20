import torch

from src.models.pretrained_detector import RepresentationAdapter


def test_single_adapter_outputs_rgb_range():
    adapter = RepresentationAdapter((10,), mode="single", width=16)
    output = adapter(torch.rand(2, 10, 32, 40))

    assert output.shape == (2, 3, 32, 40)
    assert 0.0 <= float(output.detach().min()) <= float(output.detach().max()) <= 1.0


def test_multi_branch_adapter_splits_components():
    adapter = RepresentationAdapter((2, 10, 1), mode="multi_branch", width=24)
    output = adapter(torch.rand(2, 13, 32, 40))

    assert output.shape == (2, 3, 32, 40)


def test_each_adapter_component_is_normalised_independently():
    event_frame = torch.full((1, 2, 4, 4), 100.0)
    eros = torch.ones((1, 1, 4, 4))

    event_frame_normalised = RepresentationAdapter.normalise(event_frame)
    eros_normalised = RepresentationAdapter.normalise(eros)

    assert float(event_frame_normalised.max()) == 1.0
    assert float(eros_normalised.max()) == 1.0
