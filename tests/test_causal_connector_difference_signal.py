import pytest
import torch

from autogaze.models.autogaze.modeling_autogaze import (
    CausalConnectorDifferenceSignal,
)


def test_zero_gate_is_exact_identity_and_only_gate_is_trainable():
    module = CausalConnectorDifferenceSignal(hidden_dim=12)
    features = torch.randn(2, 16, 7, 12)
    output = module(features)
    assert torch.equal(output, features)
    assert [
        (name, parameter.numel()) for name, parameter in module.named_parameters()
    ] == [("gate", 1)]


def test_difference_is_causal_and_normalized_over_each_frame():
    module = CausalConnectorDifferenceSignal(hidden_dim=5)
    features = torch.randn(2, 4, 7, 5)
    signal = module.normalized_difference(features)
    assert torch.count_nonzero(signal[:, 0]) == 0
    assert signal[:, 1:].square().mean(dim=(-2, -1)).sqrt() == pytest.approx(
        torch.ones(2, 3)
    )

    changed_future = features.clone()
    changed_future[:, 3] += 100
    changed_signal = module.normalized_difference(changed_future)
    assert torch.equal(signal[:, :3], changed_signal[:, :3])


def test_connector_spatial_position_cancels_from_difference():
    module = CausalConnectorDifferenceSignal(hidden_dim=6)
    features = torch.randn(2, 4, 7, 6)
    spatial_position = torch.randn(7, 6)
    first = module.normalized_difference(features)
    second = module.normalized_difference(features + spatial_position[None, None])
    torch.testing.assert_close(first, second, atol=1e-6, rtol=0)


def test_gate_bounds_per_frame_signal_to_feature_rms():
    module = CausalConnectorDifferenceSignal(hidden_dim=8)
    module.gate.data.fill_(-0.2)
    features = torch.randn(2, 5, 7, 8)
    contribution = module(features) - features
    feature_rms = features.square().mean(dim=(-2, -1)).sqrt()
    contribution_rms = contribution.square().mean(dim=(-2, -1)).sqrt()
    ratio = contribution_rms / feature_rms
    torch.testing.assert_close(ratio[:, 0], torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        ratio[:, 1:], torch.full_like(ratio[:, 1:], 0.2), atol=1e-6, rtol=0
    )
