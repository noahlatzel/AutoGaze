import pytest
import torch

from autogaze.models.autogaze.modeling_autogaze import TemporalPositionSignal


def test_zero_gate_is_exact_identity_and_only_gate_is_trainable():
    module = TemporalPositionSignal(hidden_dim=12)
    features = torch.randn(2, 16, 7, 12)
    output = module(features)
    assert torch.equal(output, features)
    assert [(name, parameter.numel()) for name, parameter in module.named_parameters()] == [
        ("gate", 1)
    ]


def test_signal_is_deterministic_unit_rms_and_temporally_distinct():
    module = TemporalPositionSignal(hidden_dim=13, max_frequency=8.0)
    first = module.encoding(16, device=torch.device("cpu"), dtype=torch.float32)
    second = module.encoding(16, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(first, second)
    assert first.square().mean(-1).sqrt() == pytest.approx(torch.ones(16))
    assert not torch.equal(first[0], first[-1])


def test_gate_controls_signal_to_feature_rms_ratio():
    module = TemporalPositionSignal(hidden_dim=12)
    module.gate.data.fill_(0.2)
    features = torch.randn(2, 16, 7, 12)
    contribution = module(features) - features
    feature_rms = features.square().mean(dim=(-2, -1)).sqrt()
    contribution_rms = contribution.square().mean(dim=(-2, -1)).sqrt()
    torch.testing.assert_close(
        contribution_rms / feature_rms,
        torch.full_like(feature_rms, 0.2),
        atol=1e-6,
        rtol=0,
    )
