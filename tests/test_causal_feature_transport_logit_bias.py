import pytest
import torch

from autogaze.models.autogaze.modeling_autogaze import (
    AdditiveTokenBiasLogitsProcessor,
    CausalFeatureTransportLogitBias,
)


def test_correspondence_follows_selected_appearance_and_is_normalized():
    module = CausalFeatureTransportLogitBias(hidden_dim=2, temperature=0.1)
    previous = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]
    )
    current = torch.tensor(
        [[[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]]]
    )
    scores = module.normalized_scores(
        previous, current, torch.tensor([[69]]), range(69, 73)
    )
    assert scores.argmax(dim=-1).item() == 2
    torch.testing.assert_close(scores.mean(-1), torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        scores.square().mean(-1).sqrt(), torch.ones(1), atol=1e-6, rtol=0
    )


def test_zero_gate_is_exact_zero_and_only_parameter_is_gate():
    module = CausalFeatureTransportLogitBias(hidden_dim=3)
    scores = torch.randn(2, 4)
    bias = module.token_bias(scores, [2, 3, 4, 5], vocab_size=7)
    assert torch.count_nonzero(bias) == 0
    assert [(name, value.numel()) for name, value in module.named_parameters()] == [
        ("gate", 1)
    ]


def test_gate_has_direct_unit_rms_logit_effect_on_allowed_candidates():
    module = CausalFeatureTransportLogitBias(hidden_dim=3)
    module.gate.data.fill_(-0.2)
    normalized = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])
    bias = module.token_bias(normalized, [2, 3, 4, 5], vocab_size=7)
    assert torch.count_nonzero(bias[:, :2]) == 0
    assert torch.count_nonzero(bias[:, 6:]) == 0
    torch.testing.assert_close(
        bias[:, 2:6].square().mean(-1).sqrt(),
        torch.tensor([0.2]),
        atol=1e-6,
        rtol=0,
    )


def test_additive_processor_broadcasts_over_multi_token_heads():
    token_bias = torch.tensor([[0.0, 1.0, 2.0]])
    processor = AdditiveTokenBiasLogitsProcessor(token_bias)
    two_dimensional = processor(torch.empty(1, 0, dtype=torch.long), torch.zeros(1, 3))
    three_dimensional = processor(
        torch.empty(1, 0, dtype=torch.long), torch.zeros(1, 4, 3)
    )
    assert torch.equal(two_dimensional, token_bias)
    assert torch.equal(three_dimensional, token_bias[:, None].expand(-1, 4, -1))


def test_frame_bias_is_applied_to_each_aligned_prediction_head():
    module = CausalFeatureTransportLogitBias(hidden_dim=3)
    logits = torch.zeros(1, 8, 2, 5)
    frame_biases = torch.zeros(1, 2, 5)
    frame_biases[:, 1] = torch.arange(5)
    gaze_mask = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    relative = torch.tensor([-1, -1, -1, -2, -1, -1, -1, -2])
    result = module.apply_to_predictions(
        logits,
        frame_biases,
        gaze_mask,
        relative,
        torch.tensor([2, 2]),
    )
    assert torch.equal(result[:, 1, 0], torch.zeros(1, 5))
    assert torch.equal(result[:, 1, 1], torch.zeros(1, 5))
    assert torch.equal(result[:, 5, 0], frame_biases[:, 1])
    assert torch.equal(result[:, 5, 1], frame_biases[:, 1])
    assert torch.count_nonzero(result[:, [0, 2, 3, 4, 6, 7]]) == 0


def test_aligned_logit_path_provides_gate_gradient():
    module = CausalFeatureTransportLogitBias(hidden_dim=3)
    normalized = torch.tensor([[-1.0, 1.0]])
    frame_zero = torch.zeros(1, 3)
    frame_one = module.token_bias(normalized, [0, 1], vocab_size=3)
    result = module.apply_to_predictions(
        torch.zeros(1, 8, 2, 3),
        torch.stack((frame_zero, frame_one), dim=1),
        torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
        torch.tensor([-1, -1, -1, -2, -1, -1, -1, -2]),
        torch.tensor([2, 2]),
    )
    loss = result[:, 5, 0, 1] - result[:, 5, 0, 0]
    loss.backward()
    assert module.gate.grad.item() == pytest.approx(2.0)


def test_invalid_previous_tokens_produce_finite_zero_scores():
    module = CausalFeatureTransportLogitBias(hidden_dim=3)
    features = torch.randn(2, 5, 3)
    scores = module.normalized_scores(
        features,
        features,
        torch.full((2, 3), 265),
        range(69, 74),
    )
    assert torch.isfinite(scores).all()
    assert torch.count_nonzero(scores) == 0
