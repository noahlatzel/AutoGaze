import pytest
import torch
import torch.nn as nn

from autogaze.models.autogaze.modeling_autogaze import (
    SelectionConditionedRecurrentStateLogitBias,
)


ALLOWED = tuple(range(69, 73))


def module():
    torch.manual_seed(7)
    return SelectionConditionedRecurrentStateLogitBias(3, 4, 3)


def test_selected_summary_uses_only_completed_valid_fine_actions():
    recurrent = module()
    features = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]]
    )
    token_ids = torch.tensor([[69, 71, 265]])
    summary = recurrent.selected_summary(features, token_ids, ALLOWED)
    expected = torch.tensor([[0.5, 0.0, 1.5]])
    expected = expected / expected.square().mean(dim=-1, keepdim=True).sqrt()
    torch.testing.assert_close(summary, expected)
    torch.testing.assert_close(summary.square().mean(dim=-1).sqrt(), torch.ones(1))


def test_zero_gate_is_exact_control_identity_and_gate_is_zero_initialized():
    recurrent = module()
    state = torch.randn(2, 3)
    bias = recurrent.token_bias(state, ALLOWED, vocab_size=74)
    assert recurrent.gate.item() == 0.0
    assert torch.count_nonzero(bias) == 0


def test_state_updates_once_after_each_completed_frame():
    recurrent = module()

    class CountingCell(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, summary, state):
            self.calls += 1
            return state + summary

    cell = CountingCell()
    recurrent.state_cell = cell
    recurrent.gate.data.fill_(0.2)
    features = torch.zeros(1, 3, 4, 3)
    features[:, 0, 0, 0] = 1
    features[:, 1, 1, 1] = 2
    features[:, 2, 2, 2] = 3
    selections = [torch.tensor([[69]]), torch.tensor([[70]]), torch.tensor([[71]])]
    _, states = recurrent.sequence_biases(features, selections, ALLOWED, 74)
    assert cell.calls == 3
    torch.testing.assert_close(states[:, 0], torch.zeros(1, 3))
    root_three = 3**0.5
    torch.testing.assert_close(states[:, 1], torch.tensor([[root_three, 0.0, 0.0]]))
    torch.testing.assert_close(states[:, 2], torch.tensor([[root_three, root_three, 0.0]]))


def test_future_features_and_selections_cannot_change_earlier_biases():
    recurrent = module()
    recurrent.gate.data.fill_(0.2)
    features = torch.randn(2, 4, 4, 3)
    selections = [torch.tensor([[69], [70]]) for _ in range(4)]
    baseline, _ = recurrent.sequence_biases(features, selections, ALLOWED, 74)
    changed_features = features.clone()
    changed_features[:, 2:] = torch.randn_like(changed_features[:, 2:]) * 100
    changed_selections = list(selections)
    changed_selections[2] = torch.tensor([[72], [73]])
    changed_selections[3] = torch.tensor([[73], [72]])
    changed, _ = recurrent.sequence_biases(
        changed_features,
        changed_selections,
        ALLOWED,
        74,
    )
    torch.testing.assert_close(baseline[:, :3], changed[:, :3])
    assert not torch.equal(baseline[:, 3], changed[:, 3])


def test_reset_state_removes_all_recurrent_bias():
    recurrent = module()
    recurrent.gate.data.fill_(0.2)
    features = torch.randn(2, 3, 4, 3)
    selections = [torch.tensor([[69], [70]]) for _ in range(3)]
    biases, states = recurrent.sequence_biases(
        features,
        selections,
        ALLOWED,
        74,
        reset_each_frame=True,
    )
    assert torch.count_nonzero(biases) == 0
    assert torch.count_nonzero(states) == 0


def test_alternative_history_changes_only_post_update_state():
    recurrent = module()
    recurrent.gate.data.fill_(0.2)
    current = torch.randn(1, 3, 4, 3)
    alternative = current.clone()
    alternative[:, 0] += 10
    selections = [torch.tensor([[69, 70]]) for _ in range(3)]
    normal, _ = recurrent.sequence_biases(current, selections, ALLOWED, 74)
    changed, _ = recurrent.sequence_biases(
        current,
        selections,
        ALLOWED,
        74,
        history_features=alternative,
    )
    torch.testing.assert_close(normal[:, 0], changed[:, 0])
    assert not torch.equal(normal[:, 1], changed[:, 1])


def test_log_probability_path_reaches_gate_then_recurrent_parameters():
    recurrent = module()
    features = torch.randn(2, 3, 4, 3)
    selections = [torch.tensor([[69, 70], [70, 71]]) for _ in range(3)]

    biases, _ = recurrent.sequence_biases(features, selections, ALLOWED, 74)
    log_probs = torch.log_softmax(biases[:, 1:, list(ALLOWED)], dim=-1)
    loss = -log_probs[..., 0].mean()
    loss.backward()
    assert recurrent.gate.grad.abs().item() > 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in recurrent.state_cell.parameters()
    )

    recurrent.zero_grad(set_to_none=True)
    recurrent.gate.data.fill_(0.1)
    biases, _ = recurrent.sequence_biases(features, selections, ALLOWED, 74)
    loss = -torch.log_softmax(biases[:, 1:, list(ALLOWED)], dim=-1)[..., 0].mean()
    loss.backward()
    assert sum(
        parameter.grad.abs().sum().item()
        for parameter in recurrent.state_cell.parameters()
        if parameter.grad is not None
    ) > 0
    assert recurrent.action_readout.weight.grad.abs().sum().item() > 0


def test_state_and_bias_diagnostics_are_finite_and_bounded():
    recurrent = module()
    recurrent.gate.data.fill_(0.2)
    features = torch.randn(2, 5, 4, 3)
    selections = [torch.tensor([[69, 70], [70, 71]]) for _ in range(5)]
    biases, states = recurrent.sequence_biases(features, selections, ALLOWED, 74)
    recurrent.record_diagnostics(states, biases)
    diagnostics = recurrent.diagnostics()
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    assert diagnostics["recurrent_state_all_finite"].item() == 1.0
    assert diagnostics["recurrent_state_absolute_max"].item() <= 1.0


def test_action_mapping_rejects_noncontiguous_or_wrong_sized_ranges():
    recurrent = module()
    state = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        recurrent.token_bias(state, [69, 70, 72, 73], 74)
    with pytest.raises(ValueError):
        recurrent.token_bias(state, [69, 70, 71], 74)
