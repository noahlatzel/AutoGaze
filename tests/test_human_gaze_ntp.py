# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math

import pytest
import torch

from autogaze.algorithms import HumanGazeNTP


def toy_algorithm(**kwargs):
    options = dict(clip_len=2, actions_per_frame=5, fine_action_offset=1, exact_budget=3, teacher_seed=314)
    options.update(kwargs)
    return HumanGazeNTP(**options)


def with_teacher(algorithm, mass, fine):
    fine = torch.tensor(fine, dtype=torch.long)
    offsets = torch.arange(algorithm.clip_len).reshape(1, -1, 1) * algorithm.actions_per_frame
    positions = (fine + offsets + algorithm.fine_action_offset).reshape(fine.shape[0], -1)
    return {
        "cell_mass": torch.tensor(mass, dtype=torch.float64),
        "gt_gazing_info": {
            "gazing_pos": positions,
            "num_gazing_each_frame": torch.full((algorithm.clip_len,), algorithm.exact_budget, dtype=torch.long),
            "if_padded_gazing": torch.zeros_like(positions, dtype=torch.bool),
        },
    }


def score_teacher(algorithm, inputs, logits=None):
    targets, _, available = algorithm.conditional_targets(inputs)
    if logits is None:
        logits = torch.zeros_like(targets, requires_grad=True)
    output = dict(inputs["gt_gazing_info"])
    output["supervised_action_log_probs_all"] = torch.log_softmax(logits.masked_fill(~available, -torch.inf), dim=-1)
    return output, logits


def test_conditional_target_excludes_previous_not_current_and_resets_each_frame():
    algorithm = toy_algorithm()
    inputs = with_teacher(algorithm, [[[1, 2, 3, 4], [4, 3, 2, 1]]], [[[2, 0, 3], [2, 0, 3]]])
    targets, valid, available = algorithm.conditional_targets(inputs)
    expected = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [1 / 7, 2 / 7, 0, 4 / 7],
        [0, 1 / 3, 0, 2 / 3],
        [0.4, 0.3, 0.2, 0.1],
        [0.5, 3 / 8, 0, 1 / 8],
        [0, 0.75, 0, 0.25],
    ], dtype=torch.float64).unsqueeze(0)
    torch.testing.assert_close(targets, expected)
    assert valid.all()
    assert available[0, 0, 2] and not available[0, 1, 2]
    assert available[0, 3].all()


def test_real_contract_exact_k16_global_offsets_and_no_special_or_repeat_actions():
    algorithm = HumanGazeNTP(teacher_seed=440826)
    inputs = algorithm.preprocess_inputs({"cell_mass": torch.ones(2, 16, 196)})
    info = inputs["gt_gazing_info"]
    assert info["gazing_pos"].shape == (2, 256)
    assert info["num_gazing_each_frame"].tolist() == [16] * 16
    assert info["if_padded_gazing"].dtype == torch.bool
    assert not info["if_padded_gazing"].any()
    local = info["gazing_pos"].reshape(2, 16, 16) - torch.arange(16).reshape(1, 16, 1) * 265
    assert local.min() >= 69 and local.max() <= 264
    assert all(row.unique().numel() == 16 for row in local.reshape(-1, 16))


def test_exhausted_mass_finishes_legal_history_without_inventing_uniform_targets():
    algorithm = toy_algorithm()
    inputs = algorithm.preprocess_inputs({"cell_mass": torch.tensor([[[1., 0, 0, 0], [0., 0, 1, 0]]])})
    fine = algorithm._validate_trajectory(inputs["gt_gazing_info"], 1)
    assert fine[0, 0, 0] == 0 and fine[0, 1, 0] == 2
    targets, valid, _ = algorithm.conditional_targets(inputs)
    assert valid.tolist() == [[True, False, False, True, False, False]]
    assert torch.equal(targets[~valid], torch.zeros_like(targets[~valid]))
    outputs, logits = score_teacher(algorithm, inputs)
    result = algorithm(inputs, outputs)
    torch.testing.assert_close(result["loss"], torch.tensor([math.log(4)], dtype=torch.float64))
    assert result["metrics"]["supervised_batch_valid_actions"] == 2
    assert result["metrics"]["supervised_batch_exhausted_actions"] == 4
    result["loss"].sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[~valid], torch.zeros_like(logits.grad[~valid]))


def test_known_forward_kl_cross_entropy_and_gradient():
    algorithm = HumanGazeNTP(clip_len=1, actions_per_frame=3, fine_action_offset=1, exact_budget=1)
    inputs = with_teacher(algorithm, [[[3, 1]]], [[[0]]])
    outputs, logits = score_teacher(algorithm, inputs)
    result = algorithm(inputs, outputs)
    entropy = -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    torch.testing.assert_close(result["loss"], torch.tensor([math.log(2)], dtype=torch.float64))
    assert result["metrics"]["supervised_forward_kl"].item() == pytest.approx(math.log(2) - entropy)
    result["loss"].sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[-0.25, 0.25]]], dtype=torch.float64))


def test_zero_gradient_anchor_keeps_task_head_out_of_training_objective():
    algorithm = toy_algorithm()
    inputs = algorithm.preprocess_inputs({"cell_mass": torch.ones(1, 2, 4)})
    outputs, _ = score_teacher(algorithm, inputs)
    without_head = algorithm(inputs, outputs)
    prediction = torch.tensor([[123.0, -7.0]], requires_grad=True)
    outputs["task_loss_prediction"] = prediction
    with_head = algorithm(inputs, outputs)
    torch.testing.assert_close(with_head["loss"], without_head["loss"])
    with_head["loss"].sum().backward()
    assert prediction.grad is not None
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_loss_averages_valid_actions_per_clip_before_averaging_clips():
    algorithm = toy_algorithm(clip_len=1)
    inputs = with_teacher(algorithm, [[[1, 0, 0, 0]], [[1, 1, 1, 1]]], [[[0, 1, 2]], [[0, 1, 2]]])
    outputs, _ = score_teacher(algorithm, inputs)
    result = algorithm(inputs, outputs)
    expected = torch.tensor([math.log(4), (math.log(4) + math.log(3) + math.log(2)) / 3], dtype=torch.float64)
    torch.testing.assert_close(result["loss"], expected)
    torch.testing.assert_close(result["metrics"]["supervised_soft_ce"], expected.mean())


def test_teacher_rng_is_local_and_round_trip_restores_next_history_exactly():
    inputs = {"cell_mass": torch.ones(5, 2, 4)}
    algorithm = toy_algorithm()
    global_before = torch.random.get_rng_state().clone()
    algorithm.preprocess_inputs(inputs)
    snapshot = algorithm.state_dict()
    expected = algorithm.preprocess_inputs(inputs)["gt_gazing_info"]["gazing_pos"]
    assert torch.equal(torch.random.get_rng_state(), global_before)
    resumed = toy_algorithm(teacher_seed=999)
    resumed.load_state_dict(snapshot)
    actual = resumed.preprocess_inputs(inputs)["gt_gazing_info"]["gazing_pos"]
    assert torch.equal(actual, expected)
    assert resumed.teacher_seed == 314
    assert algorithm.teacher_generator.device.type == "cpu"
    assert snapshot["teacher_rng_state"].device.type == "cpu"


def test_rng_restore_rejects_changed_contract_or_invalid_rng_without_mutation():
    algorithm = toy_algorithm()
    before = algorithm.state_dict()
    changed = copy.deepcopy(before)
    changed["contract"]["exact_budget"] = 2
    with pytest.raises(ValueError, match="different action contract"):
        algorithm.load_state_dict(changed)
    bad_rng = copy.deepcopy(before)
    bad_rng["teacher_rng_state"] = torch.ones(2, dtype=torch.uint8)
    with pytest.raises(ValueError, match="Invalid teacher RNG"):
        algorithm.load_state_dict(bad_rng)
    assert torch.equal(before["teacher_rng_state"], algorithm.state_dict()["teacher_rng_state"])


@pytest.mark.parametrize("mass", [
    torch.ones(1, 2, 3), torch.ones(1, 3, 4), torch.ones(2, 4), torch.ones(0, 2, 4),
    torch.ones(1, 2, 4, dtype=torch.int64), torch.full((1, 2, 4), -1.),
    torch.full((1, 2, 4), torch.nan), torch.full((1, 2, 4), torch.inf),
    torch.zeros(1, 2, 4),
])
def test_malformed_human_mass_fails_before_teacher_sampling(mass):
    algorithm = toy_algorithm()
    state = algorithm.state_dict()["teacher_rng_state"]
    with pytest.raises(ValueError):
        algorithm.preprocess_inputs({"cell_mass": mass})
    assert torch.equal(state, algorithm.state_dict()["teacher_rng_state"])


@pytest.mark.parametrize("mutation", ["duplicate", "coarse", "wrong_frame", "short", "counts_2d", "wrong_count", "padded", "float_positions"])
def test_malformed_fixed_teacher_trajectory_is_rejected(mutation):
    algorithm = toy_algorithm()
    inputs = algorithm.preprocess_inputs({"cell_mass": torch.ones(1, 2, 4)})
    info = inputs["gt_gazing_info"]
    if mutation == "duplicate":
        info["gazing_pos"][0, 1] = info["gazing_pos"][0, 0]
    elif mutation == "coarse":
        info["gazing_pos"][0, 0] = 0
    elif mutation == "wrong_frame":
        info["gazing_pos"][0, 3] -= 5
    elif mutation == "short":
        info["gazing_pos"] = info["gazing_pos"][:, :-1]
    elif mutation == "counts_2d":
        info["num_gazing_each_frame"] = info["num_gazing_each_frame"].unsqueeze(0)
    elif mutation == "wrong_count":
        info["num_gazing_each_frame"][0] = 2
    elif mutation == "padded":
        info["if_padded_gazing"][0, 0] = True
    elif mutation == "float_positions":
        info["gazing_pos"] = info["gazing_pos"].to(torch.float32)
    with pytest.raises(ValueError):
        algorithm.conditional_targets(inputs)


@pytest.mark.parametrize("mutation", ["legacy_only", "shape", "not_normalized", "nan", "available_neg_inf", "masked_finite", "different_trajectory"])
def test_scoring_requires_true_normalized_distribution_and_matching_trajectory(mutation):
    algorithm = toy_algorithm()
    inputs = algorithm.preprocess_inputs({"cell_mass": torch.ones(1, 2, 4)})
    outputs, _ = score_teacher(algorithm, inputs)
    log_probs = outputs["supervised_action_log_probs_all"].detach().clone()
    outputs["supervised_action_log_probs_all"] = log_probs
    if mutation == "legacy_only":
        outputs["action_log_probs_all"] = outputs.pop("supervised_action_log_probs_all")
    elif mutation == "shape":
        outputs["supervised_action_log_probs_all"] = log_probs[..., :-1]
    elif mutation == "not_normalized":
        outputs["supervised_action_log_probs_all"] = log_probs + 0.1
    elif mutation == "nan":
        log_probs[0, 0, 0] = torch.nan
    elif mutation == "available_neg_inf":
        log_probs[0, 0, 0] = -torch.inf
    elif mutation == "masked_finite":
        log_probs[~torch.isfinite(log_probs)] = -100
    elif mutation == "different_trajectory":
        positions = outputs["gazing_pos"].clone()
        positions[0, :2] = positions[0, :2].flip(0)
        outputs["gazing_pos"] = positions
    with pytest.raises(ValueError):
        algorithm(inputs, outputs)
