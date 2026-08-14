# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from autogaze.algorithms.grpo import GRPO


def _inputs_and_outputs(current_log_probs, reference_log_probs=None):
    gaze_outputs = {
        "log_action_probs": torch.zeros(2, 2, requires_grad=True),
        "if_padded_gazing": torch.zeros(2, 2, dtype=torch.bool),
        "action_log_probs_all": current_log_probs,
    }
    if reference_log_probs is not None:
        gaze_outputs["reference_action_log_probs_all"] = reference_log_probs
    task_outputs = {
        "reward": torch.tensor([[0.0], [1.0]]),
        "traj_len_each_reward": [2],
    }
    return {"group_size": 2}, gaze_outputs, task_outputs


def test_no_kl_arm_does_not_require_reference_distribution() -> None:
    log_probs = torch.log_softmax(torch.tensor([[[1.0, 0.0]]]).repeat(2, 2, 1), dim=-1)
    inputs, gaze_outputs, task_outputs = _inputs_and_outputs(log_probs)
    outputs = GRPO(group_size=2, discount_factor=1.0, kl_coefficient=0.0)(
        inputs, gaze_outputs, task_outputs
    )
    assert outputs["metrics"]["policy_kl"] == 0
    assert outputs["metrics"]["kl_loss"] == 0


def test_small_kl_matches_exact_categorical_kl() -> None:
    logits = torch.tensor([1.0, 0.0])
    current = torch.log_softmax(logits, dim=-1).reshape(1, 1, 2).repeat(2, 2, 1)
    reference = torch.log_softmax(logits.flip(0), dim=-1).reshape(1, 1, 2).repeat(2, 2, 1)
    inputs, gaze_outputs, task_outputs = _inputs_and_outputs(current, reference)
    outputs = GRPO(group_size=2, discount_factor=1.0, kl_coefficient=0.1)(
        inputs, gaze_outputs, task_outputs
    )
    expected = (current.exp() * (current - reference)).sum(dim=-1).mean()
    torch.testing.assert_close(outputs["metrics"]["policy_kl"], expected)
    torch.testing.assert_close(outputs["metrics"]["kl_loss"], 0.1 * expected)
