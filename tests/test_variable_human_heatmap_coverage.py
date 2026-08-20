# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from autogaze.tasks.variable_human_heatmap_coverage import (
    VariableHumanHeatmapCoverage,
)


def variable_outputs() -> dict[str, torch.Tensor]:
    # Two frame segments of width four. EOS=5 and the second frame is offset by 5.
    return {
        "gazing_pos": torch.tensor(
            [
                [1, 2, 5, 5, 6, 7, 8, 10],
                [1, 3, 4, 5, 6, 9, 10, 10],
            ]
        ),
        "if_padded_gazing": torch.tensor(
            [
                [False, False, False, True, False, False, False, False],
                [False, False, False, False, False, False, False, True],
            ]
        ),
        "num_gazing_each_frame": torch.tensor([4, 4]),
    }


def test_variable_task_computes_coverage_length_and_constrained_reward() -> None:
    task = VariableHumanHeatmapCoverage(
        clip_len=2,
        actions_per_frame=5,
        fine_action_offset=1,
        min_budget=1,
        max_budget=4,
        target_mean_budget=2.5,
        initial_token_cost=0.1,
        dual_lr=0.0,
        max_token_cost=0.2,
        report_budgets=(1, 2, 4),
    )
    task.eval()
    cell_mass = torch.tensor(
        [
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
        ]
    )
    result = task({"cell_mass": cell_mass}, variable_outputs())

    torch.testing.assert_close(result["per_sample_metrics"]["mean_tokens"], torch.tensor([2.5, 2.5]))
    torch.testing.assert_close(result["per_sample_metrics"]["coverage"], torch.tensor([0.65, 0.60]))
    torch.testing.assert_close(result["reward"], torch.tensor([[0.65], [0.60]]))
    assert result["traj_len_each_reward"] == [8]


def test_variable_task_updates_dual_only_while_training() -> None:
    task = VariableHumanHeatmapCoverage(
        clip_len=2,
        actions_per_frame=5,
        fine_action_offset=1,
        min_budget=1,
        max_budget=4,
        target_mean_budget=2.0,
        initial_token_cost=0.01,
        dual_lr=0.1,
        dual_ema_decay=0.0,
        max_token_cost=1.0,
        report_budgets=(1, 4),
    )
    inputs = {"cell_mass": torch.full((2, 2, 4), 0.25)}
    task.eval()
    task(inputs, variable_outputs())
    torch.testing.assert_close(task.token_cost, torch.tensor(0.01))
    task.train()
    task(inputs, variable_outputs())
    torch.testing.assert_close(task.token_cost, torch.tensor(0.06))


def test_variable_task_requests_fine_actions_plus_eos() -> None:
    task = VariableHumanHeatmapCoverage()
    assert task.gaze_model_kwargs == {
        "max_gaze_tokens_each_frame": 36,
        "min_gaze_tokens_each_frame": 4,
        "allowed_token_ids": list(range(69, 266)),
        "allow_eos": True,
    }
