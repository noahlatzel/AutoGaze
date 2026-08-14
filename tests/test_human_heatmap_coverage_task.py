# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from autogaze.tasks.human_heatmap_coverage import HumanHeatmapCoverage


def test_human_coverage_task_rewards_selected_mass() -> None:
    task = HumanHeatmapCoverage(clip_len=2, exact_budget=2, report_budgets=(1, 2))
    local_actions = torch.tensor([[69, 70, 265 + 69, 265 + 70]])
    cell_mass = torch.zeros(1, 2, 196)
    cell_mass[0, 0, :2] = torch.tensor([0.25, 0.5])
    cell_mass[0, 1, :2] = torch.tensor([0.5, 0.5])

    outputs = task(
        {"cell_mass": cell_mass},
        {"gazing_pos": local_actions},
    )

    torch.testing.assert_close(outputs["reward"], torch.tensor([[0.875]]))
    torch.testing.assert_close(outputs["per_sample_metrics"]["coverage_k1"], torch.tensor([0.375]))
    assert outputs["traj_len_each_reward"] == [4]
    assert outputs["fine_cells"].shape == (1, 2, 2)


def test_human_coverage_task_requests_exact_fine_only_actions() -> None:
    task = HumanHeatmapCoverage()
    assert task.gaze_model_kwargs["max_gaze_tokens_each_frame"] == 16
    assert task.gaze_model_kwargs["allowed_token_ids"] == list(range(69, 265))
