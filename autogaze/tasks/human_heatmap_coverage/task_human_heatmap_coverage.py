# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parameter-free clip-average human heatmap coverage task."""

import torch
from torch import nn

from autogaze.human_gaze.coverage import (
    global_positions_to_fine_cells,
    selected_coverage,
)


class HumanHeatmapCoverage(nn.Module):
    def __init__(
        self,
        clip_len=16,
        actions_per_frame=265,
        fine_action_offset=69,
        exact_budget=16,
        report_budgets=(1, 2, 4, 8, 16),
    ):
        super().__init__()
        self.clip_len = clip_len
        self.actions_per_frame = actions_per_frame
        self.fine_action_offset = fine_action_offset
        self.exact_budget = exact_budget
        self.report_budgets = tuple(int(budget) for budget in report_budgets)
        if not self.report_budgets or min(self.report_budgets) <= 0:
            raise ValueError("report_budgets must contain positive integers")
        if max(self.report_budgets) > self.exact_budget:
            raise ValueError("report_budgets cannot exceed exact_budget")
        self.scales = "224"
        self.transform = None
        self.gaze_model_kwargs = {
            "max_gaze_tokens_each_frame": exact_budget,
            "allowed_token_ids": list(range(fine_action_offset, actions_per_frame)),
        }

    def forward(self, inputs, gaze_outputs):
        fine_cells = global_positions_to_fine_cells(
            gaze_outputs["gazing_pos"],
            num_frames=self.clip_len,
            exact_budget=self.exact_budget,
            actions_per_frame=self.actions_per_frame,
            fine_action_offset=self.fine_action_offset,
        )
        cell_mass = inputs["cell_mass"]
        clip_coverages = {}
        for budget in self.report_budgets:
            per_frame = torch.stack(
                [
                    selected_coverage(cell_mass[index], fine_cells[index, :, :budget])
                    for index in range(cell_mass.shape[0])
                ]
            )
            clip_coverages[budget] = per_frame.mean(dim=1)
        clip_coverage = clip_coverages[self.exact_budget]
        metrics = {
            f"coverage_k{budget}": values.mean()
            for budget, values in clip_coverages.items()
        }
        metrics["coverage"] = clip_coverage.mean()
        per_sample_metrics = {
            f"coverage_k{budget}": values.detach()
            for budget, values in clip_coverages.items()
        }
        per_sample_metrics["coverage"] = clip_coverage.detach()
        return {
            "loss": torch.zeros_like(clip_coverage),
            "reward": clip_coverage.unsqueeze(1),
            "traj_len_each_reward": [self.clip_len * self.exact_budget],
            "metrics": metrics,
            "per_sample_metrics": per_sample_metrics,
            "fine_cells": fine_cells,
        }

    def visualize(self, inputs, gaze_outputs, task_outputs):
        return None
