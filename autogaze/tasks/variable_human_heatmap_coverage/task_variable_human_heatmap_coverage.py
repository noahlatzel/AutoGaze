# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-only human-gaze coverage with learned EOS and a mean-token constraint."""

from __future__ import annotations

import torch
from torch import nn


class VariableHumanHeatmapCoverage(nn.Module):
    """Reward coverage while adapting a linear token price to a target mean K."""

    def __init__(
        self,
        clip_len: int = 16,
        actions_per_frame: int = 265,
        fine_action_offset: int = 69,
        min_budget: int = 4,
        max_budget: int = 36,
        target_mean_budget: float = 16.0,
        initial_token_cost: float = 0.01,
        dual_lr: float = 1.0e-5,
        dual_ema_decay: float = 0.95,
        max_token_cost: float = 0.1,
        report_budgets=(1, 4, 8, 16, 24, 36),
    ) -> None:
        super().__init__()
        self.clip_len = int(clip_len)
        self.actions_per_frame = int(actions_per_frame)
        self.fine_action_offset = int(fine_action_offset)
        self.min_budget = int(min_budget)
        self.max_budget = int(max_budget)
        self.target_mean_budget = float(target_mean_budget)
        self.dual_lr = float(dual_lr)
        self.dual_ema_decay = float(dual_ema_decay)
        self.max_token_cost = float(max_token_cost)
        self.report_budgets = tuple(int(value) for value in report_budgets)

        num_fine_actions = self.actions_per_frame - self.fine_action_offset
        if not 0 <= self.min_budget <= self.max_budget <= num_fine_actions:
            raise ValueError("Budgets must satisfy 0 <= min <= max <= fine actions")
        if not self.min_budget <= self.target_mean_budget <= self.max_budget:
            raise ValueError("target_mean_budget must lie within [min_budget,max_budget]")
        if self.dual_lr < 0:
            raise ValueError("dual_lr must be non-negative")
        if not 0 <= self.dual_ema_decay < 1:
            raise ValueError("dual_ema_decay must lie in [0,1)")
        if not 0 <= initial_token_cost <= self.max_token_cost:
            raise ValueError("initial_token_cost must lie within its clipping range")
        if not self.report_budgets or min(self.report_budgets) <= 0:
            raise ValueError("report_budgets must contain positive integers")
        if max(self.report_budgets) > self.max_budget:
            raise ValueError("report_budgets cannot exceed max_budget")

        self.register_buffer(
            "token_cost",
            torch.tensor(float(initial_token_cost)),
            persistent=True,
        )
        self.register_buffer(
            "mean_tokens_ema",
            torch.tensor(float(target_mean_budget)),
            persistent=True,
        )
        self.scales = "224"
        self.transform = None
        self.gaze_model_kwargs = {
            "max_gaze_tokens_each_frame": self.max_budget,
            "min_gaze_tokens_each_frame": self.min_budget,
            "allowed_token_ids": list(
                range(self.fine_action_offset, self.actions_per_frame + 1)
            ),
            "allow_eos": True,
        }

    def _split_framewise(self, values: torch.Tensor, lengths: torch.Tensor):
        frame_widths = [int(length) for length in lengths.tolist()]
        if len(frame_widths) != self.clip_len:
            raise ValueError(
                f"Expected {self.clip_len} frame segments, got {len(frame_widths)}"
            )
        if sum(frame_widths) != values.shape[1]:
            raise ValueError("Frame segment lengths do not match trajectory length")
        return list(values.split(frame_widths, dim=1))

    @torch.no_grad()
    def _update_token_cost(self, observed_mean: torch.Tensor) -> None:
        observed = observed_mean.detach().to(self.mean_tokens_ema)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(observed, torch.distributed.ReduceOp.SUM)
            observed /= torch.distributed.get_world_size()
        self.mean_tokens_ema.mul_(self.dual_ema_decay).add_(
            observed * (1.0 - self.dual_ema_decay)
        )
        self.token_cost.add_(
            self.dual_lr * (self.mean_tokens_ema - self.target_mean_budget)
        ).clamp_(min=0.0, max=self.max_token_cost)

    def forward(self, inputs, gaze_outputs):
        cell_mass = inputs["cell_mass"]
        if cell_mass.ndim != 3 or cell_mass.shape[1] != self.clip_len:
            raise ValueError("cell_mass must have shape [B,clip_len,fine_cells]")

        positions = self._split_framewise(
            gaze_outputs["gazing_pos"], gaze_outputs["num_gazing_each_frame"]
        )
        padded = self._split_framewise(
            gaze_outputs["if_padded_gazing"],
            gaze_outputs["num_gazing_each_frame"],
        )

        frame_coverage = []
        frame_lengths = []
        frame_eos = []
        frame_prefix = {budget: [] for budget in self.report_budgets}
        for frame_index, (global_ids, frame_padded) in enumerate(
            zip(positions, padded)
        ):
            local_ids = global_ids - frame_index * self.actions_per_frame
            spatial = (
                (local_ids >= self.fine_action_offset)
                & (local_ids < self.actions_per_frame)
                & ~frame_padded
            )
            safe_cells = (local_ids - self.fine_action_offset).clamp(
                min=0,
                max=cell_mass.shape[-1] - 1,
            )
            selected_mass = cell_mass[:, frame_index].gather(1, safe_cells)
            selected_mass = selected_mass * spatial.to(selected_mass.dtype)
            frame_coverage.append(selected_mass.sum(dim=1))
            frame_lengths.append(spatial.sum(dim=1).to(cell_mass.dtype))
            frame_eos.append(
                (
                    (local_ids == self.actions_per_frame)
                    & ~frame_padded
                ).any(dim=1).to(cell_mass.dtype)
            )
            for budget in self.report_budgets:
                frame_prefix[budget].append(selected_mass[:, :budget].sum(dim=1))

        per_frame_coverage = torch.stack(frame_coverage, dim=1)
        per_frame_lengths = torch.stack(frame_lengths, dim=1)
        clip_coverage = per_frame_coverage.mean(dim=1)
        mean_tokens = per_frame_lengths.mean(dim=1)
        current_token_cost = self.token_cost.detach().clone()
        constrained_reward = clip_coverage - current_token_cost * (
            mean_tokens - self.target_mean_budget
        )

        per_sample = {
            "coverage": clip_coverage,
            "mean_tokens": mean_tokens,
            "length_std": per_frame_lengths.std(dim=1, unbiased=False),
            "eos_rate": torch.stack(frame_eos, dim=1).mean(dim=1),
            "min_rate": (per_frame_lengths == self.min_budget)
            .to(cell_mass.dtype)
            .mean(dim=1),
            "cap_rate": (per_frame_lengths == self.max_budget)
            .to(cell_mass.dtype)
            .mean(dim=1),
            "constrained_reward": constrained_reward,
        }
        for budget, values in frame_prefix.items():
            per_sample[f"coverage_prefix_k{budget}"] = torch.stack(
                values, dim=1
            ).mean(dim=1)

        metrics = {key: value.mean() for key, value in per_sample.items()}
        metrics["token_cost"] = current_token_cost
        metrics["budget_error"] = metrics["mean_tokens"] - self.target_mean_budget
        metrics["mean_tokens_ema"] = self.mean_tokens_ema.detach().clone()

        if self.training:
            self._update_token_cost(metrics["mean_tokens"])

        return {
            "loss": torch.zeros_like(clip_coverage),
            "reward": constrained_reward.unsqueeze(1),
            "traj_len_each_reward": [gaze_outputs["gazing_pos"].shape[1]],
            "metrics": metrics,
            "per_sample_metrics": {
                key: value.detach() for key, value in per_sample.items()
            },
            "per_frame_lengths": per_frame_lengths.detach(),
            "per_frame_coverage": per_frame_coverage.detach(),
        }

    def visualize(self, inputs, gaze_outputs, task_outputs):
        return None
