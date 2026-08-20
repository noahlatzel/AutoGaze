"""Adapters for evaluating fixed-budget human-gaze policies in NVILA."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch


@dataclass
class FixedBudgetCallStats:
    """Streaming summary of retained patches after resolution adaptation."""

    model_calls: int = 0
    frame_observations: int = 0
    retained_patch_sum: int = 0
    retained_patch_min: int | None = None
    retained_patch_max: int | None = None

    def update(self, output: dict[str, Any]) -> None:
        self.model_calls += 1
        counts = output.get("num_gazing_each_frame")
        if counts is None:
            return
        values = torch.as_tensor(counts).detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if values.numel() == 0:
            return
        current_min = int(values.min().item())
        current_max = int(values.max().item())
        self.frame_observations += int(values.numel())
        self.retained_patch_sum += int(values.sum().item())
        self.retained_patch_min = (
            current_min if self.retained_patch_min is None else min(self.retained_patch_min, current_min)
        )
        self.retained_patch_max = (
            current_max if self.retained_patch_max is None else max(self.retained_patch_max, current_max)
        )

    def as_dict(self) -> dict[str, int | float | None]:
        mean = None
        if self.frame_observations:
            mean = self.retained_patch_sum / self.frame_observations
        return {
            "model_calls": self.model_calls,
            "frame_observations": self.frame_observations,
            "retained_patches_per_frame_mean": mean,
            "retained_patches_per_frame_min": self.retained_patch_min,
            "retained_patches_per_frame_max": self.retained_patch_max,
        }


def install_fixed_budget_forward(
    model: torch.nn.Module,
    *,
    exact_budget: int,
    fine_action_offset: int = 69,
    actions_per_frame: int = 265,
) -> FixedBudgetCallStats:
    """Force fine-only exact-K generation for calls made by the NVILA processor.

    NVILA normally calls AutoGaze with a gazing ratio and an adaptive task-loss
    threshold. Human-gaze K24/K36 policies were trained with a different action
    contract: exactly K non-repeating fine actions and no EOS. This adapter
    replaces only those generation arguments while retaining NVILA's normal
    resolution adaptation and patch mapping.
    """

    if exact_budget <= 0:
        raise ValueError("exact_budget must be positive")
    if fine_action_offset < 0 or fine_action_offset >= actions_per_frame:
        raise ValueError("fine_action_offset must lie inside the action vocabulary")
    if exact_budget > actions_per_frame - fine_action_offset:
        raise ValueError("exact_budget exceeds the number of available fine actions")

    allowed_token_ids = tuple(range(fine_action_offset, actions_per_frame))
    original_forward = model.forward
    stats = FixedBudgetCallStats()

    def fixed_forward(_model, inputs, *args, **kwargs):
        kwargs["gazing_ratio"] = None
        kwargs["task_loss_requirement"] = None
        kwargs["max_gaze_tokens_each_frame"] = exact_budget
        kwargs["allowed_token_ids"] = allowed_token_ids
        kwargs["allow_eos"] = False
        output = original_forward(inputs, *args, **kwargs)
        stats.update(output)
        return output

    model.forward = MethodType(fixed_forward, model)
    return stats
