# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thumbnail-based temporal proxy metrics.

The functions in this module score visual evidence only. Action costs and label
selection are kept separate so the same metric can be reused with different
rate-distortion tradeoffs.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


PAIRWISE_METRICS = ("pixel_mse", "ssim_distance", "edge_l1")
WINDOW_METRICS = ("mean_novelty", "within_diversity", "mean_pairwise_change")
TEMPORAL_COSTS = ("none", "linear", "log")


def thumbnail_tensor(frames: np.ndarray | torch.Tensor, size: int) -> torch.Tensor:
    """Convert RGB frames to resized thumbnail tensor ``(T, C, H, W)`` in [0, 1]."""
    if isinstance(frames, np.ndarray):
        tensor = torch.from_numpy(frames)
    else:
        tensor = frames.detach().cpu()

    if tensor.ndim != 4:
        raise ValueError(f"Expected frames with shape (T,H,W,C) or (T,C,H,W), got {tuple(tensor.shape)}")

    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.shape[1] not in (1, 3):
        raise ValueError(f"Could not infer channel dimension from shape {tuple(tensor.shape)}")

    tensor = tensor.float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    tensor = F.interpolate(tensor, size=(size, size), mode="bicubic", align_corners=False)
    return tensor.clamp(0, 1)


def pairwise_distance(first: torch.Tensor, second: torch.Tensor, metric: str) -> torch.Tensor:
    """Score one pair of thumbnail frames.

    Args:
        first: Tensor shaped ``(C, H, W)`` in [0, 1].
        second: Tensor shaped ``(C, H, W)`` in [0, 1].
        metric: One of ``PAIRWISE_METRICS``.
    """
    if metric == "pixel_mse":
        return (first - second).pow(2).mean()
    if metric == "ssim_distance":
        return 1.0 - _global_ssim(first, second)
    if metric == "edge_l1":
        return (_edge_magnitude(first) - _edge_magnitude(second)).abs().mean()
    raise ValueError(f"Unknown pairwise metric: {metric}")


def pairwise_distance_matrix(current: torch.Tensor, candidate: torch.Tensor, metric: str) -> torch.Tensor:
    """Return pairwise distances between two chunks.

    Args:
        current: Tensor shaped ``(T, C, H, W)``.
        candidate: Tensor shaped ``(U, C, H, W)``.
    Returns:
        Tensor shaped ``(T, U)``.
    """
    if current.ndim != 4 or candidate.ndim != 4:
        raise ValueError("current and candidate must both have shape (T,C,H,W)")
    if current.shape[1:] != candidate.shape[1:]:
        raise ValueError(f"Frame shapes differ: {tuple(current.shape[1:])} vs {tuple(candidate.shape[1:])}")

    if metric == "pixel_mse":
        diff = current[:, None] - candidate[None]
        return diff.pow(2).mean(dim=(2, 3, 4))

    values = [
        [pairwise_distance(current_i, candidate_j, metric) for candidate_j in candidate]
        for current_i in current
    ]
    return torch.stack([torch.stack(row) for row in values])


def score_window_metric(
    current: torch.Tensor,
    candidate: torch.Tensor,
    metric: str,
    *,
    base_metric: str = "pixel_mse",
) -> torch.Tensor:
    """Score a candidate chunk against the current chunk."""
    if metric == "mean_novelty":
        distances = pairwise_distance_matrix(current, candidate, base_metric)
        return distances.min(dim=0).values.mean()

    if metric == "within_diversity":
        if candidate.shape[0] < 2:
            return candidate.new_tensor(0.0)
        distances = pairwise_distance_matrix(candidate, candidate, base_metric)
        row, col = torch.triu_indices(candidate.shape[0], candidate.shape[0], offset=1)
        return distances[row, col].mean()

    if metric == "mean_pairwise_change":
        if current.shape[0] != candidate.shape[0]:
            raise ValueError(
                "mean_pairwise_change requires equal chunk lengths, "
                f"got {current.shape[0]} and {candidate.shape[0]}"
            )
        return torch.stack(
            [pairwise_distance(first, second, base_metric) for first, second in zip(current, candidate)]
        ).mean()

    raise ValueError(f"Unknown window metric: {metric}")


def temporal_cost(action: int | float, cost: str) -> float:
    """Return a scalar action cost."""
    if action < 0:
        raise ValueError(f"Action cost is undefined for negative action: {action}")
    if cost == "none":
        return 0.0
    if cost == "linear":
        return float(action)
    if cost == "log":
        return float(np.log1p(action))
    raise ValueError(f"Unknown temporal cost: {cost}")


def select_best_action(
    scores: Mapping[int, float],
    *,
    lambda_cost: float = 0.0,
    cost: str = "none",
) -> tuple[int, dict[int, float]]:
    """Select the action with max ``score - lambda_cost * temporal_cost``."""
    if not scores:
        raise ValueError("Cannot select an action from empty scores")

    utilities = {
        int(action): float(score) - lambda_cost * temporal_cost(int(action), cost)
        for action, score in scores.items()
    }
    best = max(utilities, key=lambda action: (utilities[action], -action))
    return best, utilities


def _global_ssim(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_gray = _to_gray(first)
    second_gray = _to_gray(second)
    c1 = 0.01**2
    c2 = 0.03**2

    mu_first = first_gray.mean()
    mu_second = second_gray.mean()
    var_first = (first_gray - mu_first).pow(2).mean()
    var_second = (second_gray - mu_second).pow(2).mean()
    covariance = ((first_gray - mu_first) * (second_gray - mu_second)).mean()

    numerator = (2 * mu_first * mu_second + c1) * (2 * covariance + c2)
    denominator = (mu_first.pow(2) + mu_second.pow(2) + c1) * (var_first + var_second + c2)
    return (numerator / denominator).clamp(0, 1)


def _edge_magnitude(frame: torch.Tensor) -> torch.Tensor:
    gray = _to_gray(frame).unsqueeze(0).unsqueeze(0)
    kernel_x = frame.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = frame.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12).squeeze(0).squeeze(0)


def _to_gray(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim != 3:
        raise ValueError(f"Expected frame shape (C,H,W), got {tuple(frame.shape)}")
    if frame.shape[0] == 1:
        return frame[0]
    if frame.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {frame.shape[0]}")
    weights = frame.new_tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
    return (frame * weights).sum(dim=0)
