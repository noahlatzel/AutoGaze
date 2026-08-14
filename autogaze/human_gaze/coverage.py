# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-grid human heatmap coverage metrics and fixed baselines."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES


def center_order(grid_size: int = 14) -> torch.Tensor:
    center = (grid_size - 1) / 2
    cells = list(range(grid_size * grid_size))
    cells.sort(
        key=lambda index: (
            (index // grid_size - center) ** 2 + (index % grid_size - center) ** 2,
            index // grid_size,
            index % grid_size,
        )
    )
    return torch.tensor(cells, dtype=torch.long)


def selected_coverage(cell_mass: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Return per-frame mass covered by selected cell indices."""
    if cell_mass.ndim != 2:
        raise ValueError("cell_mass must have shape [T, cells]")
    if selected.ndim == 1:
        selected = selected.unsqueeze(0).expand(cell_mass.shape[0], -1)
    if selected.shape[0] != cell_mass.shape[0]:
        raise ValueError("selected and cell_mass must contain the same number of frames")
    if selected.min() < 0 or selected.max() >= cell_mass.shape[1]:
        raise ValueError("selected cell index is out of range")
    if any(torch.unique(row).numel() != row.numel() for row in selected):
        raise ValueError("selected cells must not repeat within a frame")
    return cell_mass.gather(1, selected).sum(dim=1)


def global_positions_to_fine_cells(
    gazing_pos: torch.Tensor,
    num_frames: int = 16,
    exact_budget: int = 16,
    actions_per_frame: int = 265,
    fine_action_offset: int = 69,
) -> torch.Tensor:
    """Validate exact-budget global action IDs and convert them to 0..195 cells."""
    if gazing_pos.ndim != 2 or gazing_pos.shape[1] != num_frames * exact_budget:
        raise ValueError(
            f"Expected [B,{num_frames * exact_budget}] positions, got {tuple(gazing_pos.shape)}"
        )
    positions = gazing_pos.reshape(gazing_pos.shape[0], num_frames, exact_budget)
    offsets = torch.arange(num_frames, device=gazing_pos.device).reshape(1, num_frames, 1)
    local = positions - offsets * actions_per_frame
    if local.min() < fine_action_offset or local.max() >= actions_per_frame:
        raise ValueError("Generated action lies outside the fine-only vocabulary")
    fine = local - fine_action_offset
    for batch in fine:
        for frame in batch:
            if torch.unique(frame).numel() != exact_budget:
                raise ValueError("Generated fine actions repeat within a frame")
    return fine


class CoverageAccumulator:
    def __init__(self) -> None:
        self.clip_values: Dict[tuple, List[float]] = defaultdict(list)
        self.frame_sum: Dict[tuple, float] = defaultdict(float)
        self.frame_count: Dict[tuple, int] = defaultdict(int)

    def add(
        self,
        method: str,
        budget: int,
        source: str,
        video_id: str,
        frame_values: torch.Tensor,
    ) -> None:
        key = (method, budget, source, video_id)
        self.clip_values[key].append(float(frame_values.mean()))
        pooled_key = (method, budget)
        self.frame_sum[pooled_key] += float(frame_values.sum())
        self.frame_count[pooled_key] += frame_values.numel()

    def finalize(self, methods: Sequence[str], budgets: Sequence[int]) -> dict:
        result: Dict[str, dict] = {}
        for method in methods:
            result[method] = {}
            for budget in budgets:
                per_source: Dict[str, dict] = {}
                source_means: List[float] = []
                for source in STAVIS_SOURCES:
                    videos = {
                        video_id: values
                        for (cur_method, cur_budget, cur_source, video_id), values in self.clip_values.items()
                        if cur_method == method and cur_budget == budget and cur_source == source
                    }
                    video_means = {
                        video_id: float(np.mean(values)) for video_id, values in videos.items()
                    }
                    source_mean = float(np.mean(list(video_means.values())))
                    source_means.append(source_mean)
                    per_source[source] = {
                        "mean_video_coverage": source_mean,
                        "num_videos": len(video_means),
                        "num_clips": sum(len(values) for values in videos.values()),
                        "video_coverage": dict(sorted(video_means.items())),
                    }
                pooled_key = (method, budget)
                result[method][str(budget)] = {
                    "macro_source_mean": float(np.mean(source_means)),
                    "pooled_frame_mean": self.frame_sum[pooled_key] / self.frame_count[pooled_key],
                    "num_frames": self.frame_count[pooled_key],
                    "per_source": per_source,
                }
        return result


def compute_source_priors(dataset) -> Dict[str, torch.Tensor]:
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = defaultdict(int)
    for item in dataset:
        source = item["source"]
        mass = item["cell_mass"].sum(dim=0).to(torch.float64)
        sums[source] = mass if source not in sums else sums[source] + mass
        counts[source] += item["cell_mass"].shape[0]
    missing = set(STAVIS_SOURCES) - set(sums)
    if missing:
        raise ValueError(f"Training prior is missing sources: {sorted(missing)}")
    return {source: sums[source] / counts[source] for source in STAVIS_SOURCES}


def evaluate_coverage_baselines(
    dataset,
    source_priors: Mapping[str, torch.Tensor],
    budgets: Sequence[int] = (1, 2, 4, 8, 16),
    random_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    grid_size: int = 14,
) -> dict:
    num_cells = grid_size * grid_size
    if not budgets or min(budgets) <= 0 or max(budgets) > num_cells:
        raise ValueError("Budgets must lie between one and the number of cells")
    methods = ("random", "center", "prior", "oracle")
    accumulator = CoverageAccumulator()
    center = center_order(grid_size)
    generators = [torch.Generator().manual_seed(seed) for seed in random_seeds]

    for item in dataset:
        cell_mass = item["cell_mass"]
        source = item["source"]
        video_id = item["video_id"]
        prior_order = torch.argsort(source_priors[source], descending=True, stable=True)
        oracle_order = torch.argsort(cell_mass, dim=1, descending=True, stable=True)
        random_orders = torch.stack(
            [
                torch.stack([torch.randperm(num_cells, generator=generator) for _ in range(cell_mass.shape[0])])
                for generator in generators
            ]
        )

        for budget in budgets:
            random_values = torch.stack(
                [selected_coverage(cell_mass, order[:, :budget]) for order in random_orders]
            ).mean(dim=0)
            accumulator.add("random", budget, source, video_id, random_values)
            accumulator.add(
                "center", budget, source, video_id, selected_coverage(cell_mass, center[:budget])
            )
            accumulator.add(
                "prior", budget, source, video_id, selected_coverage(cell_mass, prior_order[:budget])
            )
            accumulator.add(
                "oracle", budget, source, video_id, selected_coverage(cell_mass, oracle_order[:, :budget])
            )

    return accumulator.finalize(methods, budgets)
