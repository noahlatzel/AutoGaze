# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal metrics for aligned human-gaze cell-mass sequences."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def build_video_series(records: Sequence[Mapping], cell_mass: np.ndarray) -> dict:
    """Reconstruct ordered per-video target-grid sequences across clip boundaries."""
    frames = defaultdict(dict)
    for record in records:
        values = np.asarray(cell_mass[int(record["cell_mass_index"])], dtype=np.float64)
        start = int(record["target_start_index"])
        key = (record["source"], record["video_id"])
        for position, mass in enumerate(values):
            index = start + position
            if index in frames[key]:
                raise ValueError(f"Duplicate target index {index} for {key}")
            frames[key][index] = mass
    return {
        key: {
            "indices": np.asarray(sorted(by_index), dtype=np.int64),
            "mass": np.stack([by_index[index] for index in sorted(by_index)]),
        }
        for key, by_index in frames.items()
    }


def distribution_pair_metrics(left: np.ndarray, right: np.ndarray, topk=(16, 24, 32)) -> dict:
    """Return vectorized temporal metrics for paired distributions."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right must have matching [pairs,cells] shapes")
    eps = np.finfo(np.float64).tiny
    middle = 0.5 * (left + right)
    jsd = 0.5 * (
        (left * (np.log(left + eps) - np.log(middle + eps))).sum(axis=1)
        + (right * (np.log(right + eps) - np.log(middle + eps))).sum(axis=1)
    )
    cosine = (left * right).sum(axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), eps
    )
    grid_size = int(round(np.sqrt(left.shape[1])))
    rows, columns = np.divmod(np.arange(left.shape[1]), grid_size)
    left_centroid = np.stack([left @ columns, left @ rows], axis=1)
    right_centroid = np.stack([right @ columns, right @ rows], axis=1)
    result = {
        "jsd": jsd,
        "cosine": cosine,
        "centroid_cells": np.linalg.norm(right_centroid - left_centroid, axis=1),
    }
    for budget in topk:
        left_top = np.argpartition(left, -budget, axis=1)[:, -budget:]
        right_top = np.argpartition(right, -budget, axis=1)[:, -budget:]
        overlap = np.asarray(
            [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(left_top, right_top)]
        )
        result[f"top{budget}_jaccard"] = overlap / (2 * budget - overlap)
    return result


def lagged_pairs(series: Mapping, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair only samples whose target-grid indices differ by exactly lag."""
    if lag <= 0:
        raise ValueError("lag must be positive")
    lookup = {int(index): position for position, index in enumerate(series["indices"])}
    left_positions = []
    right_positions = []
    for position, index in enumerate(series["indices"]):
        target = int(index) + lag
        if target in lookup:
            left_positions.append(position)
            right_positions.append(lookup[target])
    return series["mass"][left_positions], series["mass"][right_positions]


def deterministic_nonadjacent_pairs(series: Mapping, minimum_lag: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic, long-lag within-video control pairing."""
    count = len(series["indices"])
    if count < 2:
        return series["mass"][:0], series["mass"][:0]
    shift = max(minimum_lag, count // 2)
    shift = min(shift, count - 1)
    positions = np.arange(count)
    targets = (positions + shift) % count
    valid = np.abs(series["indices"][targets] - series["indices"][positions]) >= minimum_lag
    return series["mass"][positions[valid]], series["mass"][targets[valid]]
