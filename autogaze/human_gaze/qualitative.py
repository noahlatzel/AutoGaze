# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic ground-truth selection for qualitative gaze examples."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from autogaze.human_gaze.coverage import center_order


def strongest_off_center_window(
    cell_mass: np.ndarray,
    frame_count: int,
    center_budget: int = 32,
    grid_size: int = 14,
) -> tuple[int, float, np.ndarray]:
    """Find the consecutive window with most mass outside a fixed center region."""
    mass = np.asarray(cell_mass)
    if mass.ndim != 2 or mass.shape[1] != grid_size * grid_size:
        raise ValueError("cell_mass must have shape [frames, grid_size**2]")
    if not 1 <= frame_count <= mass.shape[0]:
        raise ValueError("frame_count must lie between one and the clip length")
    if not 1 <= center_budget <= mass.shape[1]:
        raise ValueError("center_budget must lie between one and the number of cells")

    center = center_order(grid_size)[:center_budget].numpy()
    outside = 1.0 - mass[:, center].sum(axis=1)
    window_scores = np.convolve(
        outside, np.ones(frame_count, dtype=np.float64) / frame_count, mode="valid"
    )
    start = int(np.argmax(window_scores))
    return start, float(window_scores[start]), outside


def select_off_center_examples(
    records: Sequence[Mapping],
    cell_mass: np.ndarray,
    num_clips: int = 3,
    frame_count: int = 4,
    center_budget: int = 32,
    grid_size: int = 14,
) -> list[dict]:
    """Select strongest off-center clips, using at most one clip per source."""
    if num_clips <= 0:
        raise ValueError("num_clips must be positive")

    candidates = []
    for dataset_index, record in enumerate(records):
        cache_index = int(record["cell_mass_index"])
        start, score, outside = strongest_off_center_window(
            cell_mass[cache_index],
            frame_count=frame_count,
            center_budget=center_budget,
            grid_size=grid_size,
        )
        candidates.append(
            {
                "dataset_index": dataset_index,
                "clip_id": record["clip_id"],
                "source": record["source"],
                "video_id": record["video_id"],
                "window_start": start,
                "off_center_score": score,
                "outside_center_mass": outside.tolist(),
            }
        )

    candidates.sort(key=lambda value: (-value["off_center_score"], value["clip_id"]))
    selected = []
    used_sources = set()
    for candidate in candidates:
        if candidate["source"] in used_sources:
            continue
        selected.append(candidate)
        used_sources.add(candidate["source"])
        if len(selected) == num_clips:
            break
    if len(selected) < num_clips:
        raise ValueError(
            f"Requested {num_clips} distinct-source clips, found only {len(selected)}"
        )
    return selected


def select_off_center_examples_per_source(
    records: Sequence[Mapping],
    cell_mass: np.ndarray,
    sources: Sequence[str],
    clips_per_source: int = 4,
    frame_count: int = 4,
    center_budget: int = 32,
    grid_size: int = 14,
) -> list[dict]:
    """Select off-center clips per source, preferring distinct videos."""
    if clips_per_source <= 0:
        raise ValueError("clips_per_source must be positive")

    candidates_by_source = {source: [] for source in sources}
    for dataset_index, record in enumerate(records):
        source = record["source"]
        if source not in candidates_by_source:
            continue
        cache_index = int(record["cell_mass_index"])
        start, score, outside = strongest_off_center_window(
            cell_mass[cache_index],
            frame_count=frame_count,
            center_budget=center_budget,
            grid_size=grid_size,
        )
        candidates_by_source[source].append(
            {
                "dataset_index": dataset_index,
                "clip_id": record["clip_id"],
                "source": source,
                "video_id": record["video_id"],
                "window_start": start,
                "off_center_score": score,
                "outside_center_mass": outside.tolist(),
            }
        )

    selected = []
    for source in sources:
        candidates = sorted(
            candidates_by_source[source],
            key=lambda value: (-value["off_center_score"], value["clip_id"]),
        )
        source_selected = []
        used_clips = set()
        used_videos = set()
        for candidate in candidates:
            if candidate["video_id"] in used_videos:
                continue
            source_selected.append(candidate)
            used_clips.add(candidate["clip_id"])
            used_videos.add(candidate["video_id"])
            if len(source_selected) == clips_per_source:
                break
        for candidate in candidates:
            if len(source_selected) == clips_per_source:
                break
            if candidate["clip_id"] in used_clips:
                continue
            source_selected.append(candidate)
            used_clips.add(candidate["clip_id"])
        if len(source_selected) < clips_per_source:
            raise ValueError(
                f"Requested {clips_per_source} clips for {source}, found only "
                f"{len(source_selected)}"
            )
        for source_rank, candidate in enumerate(source_selected, start=1):
            selected.append({**candidate, "source_rank": source_rank})
    return selected
