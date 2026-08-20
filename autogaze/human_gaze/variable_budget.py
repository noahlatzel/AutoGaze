# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for variable-length fine-only gaze trajectories."""

from __future__ import annotations

import torch


def variable_global_positions_to_fine_cells(
    gazing_pos: torch.Tensor,
    if_padded_gazing: torch.Tensor,
    frame_widths: torch.Tensor,
    num_frames: int = 16,
    max_budget: int = 36,
    actions_per_frame: int = 265,
    fine_action_offset: int = 69,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded fine-cell sequences, spatial lengths, and EOS indicators."""
    if gazing_pos.shape != if_padded_gazing.shape or gazing_pos.ndim != 2:
        raise ValueError("gazing_pos and if_padded_gazing must share shape [B,N]")
    widths = [int(value) for value in frame_widths.tolist()]
    if len(widths) != num_frames or sum(widths) != gazing_pos.shape[1]:
        raise ValueError("Frame widths do not match the trajectory")

    batch_size = gazing_pos.shape[0]
    cells = torch.full(
        (batch_size, num_frames, max_budget),
        -1,
        dtype=torch.long,
        device=gazing_pos.device,
    )
    lengths = torch.zeros(
        batch_size,
        num_frames,
        dtype=torch.long,
        device=gazing_pos.device,
    )
    emitted_eos = torch.zeros(
        batch_size,
        num_frames,
        dtype=torch.bool,
        device=gazing_pos.device,
    )
    positions = gazing_pos.split(widths, dim=1)
    padded = if_padded_gazing.split(widths, dim=1)
    for frame_index, (frame_positions, frame_padded) in enumerate(
        zip(positions, padded)
    ):
        local = frame_positions - frame_index * actions_per_frame
        spatial = (
            (local >= fine_action_offset)
            & (local < actions_per_frame)
            & ~frame_padded
        )
        eos = (local == actions_per_frame) & ~frame_padded
        if (eos.sum(dim=1) > 1).any():
            raise ValueError("A frame contains more than one EOS action")
        if ((~frame_padded) & ~spatial & ~eos).any():
            raise ValueError("Trajectory contains a non-fine, non-EOS action")
        for batch_index in range(batch_size):
            selected = local[batch_index, spatial[batch_index]] - fine_action_offset
            if selected.numel() > max_budget:
                raise ValueError("Trajectory exceeds max_budget")
            if torch.unique(selected).numel() != selected.numel():
                raise ValueError("Generated fine actions repeat within a frame")
            cells[batch_index, frame_index, : selected.numel()] = selected
            lengths[batch_index, frame_index] = selected.numel()
        emitted_eos[:, frame_index] = eos.any(dim=1)
    return cells, lengths, emitted_eos


def coverage_for_variable_lengths(
    cell_mass: torch.Tensor,
    ranked_cells: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Return per-frame coverage for ranked cells truncated at each frame's K."""
    if cell_mass.ndim != 3 or ranked_cells.ndim != 3 or lengths.ndim != 2:
        raise ValueError("Expected cell_mass/ranked_cells/lengths with ranks 3/3/2")
    if cell_mass.shape[:2] != ranked_cells.shape[:2] or lengths.shape != cell_mass.shape[:2]:
        raise ValueError("Batch and frame dimensions must match")
    if lengths.min() < 0 or lengths.max() > ranked_cells.shape[-1]:
        raise ValueError("lengths lie outside the available ranking")
    if ranked_cells.min() < 0 or ranked_cells.max() >= cell_mass.shape[-1]:
        raise ValueError("ranked cell index is out of range")
    selected_mass = cell_mass.gather(2, ranked_cells)
    active = torch.arange(
        ranked_cells.shape[-1], device=lengths.device
    ).reshape(1, 1, -1) < lengths.unsqueeze(-1)
    return (selected_mass * active.to(selected_mass.dtype)).sum(dim=-1)
