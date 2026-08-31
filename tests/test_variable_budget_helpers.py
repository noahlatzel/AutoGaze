# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from autogaze.human_gaze.variable_budget import (
    coverage_for_padded_variable_cells,
    coverage_for_variable_lengths,
    variable_global_positions_to_fine_cells,
)


def test_variable_positions_and_coverage() -> None:
    positions = torch.tensor([[1, 2, 5, 5, 6, 7, 8, 10]])
    padded = torch.tensor([[False, False, False, True, False, False, False, False]])
    cells, lengths, eos = variable_global_positions_to_fine_cells(
        positions,
        padded,
        torch.tensor([4, 4]),
        num_frames=2,
        max_budget=3,
        actions_per_frame=5,
        fine_action_offset=1,
    )
    assert cells.tolist() == [[[0, 1, -1], [0, 1, 2]]]
    assert lengths.tolist() == [[2, 3]]
    assert eos.tolist() == [[True, True]]
    ranking = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
    mass = torch.tensor([[[0.4, 0.3, 0.2], [0.1, 0.2, 0.3]]])
    torch.testing.assert_close(
        coverage_for_variable_lengths(mass, ranking, lengths),
        torch.tensor([[0.7, 0.6]]),
    )


def test_variable_positions_reject_repeats() -> None:
    with pytest.raises(ValueError, match="repeat"):
        variable_global_positions_to_fine_cells(
            torch.tensor([[1, 1, 5]]),
            torch.tensor([[False, False, False]]),
            torch.tensor([3]),
            num_frames=1,
            max_budget=2,
            actions_per_frame=5,
            fine_action_offset=1,
        )


def test_coverage_uses_actual_padded_variable_cells() -> None:
    mass = torch.tensor([[[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]])
    selected = torch.tensor([[[2, 0, -1], [3, 1, 0]]])
    lengths = torch.tensor([[2, 3]])
    torch.testing.assert_close(
        coverage_for_padded_variable_cells(mass, selected, lengths),
        torch.tensor([[0.6, 0.7]]),
    )


def test_coverage_rejects_invalid_active_padded_cell() -> None:
    with pytest.raises(ValueError, match="active selected cell"):
        coverage_for_padded_variable_cells(
            torch.full((1, 1, 4), 0.25),
            torch.tensor([[[-1, 0]]]),
            torch.tensor([[1]]),
        )
