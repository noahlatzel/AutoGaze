# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from autogaze.human_gaze.variable_budget import (
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
