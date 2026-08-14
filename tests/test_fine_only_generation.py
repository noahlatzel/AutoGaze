# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from autogaze.human_gaze.coverage import global_positions_to_fine_cells
from autogaze.models.autogaze.modeling_autogaze import (
    AllowedTokensLogitsProcessor,
    mask_previously_selected,
)


def test_allowed_tokens_processor_masks_coarse_and_eos() -> None:
    processor = AllowedTokensLogitsProcessor(range(69, 265))
    scores = processor(torch.tensor([[1, 2]]), torch.zeros(1, 266))
    assert torch.isneginf(scores[0, :69]).all()
    assert torch.isfinite(scores[0, 69:265]).all()
    assert torch.isneginf(scores[0, 265])


def test_global_positions_convert_to_unique_fine_cells() -> None:
    local = torch.arange(69, 85).repeat(16, 1)
    offsets = torch.arange(16).reshape(16, 1) * 265
    global_positions = (local + offsets).reshape(1, -1)
    fine = global_positions_to_fine_cells(global_positions)
    assert fine.shape == (1, 16, 16)
    torch.testing.assert_close(fine[0, 0], torch.arange(16))


def test_global_positions_reject_coarse_or_repeated_actions() -> None:
    local = torch.arange(69, 85).repeat(16, 1)
    offsets = torch.arange(16).reshape(16, 1) * 265
    coarse = local.clone()
    coarse[0, 0] = 68
    with pytest.raises(ValueError, match="fine-only"):
        global_positions_to_fine_cells((coarse + offsets).reshape(1, -1))
    repeated = local.clone()
    repeated[0, 1] = repeated[0, 0]
    with pytest.raises(ValueError, match="repeat"):
        global_positions_to_fine_cells((repeated + offsets).reshape(1, -1))


def test_rescoring_masks_previous_actions_within_each_frame() -> None:
    probabilities = torch.full((1, 4, 3), 1 / 3)
    rescored = mask_previously_selected(
        probabilities,
        [torch.tensor([[0, 1]]), torch.tensor([[1, 2]])],
    )
    torch.testing.assert_close(rescored[0, 0], torch.full((3,), 1 / 3))
    torch.testing.assert_close(rescored[0, 1], torch.tensor([0.0, 0.5, 0.5]))
    torch.testing.assert_close(rescored[0, 2], torch.full((3,), 1 / 3))
    torch.testing.assert_close(rescored[0, 3], torch.tensor([0.5, 0.0, 0.5]))
