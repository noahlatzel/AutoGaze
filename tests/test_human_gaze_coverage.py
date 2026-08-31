# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
from autogaze.human_gaze.coverage import (
    center_order,
    evaluate_coverage_baselines,
    selected_coverage,
)


def test_center_16_is_central_four_by_four() -> None:
    selected = set(center_order(14)[:16].tolist())
    expected = {row * 14 + column for row in range(5, 9) for column in range(5, 9)}
    assert selected == expected


def test_selected_coverage_rejects_repeats() -> None:
    mass = torch.full((1, 196), 1 / 196)
    try:
        selected_coverage(mass, torch.tensor([0, 0]))
    except ValueError as error:
        assert "repeat" in str(error)
    else:
        raise AssertionError("Repeated selections were accepted")


def test_oracle_and_prior_recover_concentrated_mass() -> None:
    dataset = []
    priors = {}
    for source_index, source in enumerate(STAVIS_SOURCES):
        cell = 20 + source_index
        mass = torch.zeros(2, 196)
        mass[:, cell] = 1
        dataset.append({"source": source, "video_id": f"video_{source_index}", "cell_mass": mass})
        prior = torch.zeros(196, dtype=torch.float64)
        prior[cell] = 1
        priors[source] = prior
    results = evaluate_coverage_baselines(dataset, priors, budgets=[1, 16], random_seeds=[3])
    assert results["oracle"]["1"]["macro_source_mean"] == 1
    assert results["prior"]["1"]["macro_source_mean"] == 1
    assert results["oracle"]["16"]["macro_source_mean"] == 1
