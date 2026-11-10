# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from scripts.frame_selectors.generate_infoshot_indices import (
    Segment,
    best_segment_split,
    candidate_source_indices,
    cosine_affinity,
    dual_keyframes,
    infoshot_select,
    minmax_normalize,
)


def test_minmax_normalize_constant_values_returns_zero() -> None:
    np.testing.assert_array_equal(minmax_normalize(np.ones(4)), np.zeros(4))


def test_candidate_source_indices_are_unique_and_cover_endpoints() -> None:
    indices = candidate_source_indices(frame_count=301, source_fps=30.0, candidate_fps=10.0)

    assert indices[0] == 0
    assert indices[-1] == 300
    assert len(indices) == len(set(indices))


def test_boundary_score_recovers_piecewise_constant_change() -> None:
    features = np.concatenate(
        [
            np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (12, 1)),
            np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (12, 1)),
        ],
        axis=0,
    )
    affinity = cosine_affinity(features)

    best = best_segment_split(
        affinity,
        Segment(0, 24),
        fixed_window=3,
        min_window=3,
        max_window=15,
        min_segment_frames=3,
        max_segment_frames=300,
    )

    assert best is not None
    assert best[1] == 12


def test_dual_keyframes_select_representative_and_outlier() -> None:
    features = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (8, 1))
    features[4] = np.array([0.0, 1.0], dtype=np.float32)
    affinity = cosine_affinity(features)

    common, unique = dual_keyframes(
        affinity,
        Segment(0, 8),
        neighborhood=1,
        common_weight=0.7,
        unique_weight=0.5,
    )

    assert common != 4
    assert unique == 4


def test_infoshot_select_returns_exact_unique_budget() -> None:
    features = []
    for group in range(4):
        angle = group * np.pi / 8
        vector = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        features.extend([vector] * 6)

    selected, segments, repair_count = infoshot_select(
        np.asarray(features),
        budget=8,
        fixed_window=2,
        min_segment_frames=2,
    )

    assert len(segments) == 4
    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert repair_count == 0


def test_infoshot_select_supports_single_frame_budget() -> None:
    features = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1))

    selected, segments, repair_count = infoshot_select(features, budget=1)

    assert len(segments) == 1
    assert len(selected) == 1
    assert repair_count == 0
