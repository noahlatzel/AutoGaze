import numpy as np
import pytest

from autogaze.human_gaze.temporal import (
    build_video_series,
    distribution_pair_metrics,
    lagged_pairs,
)


def test_build_video_series_crosses_clip_boundary() -> None:
    records = [
        {"source": "S", "video_id": "V", "target_start_index": 0, "cell_mass_index": 0},
        {"source": "S", "video_id": "V", "target_start_index": 2, "cell_mass_index": 1},
    ]
    mass = np.zeros((2, 2, 4), dtype=np.float32)
    mass[0, :, 0] = 1
    mass[1, :, 1] = 1
    series = build_video_series(records, mass)[("S", "V")]
    left, right = lagged_pairs(series, 1)
    assert left.shape == right.shape == (3, 4)
    np.testing.assert_array_equal(series["indices"], [0, 1, 2, 3])


def test_pair_metrics_identical_and_disjoint() -> None:
    left = np.zeros((2, 196), dtype=np.float64)
    right = np.zeros_like(left)
    left[:, :32] = 1 / 32
    right[0] = left[0]
    right[1, -32:] = 1 / 32
    metrics = distribution_pair_metrics(left, right)
    assert metrics["jsd"][0] == 0
    assert metrics["cosine"][0] == pytest.approx(1)
    assert metrics["top16_jaccard"][0] == 1
    assert metrics["cosine"][1] == 0
    assert metrics["top32_jaccard"][1] == 0
