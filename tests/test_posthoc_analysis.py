import numpy as np
import pytest

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
from autogaze.human_gaze.posthoc_analysis import (
    overlap_values,
    pairwise_disagreement,
    pooled_prior_cells,
    report_from_video_values,
    video_cluster_bootstrap_mean_difference,
    video_pooled_bootstrap_delta,
    within_pairs,
)


def test_pooled_prior_uses_training_mass_only_and_stable_ties():
    mass = np.full((3, 1, 196), 0.0)
    mass[0, 0, 5] = 1.0  # train
    mass[1, 0, 7] = 1.0  # train
    mass[2, 0, 9] = 1.0  # test; must be ignored
    records = [{"split": "train"}, {"split": "train"}, {"split": "test"}]
    cells = pooled_prior_cells(records, mass, exact_k=3)
    assert cells == [5, 7, 0]


def test_overlap_values_intersection_and_jaccard():
    left = np.arange(16).reshape(1, 1, 16)
    right = np.arange(8, 24).reshape(1, 1, 16)
    values = overlap_values(left, right)
    assert values["intersection_over_k"][0, 0] == pytest.approx(0.5)
    assert values["set_jaccard"][0, 0] == pytest.approx(8 / 24)


def _report(offset, sizes=(1, 3)):
    values = {}
    for index, source in enumerate(STAVIS_SOURCES):
        n = sizes[index % len(sizes)]
        values[source] = {f"v{j}": offset + j for j in range(n)}
    return report_from_video_values(values)


def test_video_pooled_bootstrap_constant_shift():
    left, right = _report(1.0), _report(0.0)
    result = video_pooled_bootstrap_delta([left, left], [right, right], iterations=50, seed=1)
    assert result["difference"] == pytest.approx(1.0)
    assert result["ci90_low"] == pytest.approx(1.0)
    assert result["num_videos"] == 12


def test_video_pooled_weights_videos_not_sources():
    shape = _report(0.0)
    values = {s: {k: 0.0 for k in shape["per_source"][s]["video_values"]} for s in STAVIS_SOURCES}
    right = report_from_video_values(values)
    values = {s: dict(v) for s, v in values.items()}
    values[STAVIS_SOURCES[1]] = {k: 1.0 for k in values[STAVIS_SOURCES[1]]}  # 3 videos
    left = report_from_video_values(values)
    pooled = video_pooled_bootstrap_delta([left], [right], iterations=10, seed=0)
    assert pooled["difference"] == pytest.approx(3 / 12)
    assert left["macro_source_mean"] == pytest.approx(1 / 6)


def test_disagreement_and_cluster_bootstrap():
    answers = {"a": ["A", "B", "C", "D"], "b": ["A", "C", "C", "D"], "c": ["B", "C", "C", "A"]}
    d = pairwise_disagreement(answers, within_pairs(["a", "b", "c"]))
    assert d.shape == (3, 4)
    assert d.mean(axis=1).tolist() == [0.25, 0.75, 0.5]
    result = video_cluster_bootstrap_mean_difference(
        np.array([1.0, 1.0, 0.0, 0.0]), np.zeros(4), ["x", "x", "y", "y"], iterations=20, seed=0)
    assert result["difference"] == pytest.approx(0.5)
    assert result["num_clusters"] == 2
