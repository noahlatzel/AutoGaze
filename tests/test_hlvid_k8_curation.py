import numpy as np

from scripts.human_gaze.curate_hlvid_k8 import cluster_interval


def test_paired_cluster_interval_preserves_identical_predictions():
    result = cluster_interval(np.zeros((6, 3)), ['a', 'a', 'b'], iterations=100)
    assert result['difference'] == 0
    assert result['ci90'] == [0, 0]
    assert result['clusters'] == 2


def test_cluster_interval_weights_questions_and_not_seed_copies():
    difference = np.array([[1., 1., -1.]])
    single = cluster_interval(difference, ['a', 'a', 'b'], iterations=1000)
    six = cluster_interval(np.repeat(difference, 6, axis=0), ['a', 'a', 'b'], iterations=1000)
    assert single == six
    assert single['difference'] == 1 / 3
    assert single['ci90'] == [-1, 1]
