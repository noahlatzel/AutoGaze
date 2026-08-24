from scripts.human_gaze.verify_r3_temporal_position import nested_differences


def test_nested_differences_reports_exact_leaf_paths():
    left = {"model": {"mode": "none", "same": 1}, "trainer": {"seed": 4}}
    right = {"model": {"mode": "position", "same": 1}, "trainer": {"seed": 4}}
    assert nested_differences(left, right) == ["model.mode"]
