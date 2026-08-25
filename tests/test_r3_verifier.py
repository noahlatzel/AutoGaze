import pytest

from scripts.human_gaze.verify_r3_temporal_position import (
    nested_differences,
    verify_temporal_rows,
)


def test_nested_differences_reports_exact_leaf_paths():
    left = {"model": {"mode": "none", "same": 1}, "trainer": {"seed": 4}}
    right = {"model": {"mode": "position", "same": 1}, "trainer": {"seed": 4}}
    assert nested_differences(left, right) == ["model.mode"]


def test_verify_temporal_rows_checks_every_tracked_ratio():
    rows = [
        {"temporal_position_gate": 0.0, "temporal_signal_to_feature_rms": 0.0},
        {"temporal_position_gate": -0.1, "temporal_signal_to_feature_rms": 0.1},
    ]
    assert verify_temporal_rows(rows, max_ratio=0.25) == 0.1

    rows.append({"temporal_position_gate": -0.3, "temporal_signal_to_feature_rms": 0.3})
    with pytest.raises(AssertionError):
        verify_temporal_rows(rows, max_ratio=0.25)
