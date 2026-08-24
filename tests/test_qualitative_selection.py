import numpy as np

from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.qualitative import (
    select_off_center_examples,
    strongest_off_center_window,
)


def test_strongest_off_center_window_finds_consecutive_peak() -> None:
    center = int(center_order(14)[0])
    corner = 0
    mass = np.zeros((6, 196), dtype=np.float32)
    mass[:, center] = 1.0
    mass[2:5, center] = 0.0
    mass[2:5, corner] = 1.0

    start, score, outside = strongest_off_center_window(
        mass, frame_count=3, center_budget=32
    )

    assert start == 2
    assert score == 1.0
    np.testing.assert_allclose(outside, [0, 0, 1, 1, 1, 0])


def test_selection_uses_distinct_sources_and_deterministic_tie_break() -> None:
    center = int(center_order(14)[0])
    corner = 0
    records = [
        {"clip_id": "B/strong", "source": "B", "video_id": "b", "cell_mass_index": 0},
        {"clip_id": "A/z", "source": "A", "video_id": "az", "cell_mass_index": 1},
        {"clip_id": "A/a", "source": "A", "video_id": "aa", "cell_mass_index": 2},
    ]
    mass = np.zeros((3, 4, 196), dtype=np.float32)
    mass[:, :, center] = 1.0
    mass[0, :, center] = 0.0
    mass[0, :, corner] = 1.0
    mass[1:, 1:3, center] = 0.0
    mass[1:, 1:3, corner] = 1.0

    selected = select_off_center_examples(records, mass, num_clips=2, frame_count=2)

    assert [item["clip_id"] for item in selected] == ["A/a", "B/strong"]
    assert [item["window_start"] for item in selected] == [1, 0]
