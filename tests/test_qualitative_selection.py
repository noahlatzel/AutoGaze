import numpy as np

from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.qualitative import (
    select_off_center_examples,
    select_off_center_examples_per_source,
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


def test_per_source_selection_prefers_videos_then_fills_with_clips() -> None:
    center = int(center_order(14)[0])
    corner = 0
    records = [
        {"clip_id": "A/v1/0", "source": "A", "video_id": "v1", "cell_mass_index": 0},
        {"clip_id": "A/v1/1", "source": "A", "video_id": "v1", "cell_mass_index": 1},
        {"clip_id": "A/v2/0", "source": "A", "video_id": "v2", "cell_mass_index": 2},
        {"clip_id": "B/v3/0", "source": "B", "video_id": "v3", "cell_mass_index": 3},
        {"clip_id": "B/v3/1", "source": "B", "video_id": "v3", "cell_mass_index": 4},
        {"clip_id": "B/v3/2", "source": "B", "video_id": "v3", "cell_mass_index": 5},
    ]
    mass = np.zeros((6, 2, 196), dtype=np.float32)
    mass[:, :, center] = 1.0
    for index, strength in enumerate((1.0, 0.9, 0.8, 1.0, 0.9, 0.8)):
        mass[index, :, center] = 1.0 - strength
        mass[index, :, corner] = strength

    selected = select_off_center_examples_per_source(
        records, mass, sources=("A", "B"), clips_per_source=3, frame_count=2
    )

    assert [item["clip_id"] for item in selected] == [
        "A/v1/0",
        "A/v2/0",
        "A/v1/1",
        "B/v3/0",
        "B/v3/1",
        "B/v3/2",
    ]
    assert [item["source_rank"] for item in selected] == [1, 2, 3, 1, 2, 3]
