import torch

from autogaze.human_gaze.random_cells import install_random_fine_cells, random_fine_cells
from scripts.human_gaze.check_hlvid_control_preflight import validate_random16_actions


def test_random_cells_are_deterministic_distinct_and_in_range():
    first = random_fine_cells(20261106, "video_a.mp4", 3, 7)
    assert first == random_fine_cells(20261106, "video_a.mp4", 3, 7)
    assert len(first) == 16 and len(set(first)) == 16
    assert all(0 <= cell < 196 for cell in first)
    assert list(first) == sorted(first)


def test_random_cells_vary_across_frames_tiles_videos_and_seeds():
    base = random_fine_cells(20261106, "video_a.mp4", 0, 0)
    frames = {random_fine_cells(20261106, "video_a.mp4", 0, frame) for frame in range(16)}
    assert len(frames) == 16
    assert random_fine_cells(20261106, "video_a.mp4", 1, 0) != base
    assert random_fine_cells(20261106, "video_b.mp4", 0, 0) != base
    assert random_fine_cells(20261107, "video_a.mp4", 0, 0) != base


def test_random_cells_are_approximately_uniform():
    counts = torch.zeros(196)
    for frame in range(2000):
        counts[list(random_fine_cells(1, "v", 0, frame))] += 1
    expected = 2000 * 16 / 196
    assert counts.min() > 0.6 * expected and counts.max() < 1.4 * expected


class DummyGazingModel(torch.nn.Module):
    frame_sampling_rate = 1

    def generate(self, video, **kwargs):
        raise AssertionError("Learned generation must be replaced")


class BatchedAutoGaze(torch.nn.Module):
    """Calls generation like NVILA's processor: tile minibatches in order."""

    def __init__(self):
        super().__init__()
        self.gazing_model = DummyGazingModel()
        self.kwargs = []

    def forward(self, inputs, **kwargs):
        self.kwargs.append(kwargs)
        return self.gazing_model.generate(inputs["video"])


def test_installed_selector_is_batch_invariant_and_traced():
    outputs = {}
    for batch_size in (5, 2):
        model = BatchedAutoGaze()
        stats, selector = install_random_fine_cells(model, seed=11, exact_budget=16)
        stats.start_example()
        selector.start_example("clip.mp4")
        rows = []
        for start in range(0, 5, batch_size):
            count = min(batch_size, 5 - start)
            rows.append(model({"video": torch.zeros(count, 3, 1, 2, 2)})["gazing_pos"])
        trace = stats.finish_example()
        assert selector.finish_example() == {"video_key": "clip.mp4", "tiles": 5}
        assert all(kwargs["generate_only"] is True for kwargs in model.kwargs)
        assert all(kwargs["max_gaze_tokens_each_frame"] == 16 for kwargs in model.kwargs)
        outputs[batch_size] = torch.cat(rows)
        record = {
            "answer": {"video_path": "clip.mp4"},
            "random16": {"video_key": "clip.mp4", "tiles": 5},
            "raw_decoder_calls": trace["calls"],
        }
        validate_random16_actions({"fixed_budget_adapter": {"random16_seed": 11}}, record)
    assert torch.equal(outputs[5], outputs[2])
    positions = outputs[5]
    assert positions.shape == (5, 3 * 16)
    frame1 = positions[0, 16:32] - 265
    assert frame1.tolist() == [69 + c for c in random_fine_cells(11, "clip.mp4", 0, 1)]


def test_selector_fails_closed_without_active_example():
    model = BatchedAutoGaze()
    install_random_fine_cells(model, seed=11, exact_budget=16)
    try:
        model({"video": torch.zeros(1, 1, 1, 2, 2)})
    except RuntimeError as error:
        assert "outside an active example" in str(error)
    else:
        raise AssertionError("Expected fail-closed generation")
