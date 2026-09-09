import torch

from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.hlvid import FixedBudgetCallStats, install_fixed_budget_forward


class DummyGazingModel(torch.nn.Module):
    frame_sampling_rate = 1

    def generate(self, video, **kwargs):
        raise AssertionError("The simple forwarding fixture must not generate")


class DummyAutoGaze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_kwargs = None
        self.gazing_model = DummyGazingModel()

    def forward(self, inputs, **kwargs):
        self.seen_kwargs = kwargs
        return {"num_gazing_each_frame": torch.tensor([96, 92, 96])}


def test_fixed_budget_adapter_overrides_nvila_generation_contract():
    model = DummyAutoGaze()
    stats = install_fixed_budget_forward(model, exact_budget=24)

    model(
        {"video": torch.zeros(1)},
        gazing_ratio=[0.2, 0.06],
        task_loss_requirement=0.6,
    )

    assert model.seen_kwargs["gazing_ratio"] is None
    assert model.seen_kwargs["task_loss_requirement"] is None
    assert model.seen_kwargs["max_gaze_tokens_each_frame"] == 24
    assert model.seen_kwargs["allowed_token_ids"] == tuple(range(69, 265))
    assert model.seen_kwargs["allow_eos"] is False
    assert stats.as_dict() == {
        "model_calls": 1,
        "frame_observations": 3,
        "retained_patches_per_frame_mean": 284 / 3,
        "retained_patches_per_frame_min": 92,
        "retained_patches_per_frame_max": 96,
    }


def test_fixed_budget_adapter_rejects_more_actions_than_fine_cells():
    model = DummyAutoGaze()
    try:
        install_fixed_budget_forward(model, exact_budget=197)
    except ValueError as error:
        assert "available fine actions" in str(error)
    else:
        raise AssertionError("Expected an invalid-budget error")


class SyntheticRecoveryAutoGaze(torch.nn.Module):
    """Minimal 2x2 tile fixture for 14x14-to-28x28 recovery semantics."""

    def __init__(self):
        super().__init__()
        self.gazing_model = DummyGazingModel()
        self.gazing_model.generate = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Static adapter did not replace generation")
        )
        self.forward_kwargs = None
        self.raw = None

    def forward(self, inputs, **kwargs):
        self.forward_kwargs = kwargs
        expanded = inputs["video"].expand(4, -1, -1, -1, -1)
        self.raw = self.gazing_model.generate(expanded)
        local_actions = self.raw["gazing_pos"] - 69
        recovered = []
        for tile, cells in enumerate(local_actions):
            tile_row, tile_column = divmod(tile, 2)
            for cell in cells.tolist():
                row, column = divmod(cell, 14)
                recovered.append((tile_row * 14 + row) * 28 + tile_column * 14 + column)
        recovered = torch.tensor(recovered).unsqueeze(0)
        return {
            "gazing_pos": recovered,
            "num_gazing_each_frame": torch.tensor([len(recovered[0])]),
            "if_padded_gazing": torch.zeros_like(recovered, dtype=torch.bool),
        }


def test_center16_is_injected_tile_local_before_recovery():
    model = SyntheticRecoveryAutoGaze()
    center16 = center_order(14)[:16].tolist()
    trace = install_fixed_budget_forward(
        model,
        exact_budget=16,
        static_fine_cells=center16,
    )
    trace.start_example()
    output = model({"video": torch.zeros(1, 1, 3, 4, 4)})
    observed = trace.finish_example()

    expected_actions = [69 + cell for cell in center16]
    assert model.raw["gazing_pos"].shape == (4, 16)
    assert model.raw["gazing_pos"][0].tolist() == expected_actions
    assert model.forward_kwargs["generate_only"] is True
    assert output["num_gazing_each_frame"].tolist() == [64]
    assert len(set(output["gazing_pos"][0].tolist())) == 64
    assert observed["post_recovery_valid_patch_count"]["mean"] == 64


def test_trace_distinguishes_padded_batch_maxima_from_valid_counts():
    stats = FixedBudgetCallStats()
    stats.start_example()
    generation = {
        "gazing_pos": torch.tensor([[69, 70, 334, 335], [69, 70, 334, 335]]),
        "if_padded_gazing": torch.zeros(2, 4, dtype=torch.bool),
        "num_gazing_each_frame": torch.tensor([2, 2]),
    }
    stats.record_generation(
        generation,
        elapsed_seconds=0.1,
        actions_per_frame=265,
        exact_budget=2,
        fine_action_offset=69,
    )
    stats.update(
        {
            "gazing_pos": torch.zeros(2, 7, dtype=torch.long),
            "if_padded_gazing": torch.tensor(
                [
                    [False, False, False, False, False, False, False],
                    [False, False, True, False, False, False, True],
                ]
            ),
            "num_gazing_each_frame": torch.tensor([3, 4]),
        },
        elapsed_seconds=0.2,
    )
    result = stats.finish_example()

    assert result["post_recovery_padded_length"]["mean"] == 3.5
    assert result["post_recovery_valid_patch_count"]["mean"] == 3.0
    assert result["calls"][0]["post_recovery_valid_counts_per_item_frame"] == [
        [3, 4],
        [2, 3],
    ]
