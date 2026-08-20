import torch

from autogaze.human_gaze.hlvid import install_fixed_budget_forward


class DummyAutoGaze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_kwargs = None

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
