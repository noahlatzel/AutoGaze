from types import SimpleNamespace

import pytest
import torch

from autogaze.human_gaze.r2d_hlvid import R2DHLVidCallStats, install_r2d_hlvid_forward


class DummyDecoder:
    def __init__(self):
        self.biases = {}

    def set_output_token_logit_bias(self, token_id, bias):
        self.biases[token_id] = bias


class DummyGazingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gaze_decoder = DummyDecoder()
        self.gaze_decoder_config = SimpleNamespace(eos_token_id=265)
        self.seen_kwargs = None

    def generate(self, video, **kwargs):
        self.seen_kwargs = kwargs
        positions = torch.tensor(
            [
                [69, 70, 71, 72, 265, 334, 335, 336, 337, 338, 530],
                [69, 70, 71, 72, 265, 334, 335, 336, 337, 338, 530],
            ]
        )
        return {
            "gazing_pos": positions,
            "if_padded_gazing": torch.zeros_like(positions, dtype=torch.bool),
            "num_gazing_each_frame": torch.tensor([5, 6]),
        }


class DummyAutoGaze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gazing_model = DummyGazingModel()
        self.seen_kwargs = None

    def forward(self, inputs, **kwargs):
        self.seen_kwargs = kwargs
        self.gazing_model.generate(inputs["video"], **kwargs)
        return {
            "num_gazing_each_frame": torch.tensor([3, 2]),
            "if_padded_gazing": torch.tensor(
                [[False, False, True, False, True], [False, True, True, False, False]]
            ),
        }


def test_variable_adapter_uses_calibrated_eos_contract_and_true_lengths():
    model = DummyAutoGaze()
    stats = install_r2d_hlvid_forward(model, mode="variable", eos_logit_bias=2.3046875)
    model({"video": torch.zeros(2, 2, 3, 4, 4)}, gazing_ratio=0.2, task_loss_requirement=0.6)

    assert model.seen_kwargs["gazing_ratio"] is None
    assert model.seen_kwargs["task_loss_requirement"] is None
    assert model.seen_kwargs["max_gaze_tokens_each_frame"] == 36
    assert model.seen_kwargs["min_gaze_tokens_each_frame"] == 4
    assert model.seen_kwargs["allowed_token_ids"] == tuple(range(69, 266))
    assert model.seen_kwargs["allow_eos"] is True
    assert model.gazing_model.gaze_decoder.biases == {265: 2.3046875}
    summary = stats.as_dict()
    assert summary["decoder"]["spatial_actions_per_frame_mean"] == 4.5
    assert summary["decoder"]["eos_action_rate"] == 1.0
    assert summary["post_resolution_adaptation"] == {
        "frame_observations": 4,
        "retained_patches_per_frame_mean": 1.5,
        "retained_patches_per_frame_min": 1,
        "retained_patches_per_frame_max": 2,
    }


def test_forced_k16_adapter_disables_eos_coherently():
    model = DummyAutoGaze()
    install_r2d_hlvid_forward(model, mode="forced_k16")
    model({"video": torch.zeros(2, 2, 3, 4, 4)})
    assert model.seen_kwargs["max_gaze_tokens_each_frame"] == 16
    assert model.seen_kwargs["min_gaze_tokens_each_frame"] == 0
    assert model.seen_kwargs["allowed_token_ids"] == tuple(range(69, 265))
    assert model.seen_kwargs["allow_eos"] is False
    assert model.gazing_model.gaze_decoder.biases == {}


def test_variable_adapter_requires_calibration_mapping():
    with pytest.raises(ValueError, match="calibrated EOS"):
        install_r2d_hlvid_forward(DummyAutoGaze(), mode="variable")


def test_decoder_stats_reject_non_fine_actions():
    stats = R2DHLVidCallStats()
    with pytest.raises(ValueError, match="outside the R2d contract"):
        stats.update_decoder(
            {
                "gazing_pos": torch.tensor([[0]]),
                "if_padded_gazing": torch.tensor([[False]]),
                "num_gazing_each_frame": torch.tensor([1]),
            },
            fine_action_offset=69,
            actions_per_frame=265,
        )
