"""Exercise the real ten-head embedding/scoring route without pretrained weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from autogaze.models.autogaze.autogaze import AutoGaze
from autogaze.models.autogaze.modeling_autogaze import AutoGazeModel


class _Vision(nn.Module):
    def forward(self, video, **kwargs):
        return video.new_zeros(video.shape[0], 4, video.shape[1], 1, 1), None


class _Decoder(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(266, 4)
        self.logits = nn.Parameter(logits)

    def forward(self, inputs_embeds, attention_mask, position_ids):
        # No **kwargs: a leaked return_supervised_log_probs flag fails this call.
        batch, positions = inputs_embeds.shape[:2]
        assert positions == self.logits.shape[0]
        return SimpleNamespace(
            logits=self.logits[None].expand(batch, -1, -1),
            task_loss_prediction=self.logits.new_zeros(batch, positions, 10),
            loss=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def _fixture(*, extreme=False):
    model = AutoGazeModel.__new__(AutoGazeModel)
    nn.Module.__init__(model)
    model.num_vision_tokens_each_frame = 265
    model.input_img_size = 2
    model.frame_sampling_rate = 1
    model.num_multi_token_pred = 10
    model.gaze_decoder_config = SimpleNamespace(vocab_size=266, eos_token_id=265)
    model.vision_model = _Vision()
    model.connector = nn.Identity()
    generator = torch.Generator().manual_seed(21)
    logits = torch.randn(34, 10, 266, generator=generator)
    if extreme:
        logits.zero_()
        logits[..., 69] = 10000
        logits[..., 264] = -10000
    model.gaze_decoder = _Decoder(logits.reshape(34, -1))
    local = torch.arange(69, 85).repeat(2, 1)
    positions = local + 265 * torch.arange(2)[:, None]
    info = {
        "gazing_pos": positions.reshape(1, -1),
        "num_gazing_each_frame": torch.tensor([16, 16]),
        "if_padded_gazing": torch.zeros(1, 32, dtype=torch.bool),
    }
    return model, torch.zeros(1, 2, 3, 2, 2), info


def _score(model, video, info, **kwargs):
    return model(video, info, allowed_token_ids=list(range(69, 265)), **kwargs)


def test_supervised_probabilities_follow_real_ten_head_sources_and_reset_per_frame():
    model, video, info = _fixture()
    output = _score(model, video, info, return_supervised_log_probs=True)
    actual = output.supervised_action_log_probs_all
    assert actual.shape == (1, 32, 196)
    assert actual.dtype == torch.float32
    logits = model.gaze_decoder.logits.reshape(34, 10, 266)
    for frame in range(2):
        for step in range(16):
            # Frame embedding starts with one visual token. Heads 0..9 predict
            # actions 1..10; the next six originate from the tenth gaze token.
            source = frame * 17 + (step // 10) * 10
            head = step % 10
            expected_logits = logits[source, head, 69:265].clone()
            expected_logits[:step] = -torch.inf
            expected = torch.log_softmax(expected_logits, dim=-1)
            torch.testing.assert_close(actual[0, frame * 16 + step], expected)
            assert torch.isfinite(actual[0, frame * 16 + step]).sum() == 196 - step
    torch.testing.assert_close(torch.logsumexp(actual, -1), torch.zeros(1, 32), atol=1e-6, rtol=0)
    assert torch.isfinite(actual[0, 16, :16]).all()


def test_opt_in_matches_legacy_no_repeat_probabilities_without_changing_default():
    model, video, info = _fixture()
    default = _score(model, video, info)
    explicit_legacy = _score(model, video, info, return_supervised_log_probs=False)
    supervised = _score(model, video, info, return_supervised_log_probs=True)
    assert default.supervised_action_log_probs_all is None
    assert "supervised_action_log_probs_all" not in default
    assert torch.equal(default.gaze_probs, explicit_legacy.gaze_probs)
    assert torch.equal(default.gaze_log_probs_all, explicit_legacy.gaze_log_probs_all)
    # Remove precisely the legacy epsilon floor before comparing distributions.
    torch.testing.assert_close(
        supervised.supervised_action_log_probs_all.exp(),
        default.gaze_log_probs_all.exp() - 1e-8,
        atol=3e-8,
        rtol=2e-5,
    )
    repeated_default = _score(model, video, info)
    assert torch.equal(default.gaze_log_probs_all, repeated_default.gaze_log_probs_all)


def test_extreme_logits_keep_finite_nonzero_gradients_after_dominant_cell_is_removed():
    model, video, info = _fixture(extreme=True)
    output = _score(model, video, info, return_supervised_log_probs=True)
    log_probs = output.supervised_action_log_probs_all
    assert torch.isneginf(log_probs[0, 1, 0])
    assert log_probs[0, 1, 195] < -9000  # No epsilon floor around log(1e-8).
    loss = -log_probs[0, 1, 195]
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.gaze_decoder.logits.grad.reshape(34, 10, 266)
    assert torch.isfinite(gradient).all()
    assert gradient[0, 1, 264] < -0.99
    assert abs(gradient[0, 1, 69]) < 1e-6


def test_bfloat16_decoder_logits_are_conditioned_and_normalized_in_float32():
    model, video, info = _fixture(extreme=True)
    model.gaze_decoder.logits = nn.Parameter(model.gaze_decoder.logits.to(torch.bfloat16))
    log_probs = _score(model, video, info, return_supervised_log_probs=True).supervised_action_log_probs_all
    assert log_probs.dtype == torch.float32
    torch.testing.assert_close(torch.logsumexp(log_probs, -1), torch.zeros(1, 32), atol=1e-6, rtol=0)
    (-log_probs[0, 1, 195]).backward()
    assert torch.isfinite(model.gaze_decoder.logits.grad).all()


@pytest.mark.parametrize("invalid", ["repeat", "coarse", "padding", "eos"])
def test_supervised_fixed_fine_contract_rejects_invalid_teacher_histories(invalid):
    model, video, info = _fixture()
    kwargs = {}
    if invalid == "repeat":
        info["gazing_pos"][0, 1] = info["gazing_pos"][0, 0]
    elif invalid == "coarse":
        info["gazing_pos"][0, 0] = 68
    elif invalid == "padding":
        info["if_padded_gazing"][0, 0] = True
    else:
        kwargs["allow_eos"] = True
    with pytest.raises(ValueError, match="Supervised|Teacher"):
        _score(model, video, info, return_supervised_log_probs=True, **kwargs)


def test_supervised_distribution_requires_the_exact_fine_vocabulary():
    model, video, info = _fixture()
    with pytest.raises(ValueError, match="fine action IDs"):
        model(video, info, allowed_token_ids=list(range(265)), return_supervised_log_probs=True)


def test_public_wrapper_forwards_opt_in_only_for_supplied_rescoring_histories(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, video, info = _fixture()
    wrapper = AutoGaze.__new__(AutoGaze)
    nn.Module.__init__(wrapper)
    wrapper.gazing_model = model
    wrapper.attn_mode = "eager"
    wrapper.scales = [32, 64, 112, 224]
    wrapper.num_vision_tokens_each_frame = 265
    wrapper.num_vision_tokens_each_scale_each_frame = [4, 16, 49, 196]
    wrapper.frame_sampling_rate = 1
    kwargs = {"gazing_info": info, "allowed_token_ids": list(range(69, 265))}
    with pytest.warns(UserWarning, match="CUDA is not available"):
        supervised = wrapper({"video": video}, return_supervised_log_probs=True, **kwargs)
    assert supervised["supervised_action_log_probs_all"].shape == (1, 32, 196)
    with pytest.warns(UserWarning, match="CUDA is not available"):
        default = wrapper({"video": video}, **kwargs)
    assert "supervised_action_log_probs_all" not in default
    for invalid in ({}, {"gazing_info": info, "generate_only": True}):
        with pytest.raises(ValueError, match="supplied gazing_info and rescoring"):
            wrapper({"video": video}, return_supervised_log_probs=True, **invalid)
