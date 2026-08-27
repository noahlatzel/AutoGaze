# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import math
from typing import Optional, Tuple
from dataclasses import dataclass
from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.convnext import ConvNeXtBlock
from timm.layers import LayerNorm2d

from transformers.modeling_outputs import ModelOutput
from transformers import LogitsProcessor, LogitsProcessorList

from .configuration_autogaze import GazeModelConfig, VisionModelConfig, ConnectorConfig
from .modeling_llama_multi_token_pred import LlamaForCausalLM_MultiTokenPred


@dataclass
class AutoGazeOutput(ModelOutput):
    gaze_logits: Optional[torch.FloatTensor] = None
    gaze_probs: Optional[torch.FloatTensor] = None
    gaze_log_probs_all: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    task_loss_prediction: Optional[torch.FloatTensor] = None


class NoRepeatTokensLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: (batch_size, sequence_length)
        # scores: (batch_size, vocab_size) or (batch_size, num_multi_token_pred, vocab_size)
        if scores.ndim == 3:
            scores[torch.arange(scores.shape[0])[..., None], :, input_ids] = -float("inf")
        else:
            scores[torch.arange(scores.shape[0])[..., None], input_ids] = -float("inf")
        return scores


class NoEosTokenLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: (batch_size, sequence_length)
        # scores: (batch_size, vocab_size) or (batch_size, num_multi_token_pred, vocab_size)
        scores[..., -1] = -float("inf")
        return scores


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """Mask every vocabulary entry outside a fixed set of action IDs."""

    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed_token_ids = tuple(int(token_id) for token_id in allowed_token_ids)
        if not self.allowed_token_ids:
            raise ValueError("allowed_token_ids must not be empty")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if min(self.allowed_token_ids) < 0 or max(self.allowed_token_ids) >= scores.shape[-1]:
            raise ValueError("allowed token ID is outside the model vocabulary")
        masked = torch.full_like(scores, -float("inf"))
        masked[..., list(self.allowed_token_ids)] = scores[..., list(self.allowed_token_ids)]
        return masked


class AdditiveTokenBiasLogitsProcessor(LogitsProcessor):
    """Add one batch-specific token bias to every active prediction head."""

    def __init__(self, token_bias):
        super().__init__()
        if token_bias.ndim != 2:
            raise ValueError("token_bias must have shape (batch, vocabulary)")
        self.token_bias = token_bias

    def __call__(self, input_ids, scores):
        del input_ids
        if scores.ndim == 2:
            return scores + self.token_bias
        if scores.ndim == 3:
            return scores + self.token_bias[:, None, :]
        raise ValueError("scores must have shape (batch, vocabulary) or (batch, heads, vocabulary)")


class MinimumGazeTokensLogitsProcessor(LogitsProcessor):
    """Keep EOS masked until a minimum number of spatial actions is reached."""

    def __init__(self, eos_token_id: int, minimum_tokens: int):
        super().__init__()
        self.eos_token_id = int(eos_token_id)
        self.minimum_tokens = int(minimum_tokens)
        if self.minimum_tokens < 0:
            raise ValueError("minimum_tokens must be non-negative")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids.shape[1]
        if scores.ndim == 3:
            masked_predictions = min(
                max(self.minimum_tokens - generated, 0),
                scores.shape[1],
            )
            if masked_predictions:
                scores[:, :masked_predictions, self.eos_token_id] = -float("inf")
        elif generated < self.minimum_tokens:
            scores[..., self.eos_token_id] = -float("inf")
        return scores


def eos_padding_mask(
    token_ids: torch.Tensor,
    eos_token_id: int,
    first_eos_is_action: bool,
) -> torch.Tensor:
    """Mark EOS padding while optionally retaining the first EOS as an action."""
    eos_mask = token_ids == eos_token_id
    if not first_eos_is_action:
        return eos_mask
    return torch.cat(
        [
            torch.zeros_like(eos_mask[:, :1]),
            eos_mask[:, :-1].cumsum(dim=1) > 0,
        ],
        dim=1,
    )


def mask_eos_before_minimum(
    probabilities: torch.Tensor,
    gaze_pos_ids_split,
    eos_token_id: int,
    minimum_tokens: int,
) -> torch.Tensor:
    """Match the generation-time minimum-length mask during trajectory rescoring."""
    if minimum_tokens <= 0:
        return probabilities
    masked = probabilities.clone()
    offset = 0
    for frame_ids in gaze_pos_ids_split:
        stop = offset + min(int(minimum_tokens), frame_ids.shape[1])
        masked[:, offset:stop, eos_token_id] = 0
        offset += frame_ids.shape[1]
    normalizer = masked.sum(dim=-1, keepdim=True)
    if (normalizer <= 0).any():
        raise ValueError("No valid actions remain after applying the minimum-length mask")
    return masked / normalizer


def mask_previously_selected(
    probabilities: torch.Tensor,
    gaze_pos_ids_split,
) -> torch.Tensor:
    """Renormalize each step after removing earlier actions in that frame."""
    available = torch.ones_like(probabilities, dtype=torch.bool)
    offset = 0
    for frame_ids in gaze_pos_ids_split:
        for step in range(frame_ids.shape[1]):
            if step:
                available[:, offset + step].scatter_(
                    1,
                    frame_ids[:, :step],
                    False,
                )
        offset += frame_ids.shape[1]
    masked = probabilities * available
    normalizer = masked.sum(dim=-1, keepdim=True)
    if (normalizer <= 0).any():
        raise ValueError("No valid actions remain after applying the no-repeat mask")
    return masked / normalizer


class AutoGazeModel(nn.Module):
    def __init__(self, gaze_model_config: GazeModelConfig):
        super().__init__()

        self.num_vision_tokens_each_frame = gaze_model_config.num_vision_tokens_each_frame
        self.input_img_size = gaze_model_config.input_img_size
        self.frame_sampling_rate = gaze_model_config.vision_model_config.temporal_patch_size
        self.num_multi_token_pred = gaze_model_config.gaze_decoder_config.num_multi_token_pred
        self.gaze_decoder_config = gaze_model_config.gaze_decoder_config  # Store for reference
        temporal_mode = gaze_model_config.temporal_position_encoding
        if temporal_mode == "normalized_sinusoidal_scalar_gate":
            self.temporal_position_signal = TemporalPositionSignal(
                gaze_model_config.gaze_decoder_config.hidden_size,
                max_frequency=gaze_model_config.temporal_position_max_frequency,
            )
        elif temporal_mode == "causal_connector_difference_scalar_gate":
            self.causal_difference_signal = CausalConnectorDifferenceSignal(
                gaze_model_config.gaze_decoder_config.hidden_size,
            )
        elif temporal_mode == "causal_feature_transport_logit_bias":
            self.feature_transport_bias = CausalFeatureTransportLogitBias(
                gaze_model_config.gaze_decoder_config.hidden_size,
                temperature=gaze_model_config.feature_transport_temperature,
            )
        elif temporal_mode == "selection_conditioned_recurrent_state_logit_bias":
            self.recurrent_state_bias = SelectionConditionedRecurrentStateLogitBias(
                gaze_model_config.gaze_decoder_config.hidden_size,
                gaze_model_config.connector_config.num_tokens,
                state_hidden_dim=gaze_model_config.recurrent_state_hidden_dim,
            )
        elif temporal_mode != "none":
            raise ValueError(f"Unknown temporal position encoding: {temporal_mode}")

        # Create the vision model, connector, and gaze decoder
        self.vision_model = ShallowVideoConvNet(gaze_model_config.vision_model_config)
        self.connector = Connector(gaze_model_config.connector_config)
        self.gaze_decoder = LlamaForCausalLM_MultiTokenPred(gaze_model_config.gaze_decoder_config)

        # Exact action IDs never repeat. EOS masking is selected per generation
        # call because variable-length tasks need to learn a stopping action.
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(NoRepeatTokensLogitsProcessor())  # don't allow repeated gazing

    def embed(self, video=None, gaze_pos_ids=None, use_cache=False, past_conv_values=None):
        """
        inputs:
            video: (B x T x C x H x W).
            gaze_pos_ids: list of (B, N), N is the number of gazing positions in each frame. The length of the list is T // frame_sampling_rate.
        returns:
            embeds: a list of interleaved vision and gaze embeddings. list of (B, N, C)
            gaze_token_mask: a list of masks that indicate if the current embedding is a gaze embedding. (1 is gaze embedding, 0 is vision embedding). list of (N, )
            gaze_pred_source_relative: a list of (relative) source index of where the gaze prediction is coming from. For example, if the gaze prediction is coming from two tokens before it, the source index is -2. list of (N, ).
                For vision embeddings, there's no source prediction, so the source index is -1.
            attention_mask: a list of (B, N) that indicates if the current embedding should be masked out (for EOS token). 1 is not masked, 0 is masked.
        """
        B, T = video.shape[:2]
        assert (video is None or gaze_pos_ids is None) or video.shape[1] // self.frame_sampling_rate == len(gaze_pos_ids), \
            "The number of frames in the video (after subsampling) and in gaze position IDs must be the same, but got {} and {}".format(video.shape[1] // self.frame_sampling_rate, len(gaze_pos_ids))

        if video is not None:
            vision_features, new_past_conv_values = self.vision_model(video, use_cache=use_cache, past_conv_values=past_conv_values)
            vision_features = vision_features.transpose(1, 2)
            vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
            vision_features = self.connector(vision_features)
            if hasattr(self, "temporal_position_signal"):
                vision_features = self.temporal_position_signal(vision_features)
            elif hasattr(self, "causal_difference_signal"):
                vision_features = self.causal_difference_signal(vision_features)
            vision_attention_mask = [torch.ones(B, vision_features.shape[2], device=vision_features.device).long() for _ in range(vision_features.shape[1])]

        if gaze_pos_ids is not None:
            num_gazing_each_frame = [gaze_pos_ids[t].shape[1] for t in range(len(gaze_pos_ids))]
            gaze_pos_ids = torch.cat(gaze_pos_ids, dim=1)
            gaze_attention_mask = (gaze_pos_ids != self.gaze_decoder_config.eos_token_id).to(torch.long)
            gaze_embeds = self.gaze_decoder.model.embed_tokens(gaze_pos_ids)
            gaze_embeds = list(gaze_embeds.split(num_gazing_each_frame, dim=1))
            gaze_attention_mask = list(gaze_attention_mask.split(num_gazing_each_frame, dim=1))

        embeds = []
        gaze_token_mask = []
        gaze_pred_source_relative = []
        attention_mask = []
        for t in range(T // self.frame_sampling_rate):
            if video is not None:
                embeds.append(vision_features[:, t, :, :])
                gaze_token_mask.append(torch.zeros(vision_features.shape[2], device=vision_features.device).long())
                gaze_pred_source_relative.append(torch.zeros(vision_features.shape[2], device=vision_features.device).long() - 1)
                attention_mask.append(vision_attention_mask[t])
            if gaze_pos_ids is not None:
                embeds.append(gaze_embeds[t])
                gaze_token_mask.append(torch.ones(gaze_embeds[t].shape[1], device=gaze_embeds[t].device).long())
                gaze_pred_source_relative.append(-(torch.arange(gaze_embeds[t].shape[1], device=gaze_embeds[t].device) % self.num_multi_token_pred + 1))
                attention_mask.append(gaze_attention_mask[t])
        return embeds, gaze_token_mask, gaze_pred_source_relative, attention_mask, new_past_conv_values if video is not None else None

    @torch.no_grad()
    def generate(
        self, 
        video, 
        max_gaze_tokens_each_frame=100, 
        task_loss_requirement=None, 
        use_cache=False,
        past_key_values=None, 
        past_inputs_embeds=None,
        past_attention_mask=None,
        past_conv_values=None,
        allowed_token_ids=None,
        allow_eos=False,
        min_gaze_tokens_each_frame=0,
        **generation_kwargs,
    ):
        """
        Inputs:
            video: (B, T, C, H, W)
            max_gaze_tokens_each_frame: int or (T, ). Indicating the max gazing length for each frame. If is int, then all frames have the same max gazing length.
            task_loss_requirement (optional): (B, T). Indicating the task loss requirement for each frame.
            past_key_values (optional): The past key values for the gaze model. Can be used for streaming generation.
            past_inputs_embeds (optional): The past inputs embeds for the gaze model. Can be used for streaming generation.
            past_attention_mask (optional): The past attention mask for the gaze model. Can be used for streaming generation.
        """
        if past_key_values is not None or past_inputs_embeds is not None or past_attention_mask is not None or past_conv_values is not None:
            assert past_key_values is not None and past_inputs_embeds is not None and past_attention_mask is not None and past_conv_values is not None, \
                "If past_key_values, past_inputs_embeds, past_attention_mask, or past_conv_values is provided, then all four must be provided!"

        # Subsample frames and resize
        B, T = video.shape[:2]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        video = F.interpolate(video, size=(self.input_img_size, self.input_img_size), mode="bicubic", align_corners=False)
        video = rearrange(video, '(b t) c h w -> b t c h w', b=B)

        # Embed all the frames
        video_embeds, _, __, ___, past_conv_values = self.embed(video=video, use_cache=use_cache, past_conv_values=past_conv_values)

        # Generate gaze position IDs for each frame
        gaze_pos_ids_list = []
        previous_gaze_pos_ids = None
        recurrent_state = None
        recurrent_states = []
        recurrent_biases = []
        if hasattr(self, "recurrent_state_bias"):
            if allowed_token_ids is None:
                raise ValueError("Recurrent state requires explicit allowed_token_ids")
            recurrent_state = self.recurrent_state_bias.initial_state(
                B,
                device=video_embeds[0].device,
                dtype=video_embeds[0].dtype,
            )
        inputs_embeds = [] if past_inputs_embeds is None else past_inputs_embeds
        attention_mask = [] if past_attention_mask is None else past_attention_mask
        past_key_values = None if past_key_values is None else past_key_values
        num_gazing_each_frame = []
        if_padded_gazing = []
        for t in range(len(video_embeds)):

            # Update inputs_embeds and attention mask for the new frame
            inputs_embeds.append(video_embeds[t])
            attention_mask.append(torch.ones(video_embeds[t].shape[0], video_embeds[t].shape[1], device=video_embeds[t].device).long())

            # Put task loss requirement into generation config
            generation_config = self.gaze_decoder.generation_config
            generation_config.task_loss_requirement = task_loss_requirement[:, t] if task_loss_requirement is not None else None

            # Get the max gazing length for the current frame
            assert isinstance(max_gaze_tokens_each_frame, int) or len(max_gaze_tokens_each_frame) == len(video_embeds), \
                "max_gaze_tokens_each_frame must be an int or a tensor of the same length as the video embeddings, but got {} and {}".format(max_gaze_tokens_each_frame, len(video_embeds))
            max_gaze_tokens = max_gaze_tokens_each_frame if isinstance(max_gaze_tokens_each_frame, int) else max_gaze_tokens_each_frame[t]

            # Generate gaze position IDs for the current frame
            is_gradient_checkpointing = self.gaze_decoder.is_gradient_checkpointing
            if is_gradient_checkpointing:
                self.gaze_decoder.gradient_checkpointing_disable()
            logits_processor = LogitsProcessorList(list(self.logits_processor))
            if hasattr(self, "feature_transport_bias"):
                if allowed_token_ids is None:
                    raise ValueError("Feature transport requires explicit allowed_token_ids")
                if t:
                    previous_appearance = (
                        video_embeds[t - 1] - self.connector.pos_embed[None]
                    )
                    current_appearance = video_embeds[t] - self.connector.pos_embed[None]
                    transport_scores = self.feature_transport_bias.normalized_scores(
                        previous_appearance,
                        current_appearance,
                        previous_gaze_pos_ids,
                        allowed_token_ids,
                    )
                    token_bias = self.feature_transport_bias.token_bias(
                        transport_scores,
                        allowed_token_ids,
                        self.gaze_decoder_config.vocab_size,
                    )
                    logits_processor.append(AdditiveTokenBiasLogitsProcessor(token_bias))
            elif hasattr(self, "recurrent_state_bias"):
                recurrent_states.append(recurrent_state)
                token_bias = self.recurrent_state_bias.token_bias(
                    recurrent_state,
                    allowed_token_ids,
                    self.gaze_decoder_config.vocab_size,
                )
                recurrent_biases.append(token_bias)
                logits_processor.append(AdditiveTokenBiasLogitsProcessor(token_bias))
            if not allow_eos:
                logits_processor.append(NoEosTokenLogitsProcessor())
            elif min_gaze_tokens_each_frame:
                logits_processor.append(
                    MinimumGazeTokensLogitsProcessor(
                        self.gaze_decoder_config.eos_token_id,
                        min_gaze_tokens_each_frame,
                    )
                )
            if allowed_token_ids is not None:
                logits_processor.append(AllowedTokensLogitsProcessor(allowed_token_ids))
            gaze_outputs = self.gaze_decoder.generate(
                inputs_embeds=torch.cat(inputs_embeds, dim=1),  # We need to pass the whole sequence of inputs_embeds (both current and past) to the model even when we use use_cache=True!!!
                attention_mask=torch.cat(attention_mask, dim=1),
                position_ids=torch.cat(attention_mask, dim=1).cumsum(dim=-1) - 1,
                max_new_tokens=max_gaze_tokens,
                logits_processor=logits_processor,
                pad_token_id=self.gaze_decoder_config.eos_token_id,
                eos_token_id=self.gaze_decoder_config.eos_token_id,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict_in_generate=True,
                generation_config=generation_config,
                **generation_kwargs,
            )
            if is_gradient_checkpointing:
                self.gaze_decoder.gradient_checkpointing_enable()

            # Get the predicted gaze ids
            gaze_pos_ids = gaze_outputs.sequences  # B * N
            gaze_pos_ids_list.append(gaze_pos_ids + self.num_vision_tokens_each_frame * t)
            previous_gaze_pos_ids = gaze_pos_ids
            if hasattr(self, "recurrent_state_bias"):
                recurrent_state = self.recurrent_state_bias.update(
                    recurrent_state,
                    video_embeds[t],
                    gaze_pos_ids,
                    allowed_token_ids,
                )

            # Update inputs_embeds for the next frame
            inputs_embeds.append(self.gaze_decoder.model.embed_tokens(gaze_pos_ids))

            # Update past_key_values for the next frame
            past_key_values = gaze_outputs.past_key_values

            # Update auxiliary information
            num_gazing_each_frame.append(gaze_pos_ids.shape[1])
            if_padded_gazing.append(
                eos_padding_mask(
                    gaze_pos_ids,
                    self.gaze_decoder_config.eos_token_id,
                    first_eos_is_action=allow_eos,
                )
            )

            # Update attention mask
            attention_mask.append((gaze_pos_ids != self.gaze_decoder_config.eos_token_id).to(torch.long))

        if hasattr(self, "recurrent_state_bias"):
            self.recurrent_state_bias.record_diagnostics(
                torch.stack(recurrent_states, dim=1),
                torch.stack(recurrent_biases, dim=1),
            )

        # Concatenate gaze position IDs from all frames
        gaze_pos_ids = torch.cat(gaze_pos_ids_list, dim=1)

        # Get auxiliary information
        num_gazing_each_frame = torch.tensor(num_gazing_each_frame, device=gaze_pos_ids.device).to(torch.long)
        if_padded_gazing = torch.cat(if_padded_gazing, dim=1)

        to_return = {
            "gazing_pos": gaze_pos_ids,  # In gaze_pos_ids, the padded gazing positions are not necessarily eos_token_id, so one needs to use if_padded_gazing to determine if the gazing position is padded!!!
            "num_gazing_each_frame": num_gazing_each_frame,
            "if_padded_gazing": if_padded_gazing,
            "task_loss_requirement": task_loss_requirement,
            "past_input_embeds": inputs_embeds if use_cache else None,
            "past_attention_mask": attention_mask if use_cache else None,
            "past_key_values": past_key_values if use_cache else None,
            "past_conv_values": past_conv_values if use_cache else None,
        }
        return to_return

    def forward(
        self,
        video,
        gazing_info,
        allowed_token_ids=None,
        allow_eos=False,
        min_gaze_tokens_each_frame=0,
        **kwargs,
    ):
        # Unpack gazing_info
        gaze_pos_ids = gazing_info["gazing_pos"]
        num_gazing_each_frame = gazing_info["num_gazing_each_frame"]
        if_padded_gazing = gazing_info["if_padded_gazing"]
        
        # Subsample frames and resize
        B, T = video.shape[:2]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        video = F.interpolate(video, size=(self.input_img_size, self.input_img_size), mode="bicubic", align_corners=False)
        video = rearrange(video, '(b t) c h w -> b t c h w', b=B)

        # Split the gaze frame-wise
        gaze_pos_ids_split = list(gaze_pos_ids.split(num_gazing_each_frame.tolist(), dim=1))
        gaze_pos_ids_split = [gaze_pos_ids_split[t] - self.num_vision_tokens_each_frame * t for t in range(len(gaze_pos_ids_split))]
        if_padded_gazing_split = list(if_padded_gazing.split(num_gazing_each_frame.tolist(), dim=1))

        # Fill the padded gazing positions with eos_token_id
        gaze_pos_ids_split = [gaze_pos * (~padded) + self.gaze_decoder_config.eos_token_id * padded for gaze_pos, padded in zip(gaze_pos_ids_split, if_padded_gazing_split)]

        # Embed the video and gaze position IDs
        inputs_embeds, gaze_token_mask, gaze_pred_source_relative, attention_mask, _ = self.embed(video=video, gaze_pos_ids=gaze_pos_ids_split)
        inputs_embeds = torch.cat(inputs_embeds, dim=1)  # B * N * C
        gaze_token_mask = torch.cat(gaze_token_mask, dim=0)  # N
        gaze_pred_source_relative = torch.cat(gaze_pred_source_relative, dim=0)  # N
        attention_mask = torch.cat(attention_mask, dim=1)  # B * N

        effective_allowed_ids = (
            list(allowed_token_ids)
            if allowed_token_ids is not None
            else list(range(self.gaze_decoder_config.vocab_size))
        )
        if not allow_eos:
            effective_allowed_ids = [
                token_id
                for token_id in effective_allowed_ids
                if token_id != self.gaze_decoder_config.eos_token_id
            ]

        # Run model forward
        outputs = self.gaze_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=attention_mask.cumsum(dim=-1) - 1,
            **kwargs,
        )

        # Get gaze logits and probs
        logits_multi_token_pred = outputs.logits
        task_loss_prediction_multi_token_pred = outputs.task_loss_prediction  # B * N * num_multi_token_pred
        logits_multi_token_pred = rearrange(logits_multi_token_pred, 'b n (k c) -> b n k c', k=self.num_multi_token_pred)
        if hasattr(self, "feature_transport_bias"):
            if allowed_token_ids is None:
                raise ValueError("Feature transport requires explicit allowed_token_ids")
            vision_features = inputs_embeds[:, gaze_token_mask == 0].reshape(
                B,
                len(gaze_pos_ids_split),
                self.connector.num_tokens,
                -1,
            )
            appearance_features = (
                vision_features - self.connector.pos_embed[None, None]
            )
            frame_biases = [
                torch.zeros(
                    B,
                    self.gaze_decoder_config.vocab_size,
                    device=inputs_embeds.device,
                    dtype=logits_multi_token_pred.dtype,
                )
            ]
            for frame_index in range(1, len(gaze_pos_ids_split)):
                transport_scores = self.feature_transport_bias.normalized_scores(
                    appearance_features[:, frame_index - 1],
                    appearance_features[:, frame_index],
                    gaze_pos_ids_split[frame_index - 1],
                    effective_allowed_ids,
                )
                frame_biases.append(
                    self.feature_transport_bias.token_bias(
                        transport_scores,
                        effective_allowed_ids,
                        self.gaze_decoder_config.vocab_size,
                    )
                )
            logits_multi_token_pred = self.feature_transport_bias.apply_to_predictions(
                logits_multi_token_pred,
                torch.stack(frame_biases, dim=1),
                gaze_token_mask,
                gaze_pred_source_relative,
                num_gazing_each_frame,
            )
        elif hasattr(self, "recurrent_state_bias"):
            if allowed_token_ids is None:
                raise ValueError("Recurrent state requires explicit allowed_token_ids")
            vision_features = inputs_embeds[:, gaze_token_mask == 0].reshape(
                B,
                len(gaze_pos_ids_split),
                self.connector.num_tokens,
                -1,
            )
            frame_biases, recurrent_states = self.recurrent_state_bias.sequence_biases(
                vision_features,
                gaze_pos_ids_split,
                effective_allowed_ids,
                self.gaze_decoder_config.vocab_size,
            )
            logits_multi_token_pred = self.recurrent_state_bias.apply_to_predictions(
                logits_multi_token_pred,
                frame_biases,
                gaze_token_mask,
                gaze_pred_source_relative,
                num_gazing_each_frame,
            )
            self.recurrent_state_bias.record_diagnostics(
                recurrent_states,
                frame_biases,
            )
        masked_logits = torch.full_like(logits_multi_token_pred, -float("inf"))
        masked_logits[..., effective_allowed_ids] = logits_multi_token_pred[
            ..., effective_allowed_ids
        ]
        logits_multi_token_pred = masked_logits
        gaze_probs_all_multi_token_pred = F.softmax(logits_multi_token_pred, dim=-1)

        shifted_probs = []
        shifted_task_loss_prediction = []
        for i in range(self.num_multi_token_pred):
            shifted_probs.append(F.pad(gaze_probs_all_multi_token_pred[:, :-(i + 1), i, :], (0, 0, i + 1, 0), value=0))
            shifted_task_loss_prediction.append(F.pad(task_loss_prediction_multi_token_pred[:, :task_loss_prediction_multi_token_pred.shape[1] - i, i], (i, 0), value=0))
        shifted_probs = torch.stack(shifted_probs, dim=2)  # B, N, K, C
        shifted_task_loss_prediction = torch.stack(shifted_task_loss_prediction, dim=2)  # B, N, K

        gaze_probs_all = shifted_probs[:, torch.arange(logits_multi_token_pred.shape[1]), -gaze_pred_source_relative - 1]
        task_loss_prediction = shifted_task_loss_prediction[:, torch.arange(logits_multi_token_pred.shape[1]), (-gaze_pred_source_relative) % self.num_multi_token_pred]  # B, N

        gaze_input_token_pos = torch.nonzero(gaze_token_mask, as_tuple=True)[0]
        gaze_probs_all = gaze_probs_all[:, gaze_input_token_pos, :]
        task_loss_prediction = task_loss_prediction[:, gaze_input_token_pos]
        if allow_eos:
            gaze_probs_all = mask_eos_before_minimum(
                gaze_probs_all,
                gaze_pos_ids_split,
                self.gaze_decoder_config.eos_token_id,
                min_gaze_tokens_each_frame,
            )
        gaze_probs_all = mask_previously_selected(
            gaze_probs_all,
            gaze_pos_ids_split,
        )
        gaze_log_probs_all = torch.log(
            gaze_probs_all[..., effective_allowed_ids] + 1e-8
        )
        B, N = gaze_probs_all.shape[:2]
        gaze_probs = gaze_probs_all.reshape(B * N, -1)[torch.arange(B * N), torch.cat(gaze_pos_ids_split, dim=1).flatten()].reshape(B, N)  # [B, T]


        outputs = AutoGazeOutput(
            gaze_probs=gaze_probs,
            gaze_log_probs_all=gaze_log_probs_all,
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            task_loss_prediction=task_loss_prediction,
        )
        return outputs


################ Shallow Vision Encoder #################

class Conv3dBlockForStreaming(nn.Module):
    def __init__(self, hidden_dim, temporal_patch_size, spatial_kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_patch_size = temporal_patch_size
        self.spatial_kernel_size = spatial_kernel_size

        self.conv3d = nn.Conv3d(
            hidden_dim, hidden_dim, 
            kernel_size=(temporal_patch_size, spatial_kernel_size, spatial_kernel_size), 
            padding=(0, (spatial_kernel_size - 1) // 2, (spatial_kernel_size - 1) // 2),  # We manually pad the temporal dimension in forward, to support streaming
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, use_cache=False, past_conv_values=None):
        if not (use_cache and past_conv_values is not None):
            x = F.pad(x, (0, 0, 0, 0, self.temporal_patch_size - 1, 0), value=0)
        else:
            x = torch.cat([past_conv_values, x], dim=2)
        new_past_conv_values = x[:, :, -(self.temporal_patch_size - 1):]

        x = self.conv3d(x)

        x = self.relu(x)

        return x, new_past_conv_values


class ShallowVideoConvNet(nn.Module):
    """
    A shallow video convolutional network for video gaze modeling, inspired by ViViT's patch embedding approach.
    Expects input of shape (B, T, C, H, W) or (B*T, C, H, W).
    """
    def __init__(self, config: VisionModelConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        out_dim = config.out_dim
        depth = config.depth
        kernel_size = config.kernel_size
        self.temporal_patch_size = getattr(config, "temporal_patch_size", 1)

        # For video, first merge temporal and batch if needed, then apply 3D conv for temporal patching
        self.temporal_conv = nn.Conv3d(
            in_channels=3,  # RGB
            out_channels=hidden_dim,
            kernel_size=(self.temporal_patch_size, kernel_size, kernel_size),
            stride=(self.temporal_patch_size, kernel_size, kernel_size),
            bias=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.trunk_temporal_kernel_size = config.trunk_temporal_kernel_size
        self.trunk_spatial_kernel_size = config.trunk_spatial_kernel_size
        blocks = []
        for i in range(depth):
            blocks.append(
                Conv3dBlockForStreaming(
                    hidden_dim=hidden_dim,
                    temporal_patch_size=self.trunk_temporal_kernel_size,
                    spatial_kernel_size=self.trunk_spatial_kernel_size,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.out_proj = nn.Conv3d(
            hidden_dim, out_dim, kernel_size=1, stride=1, bias=True
        )

    def forward(self, x, use_cache=False, past_conv_values=None):
        # x: (B, T, C, H, W) or (B*T, C, H, W)
        if x.dim() == 5:
            # (B, T, C, H, W) -> (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        elif x.dim() == 4:
            # (B*T, C, H, W) -> (B*T, C, 1, H, W)
            x = x.unsqueeze(2)
        else:
            raise ValueError("Input must be 4D or 5D tensor")
        x = self.temporal_conv(x)  # (B, hidden_dim, T', H', W')
        # Collapse temporal dimension into batch for normalization and blocks
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)  # (B*T, C, H, W)
        # Flatten spatial dims for norm: (B*T, C, H*W)
        x = x.view(B * T, C, -1).permute(0, 2, 1)  # (B*T, H*W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous().view(B * T, C, H, W)  # (B*T, C, H, W)
        # Reshape back to (B, C, T, H, W)
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        # Main trunk
        new_past_conv_values = []
        for i, block in enumerate(self.blocks):
            x, new_past_conv_values_i = block(
                x, 
                use_cache=use_cache, 
                past_conv_values=past_conv_values[i] if use_cache and past_conv_values is not None else None
            )
            new_past_conv_values.append(new_past_conv_values_i)
        x = self.out_proj(x)
        # Output shape: (B, out_dim, T', H', W')
        return x, new_past_conv_values


################ Connector Between Vision Encoder and Gaze Model #################

class Connector(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_tokens = config.num_tokens

        self.pos_embed = nn.Parameter(torch.randn(self.num_tokens, self.hidden_dim))

    def forward(self, x):
        """
        x: (B, T, N, C)
        """
        x = x + self.pos_embed[None, None]
        return x


class TemporalPositionSignal(nn.Module):
    """RMS-controlled deterministic relative-time signal with one scalar gate."""

    def __init__(self, hidden_dim: int, max_frequency: float = 8.0):
        super().__init__()
        if hidden_dim < 2 or max_frequency < 1:
            raise ValueError("hidden_dim must be >= 2 and max_frequency must be >= 1")
        self.hidden_dim = int(hidden_dim)
        self.max_frequency = float(max_frequency)
        self.gate = nn.Parameter(torch.zeros(()))

    def encoding(self, num_frames: int, *, device, dtype):
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        positions = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=torch.float32)
        pairs = math.ceil(self.hidden_dim / 2)
        frequencies = torch.logspace(
            0.0,
            math.log10(self.max_frequency),
            pairs,
            device=device,
            dtype=torch.float32,
        )
        angles = 2.0 * math.pi * positions[:, None] * frequencies[None]
        signal = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[
            :, : self.hidden_dim
        ]
        signal = signal / signal.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return signal.to(dtype=dtype)

    def forward(self, features):
        if features.ndim != 4 or features.shape[-1] != self.hidden_dim:
            raise ValueError("Expected features shaped (batch, frames, tokens, hidden_dim)")
        signal = self.encoding(
            features.shape[1], device=features.device, dtype=features.dtype
        )[None, :, None, :]
        feature_rms = features.detach().square().mean(dim=(-2, -1), keepdim=True).sqrt()
        return features + self.gate.to(features.dtype) * feature_rms * signal

    def diagnostics(self):
        return {
            "temporal_position_gate": self.gate.detach(),
            "temporal_signal_to_feature_rms": self.gate.detach().abs(),
        }


class CausalConnectorDifferenceSignal(nn.Module):
    """Framewise RMS-normalized causal connector difference with one gate."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def normalized_difference(self, features):
        if features.ndim != 4 or features.shape[-1] != self.hidden_dim:
            raise ValueError("Expected features shaped (batch, frames, tokens, hidden_dim)")
        fixed_features = features.detach()
        difference = torch.zeros_like(fixed_features)
        difference[:, 1:] = fixed_features[:, 1:] - fixed_features[:, :-1]
        difference_rms = difference.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        return torch.where(
            difference_rms > 1e-8,
            difference / difference_rms.clamp_min(1e-8),
            torch.zeros_like(difference),
        )

    def forward(self, features):
        signal = self.normalized_difference(features)
        feature_rms = features.detach().square().mean(dim=(-2, -1), keepdim=True).sqrt()
        return features + self.gate.to(features.dtype) * feature_rms * signal

    def diagnostics(self):
        return {
            "causal_difference_gate": self.gate.detach(),
            "causal_difference_signal_to_feature_rms_ceiling": self.gate.detach().abs(),
        }


class CausalFeatureTransportLogitBias(nn.Module):
    """Transport previous gaze through frozen appearance-feature similarity."""

    def __init__(self, hidden_dim: int, temperature: float = 0.1):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.gate = nn.Parameter(torch.zeros(()))

    def normalized_scores(
        self,
        previous_features,
        current_features,
        previous_token_ids,
        allowed_token_ids,
    ):
        if previous_features.ndim != 3 or current_features.shape != previous_features.shape:
            raise ValueError("features must share shape (batch, tokens, hidden_dim)")
        if previous_features.shape[-1] != self.hidden_dim:
            raise ValueError("feature hidden dimension does not match the module")
        if previous_token_ids.ndim != 2 or previous_token_ids.shape[0] != previous_features.shape[0]:
            raise ValueError("previous_token_ids must have shape (batch, selections)")
        allowed = torch.as_tensor(allowed_token_ids, device=current_features.device, dtype=torch.long)
        if allowed.numel() < 2:
            raise ValueError("at least two allowed candidates are required")
        num_tokens = previous_features.shape[1]
        action_offset = int(allowed[0])
        expected = torch.arange(
            action_offset,
            action_offset + num_tokens,
            device=allowed.device,
        )
        if not torch.equal(allowed, expected):
            raise ValueError(
                "feature transport requires one contiguous fine-action range "
                "matching the visual feature grid"
            )

        fixed_previous = F.normalize(previous_features.detach().float(), dim=-1)
        fixed_current = F.normalize(current_features.detach().float(), dim=-1)
        previous_feature_ids = previous_token_ids - action_offset
        valid_previous = (previous_feature_ids >= 0) & (previous_feature_ids < num_tokens)
        safe_previous = previous_feature_ids.clamp(0, num_tokens - 1)
        selected_previous = fixed_previous.gather(
            1,
            safe_previous[..., None].expand(-1, -1, self.hidden_dim),
        )
        similarity = torch.einsum("bic,bkc->bik", fixed_current, selected_previous)
        similarity = similarity.masked_fill(~valid_previous[:, None], -float("inf"))
        pooled = torch.logsumexp(similarity / self.temperature, dim=-1)
        has_previous = valid_previous.any(dim=-1, keepdim=True)
        pooled = torch.where(has_previous, pooled, torch.zeros_like(pooled))
        centered = pooled - pooled.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
        normalized = torch.where(
            has_previous & (rms > 1e-8),
            centered / rms.clamp_min(1e-8),
            torch.zeros_like(centered),
        )
        return normalized.to(current_features.dtype)

    def token_bias(self, normalized_scores, allowed_token_ids, vocab_size):
        allowed = torch.as_tensor(
            allowed_token_ids,
            device=normalized_scores.device,
            dtype=torch.long,
        )
        if normalized_scores.shape[-1] != allowed.numel():
            raise ValueError("score count must equal the number of allowed tokens")
        bias = torch.zeros(
            *normalized_scores.shape[:-1],
            vocab_size,
            device=normalized_scores.device,
            dtype=normalized_scores.dtype,
        )
        bias[..., allowed] = self.gate.to(normalized_scores.dtype) * normalized_scores
        return bias

    def apply_to_predictions(
        self,
        logits,
        frame_biases,
        gaze_token_mask,
        gaze_pred_source_relative,
        num_gazing_each_frame,
    ):
        gaze_positions = torch.nonzero(gaze_token_mask, as_tuple=True)[0]
        relative = gaze_pred_source_relative[gaze_positions]
        source_positions = gaze_positions + relative
        head_indices = -relative - 1
        frame_indices = torch.repeat_interleave(
            torch.arange(len(num_gazing_each_frame), device=logits.device),
            num_gazing_each_frame,
        )
        result = logits.clone()
        result[:, source_positions, head_indices, :] = (
            result[:, source_positions, head_indices, :]
            + frame_biases[:, frame_indices, :]
        )
        return result

    def diagnostics(self):
        return {
            "feature_transport_logit_gate": self.gate.detach(),
            "feature_transport_logit_bias_rms": self.gate.detach().abs(),
        }


class SelectionConditionedRecurrentStateLogitBias(nn.Module):
    """Carry a learned clip-local state across completed frame selections."""

    def __init__(self, input_dim: int, num_tokens: int, state_hidden_dim: int):
        super().__init__()
        if input_dim < 1 or num_tokens < 2 or state_hidden_dim < 1:
            raise ValueError("input_dim, num_tokens, and state_hidden_dim must be positive")
        if input_dim != state_hidden_dim:
            raise ValueError(
                "R6 keeps input and state widths equal to avoid an extra projection"
            )
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.state_hidden_dim = int(state_hidden_dim)
        self.state_cell = nn.GRUCell(self.input_dim, self.state_hidden_dim)
        self.action_readout = nn.Linear(
            self.state_hidden_dim,
            self.num_tokens,
            bias=False,
        )
        self.gate = nn.Parameter(torch.zeros(()))
        self._last_diagnostics = {}

    def initial_state(self, batch_size: int, *, device, dtype):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            batch_size,
            self.state_hidden_dim,
            device=device,
            dtype=dtype,
        )

    def _action_mapping(self, allowed_token_ids, *, device):
        allowed = torch.as_tensor(allowed_token_ids, device=device, dtype=torch.long)
        if allowed.numel() != self.num_tokens:
            raise ValueError("allowed actions must match the recurrent spatial readout")
        offset = int(allowed[0])
        expected = torch.arange(offset, offset + self.num_tokens, device=device)
        if not torch.equal(allowed, expected):
            raise ValueError(
                "recurrent state requires one contiguous fine-action range"
            )
        return allowed, offset

    def selected_summary(self, frame_features, completed_token_ids, allowed_token_ids):
        if frame_features.ndim != 3 or frame_features.shape[-1] != self.input_dim:
            raise ValueError("frame_features must have shape (batch, tokens, input_dim)")
        if frame_features.shape[1] != self.num_tokens:
            raise ValueError("frame feature count does not match the recurrent readout")
        if completed_token_ids.ndim != 2:
            raise ValueError("completed_token_ids must have shape (batch, selections)")
        if completed_token_ids.shape[0] != frame_features.shape[0]:
            raise ValueError("feature and selection batches do not match")
        _, action_offset = self._action_mapping(
            allowed_token_ids,
            device=frame_features.device,
        )
        feature_ids = completed_token_ids - action_offset
        valid = (feature_ids >= 0) & (feature_ids < self.num_tokens)
        safe_ids = feature_ids.clamp(0, self.num_tokens - 1)
        selected = frame_features.detach().gather(
            1,
            safe_ids[..., None].expand(-1, -1, self.input_dim),
        )
        weights = valid[..., None].to(selected.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        return (selected * weights).sum(dim=1) / count

    def update(self, state, frame_features, completed_token_ids, allowed_token_ids):
        if state.ndim != 2 or state.shape[-1] != self.state_hidden_dim:
            raise ValueError("state must have shape (batch, state_hidden_dim)")
        summary = self.selected_summary(
            frame_features,
            completed_token_ids,
            allowed_token_ids,
        )
        return self.state_cell(summary, state)

    def normalized_action_scores(self, state):
        scores = self.action_readout(state)
        centered = scores - scores.mean(dim=-1, keepdim=True)
        mean_square = centered.square().mean(dim=-1, keepdim=True)
        return torch.where(
            mean_square > 1e-16,
            centered / (mean_square + 1e-8).sqrt(),
            torch.zeros_like(centered),
        )

    def token_bias(self, state, allowed_token_ids, vocab_size):
        allowed, _ = self._action_mapping(allowed_token_ids, device=state.device)
        scores = self.normalized_action_scores(state)
        bias = torch.zeros(
            state.shape[0],
            vocab_size,
            device=state.device,
            dtype=state.dtype,
        )
        bias[..., allowed] = (
            self.gate.to(state.dtype) * scores
        ).to(state.dtype)
        return bias

    def sequence_biases(
        self,
        frame_features,
        completed_token_ids_by_frame,
        allowed_token_ids,
        vocab_size,
        *,
        reset_each_frame=False,
        history_features=None,
        history_token_ids_by_frame=None,
    ):
        if frame_features.ndim != 4 or frame_features.shape[-2:] != (
            self.num_tokens,
            self.input_dim,
        ):
            raise ValueError(
                "frame_features must have shape (batch, frames, tokens, input_dim)"
            )
        if len(completed_token_ids_by_frame) != frame_features.shape[1]:
            raise ValueError("one completed selection tensor is required per frame")
        update_features = frame_features if history_features is None else history_features
        update_tokens = (
            completed_token_ids_by_frame
            if history_token_ids_by_frame is None
            else history_token_ids_by_frame
        )
        if update_features.shape != frame_features.shape:
            raise ValueError("alternative history features must match the current clip shape")
        if len(update_tokens) != frame_features.shape[1]:
            raise ValueError("alternative history must provide one selection tensor per frame")

        state = self.initial_state(
            frame_features.shape[0],
            device=frame_features.device,
            dtype=frame_features.dtype,
        )
        states = []
        biases = []
        for frame_index in range(frame_features.shape[1]):
            states.append(state)
            biases.append(self.token_bias(state, allowed_token_ids, vocab_size))
            if reset_each_frame:
                state = torch.zeros_like(state)
            else:
                state = self.update(
                    state,
                    update_features[:, frame_index],
                    update_tokens[frame_index],
                    allowed_token_ids,
                )
        return torch.stack(biases, dim=1), torch.stack(states, dim=1)

    def apply_to_predictions(
        self,
        logits,
        frame_biases,
        gaze_token_mask,
        gaze_pred_source_relative,
        num_gazing_each_frame,
    ):
        gaze_positions = torch.nonzero(gaze_token_mask, as_tuple=True)[0]
        relative = gaze_pred_source_relative[gaze_positions]
        source_positions = gaze_positions + relative
        head_indices = -relative - 1
        frame_indices = torch.repeat_interleave(
            torch.arange(len(num_gazing_each_frame), device=logits.device),
            num_gazing_each_frame,
        )
        result = logits.clone()
        result[:, source_positions, head_indices, :] = (
            result[:, source_positions, head_indices, :]
            + frame_biases[:, frame_indices, :]
        )
        return result

    def record_diagnostics(self, states, frame_biases):
        fixed_states = states.detach().float()
        fixed_biases = frame_biases.detach().float()
        state_rms = fixed_states.square().mean(dim=-1).sqrt()
        finite = torch.isfinite(fixed_states).all() & torch.isfinite(fixed_biases).all()
        self._last_diagnostics = {
            "recurrent_state_rms_mean": state_rms.mean(),
            "recurrent_state_rms_max": state_rms.max(),
            "recurrent_state_absolute_max": fixed_states.abs().max(),
            "recurrent_state_saturation_fraction": (fixed_states.abs() >= 0.99).float().mean(),
            "recurrent_state_logit_bias_rms": fixed_biases.square().mean().sqrt(),
            "recurrent_state_all_finite": finite.to(fixed_states.dtype),
        }

    def diagnostics(self):
        diagnostics = {
            "recurrent_state_logit_gate": self.gate.detach(),
            "recurrent_state_logit_gate_abs": self.gate.detach().abs(),
        }
        diagnostics.update(self._last_diagnostics)
        return diagnostics
