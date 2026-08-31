"""NVILA adapters for the R2d variable-EOS HLVid evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Literal

import torch


Mode = Literal["variable", "forced_k16"]


@dataclass
class R2DHLVidCallStats:
    """Streaming decoder and post-resolution token-count diagnostics."""

    model_calls: int = 0
    decoder_calls: int = 0
    decoder_frame_observations: int = 0
    decoder_spatial_sum: int = 0
    decoder_spatial_min: int | None = None
    decoder_spatial_max: int | None = None
    decoder_eos_actions: int = 0
    decoder_spatial_histogram: Counter[int] = field(default_factory=Counter)
    post_frame_observations: int = 0
    post_retained_sum: int = 0
    post_retained_min: int | None = None
    post_retained_max: int | None = None

    @staticmethod
    def _update_range(current_min, current_max, values):
        value_min = min(values)
        value_max = max(values)
        return (
            value_min if current_min is None else min(current_min, value_min),
            value_max if current_max is None else max(current_max, value_max),
        )

    def update_decoder(
        self,
        output: dict[str, Any],
        *,
        fine_action_offset: int,
        actions_per_frame: int,
    ) -> None:
        self.decoder_calls += 1
        positions = torch.as_tensor(output["gazing_pos"]).detach().to(device="cpu", dtype=torch.long)
        padded = torch.as_tensor(output["if_padded_gazing"]).detach().to(device="cpu", dtype=torch.bool)
        widths = torch.as_tensor(output["num_gazing_each_frame"]).detach().to(device="cpu", dtype=torch.long)
        if positions.ndim != 2 or padded.shape != positions.shape or widths.ndim != 1:
            raise ValueError("Unexpected decoder trajectory shapes")
        if int(widths.sum()) != positions.shape[1]:
            raise ValueError("Decoder frame widths do not match the trajectory")

        counts: list[int] = []
        eos_count = 0
        for frame_index, (frame_positions, frame_padded) in enumerate(
            zip(positions.split(widths.tolist(), dim=1), padded.split(widths.tolist(), dim=1))
        ):
            local = frame_positions - frame_index * actions_per_frame
            spatial = (local >= fine_action_offset) & (local < actions_per_frame) & ~frame_padded
            eos = (local == actions_per_frame) & ~frame_padded
            invalid = ~frame_padded & ~spatial & ~eos
            if invalid.any():
                raise ValueError("Decoder emitted an action outside the R2d contract")
            if (eos.sum(dim=1) > 1).any():
                raise ValueError("Decoder emitted multiple EOS actions for one frame")
            counts.extend(int(value) for value in spatial.sum(dim=1).tolist())
            eos_count += int(eos.sum())

        if counts:
            self.decoder_frame_observations += len(counts)
            self.decoder_spatial_sum += sum(counts)
            self.decoder_eos_actions += eos_count
            self.decoder_spatial_histogram.update(counts)
            self.decoder_spatial_min, self.decoder_spatial_max = self._update_range(
                self.decoder_spatial_min, self.decoder_spatial_max, counts
            )

    def update_post_resolution(self, output: dict[str, Any]) -> None:
        self.model_calls += 1
        widths_value = output.get("num_gazing_each_frame")
        if widths_value is None:
            return
        widths = torch.as_tensor(widths_value).detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if widths.numel() == 0:
            return
        padded_value = output.get("if_padded_gazing")
        if padded_value is None:
            counts = [int(value) for value in widths.tolist()]
        else:
            padded = torch.as_tensor(padded_value).detach().to(device="cpu", dtype=torch.bool)
            if padded.ndim != 2 or int(widths.sum()) != padded.shape[1]:
                raise ValueError("Post-resolution frame widths do not match the padding mask")
            counts = []
            for frame_padded in padded.split(widths.tolist(), dim=1):
                counts.extend(int(value) for value in (~frame_padded).sum(dim=1).tolist())
        self.post_frame_observations += len(counts)
        self.post_retained_sum += sum(counts)
        self.post_retained_min, self.post_retained_max = self._update_range(
            self.post_retained_min, self.post_retained_max, counts
        )

    def as_dict(self) -> dict[str, Any]:
        decoder_mean = (
            self.decoder_spatial_sum / self.decoder_frame_observations
            if self.decoder_frame_observations
            else None
        )
        post_mean = (
            self.post_retained_sum / self.post_frame_observations
            if self.post_frame_observations
            else None
        )
        return {
            "model_calls": self.model_calls,
            "decoder": {
                "calls": self.decoder_calls,
                "frame_observations": self.decoder_frame_observations,
                "spatial_actions_per_frame_mean": decoder_mean,
                "spatial_actions_per_frame_min": self.decoder_spatial_min,
                "spatial_actions_per_frame_max": self.decoder_spatial_max,
                "eos_action_rate": (
                    self.decoder_eos_actions / self.decoder_frame_observations
                    if self.decoder_frame_observations
                    else None
                ),
                "spatial_action_histogram": {
                    str(key): value for key, value in sorted(self.decoder_spatial_histogram.items())
                },
            },
            "post_resolution_adaptation": {
                "frame_observations": self.post_frame_observations,
                "retained_patches_per_frame_mean": post_mean,
                "retained_patches_per_frame_min": self.post_retained_min,
                "retained_patches_per_frame_max": self.post_retained_max,
            },
        }


def install_r2d_hlvid_forward(
    model: torch.nn.Module,
    *,
    mode: Mode,
    eos_logit_bias: float | None = None,
    fine_action_offset: int = 69,
    actions_per_frame: int = 265,
    min_budget: int = 4,
    max_budget: int = 36,
    forced_budget: int = 16,
) -> R2DHLVidCallStats:
    """Install the calibrated variable-EOS or coherent forced-K16 contract."""

    if mode not in {"variable", "forced_k16"}:
        raise ValueError(f"Unsupported R2d HLVid mode: {mode}")
    if not 0 <= fine_action_offset < actions_per_frame:
        raise ValueError("fine_action_offset must lie inside the action vocabulary")
    if not 0 < min_budget <= max_budget <= actions_per_frame - fine_action_offset:
        raise ValueError("Invalid variable-budget range")
    if not 0 < forced_budget <= actions_per_frame - fine_action_offset:
        raise ValueError("Invalid forced budget")
    if mode == "variable" and eos_logit_bias is None:
        raise ValueError("Variable mode requires the calibrated EOS logit bias")

    stats = R2DHLVidCallStats()
    original_forward = model.forward
    original_generate = model.gazing_model.generate

    def measured_generate(_gazing_model, *args, **kwargs):
        output = original_generate(*args, **kwargs)
        stats.update_decoder(
            output,
            fine_action_offset=fine_action_offset,
            actions_per_frame=actions_per_frame,
        )
        return output

    model.gazing_model.generate = MethodType(measured_generate, model.gazing_model)

    if mode == "variable":
        decoder = model.gazing_model.gaze_decoder
        eos_id = int(model.gazing_model.gaze_decoder_config.eos_token_id)
        if eos_id != actions_per_frame:
            raise ValueError(f"Checkpoint EOS id {eos_id} does not match {actions_per_frame}")
        decoder.set_output_token_logit_bias(eos_id, float(eos_logit_bias))
        allowed_token_ids = tuple(range(fine_action_offset, actions_per_frame + 1))
    else:
        allowed_token_ids = tuple(range(fine_action_offset, actions_per_frame))

    def r2d_forward(_model, inputs, *args, **kwargs):
        kwargs["gazing_ratio"] = None
        kwargs["task_loss_requirement"] = None
        kwargs["allowed_token_ids"] = allowed_token_ids
        if mode == "variable":
            kwargs["max_gaze_tokens_each_frame"] = max_budget
            kwargs["min_gaze_tokens_each_frame"] = min_budget
            kwargs["allow_eos"] = True
        else:
            kwargs["max_gaze_tokens_each_frame"] = forced_budget
            kwargs["min_gaze_tokens_each_frame"] = 0
            kwargs["allow_eos"] = False
        output = original_forward(inputs, *args, **kwargs)
        stats.update_post_resolution(output)
        return output

    model.forward = MethodType(r2d_forward, model)
    return stats
