"""NVILA adapters for the R2d variable-EOS HLVid evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterable, Literal

import torch


Mode = Literal["variable", "forced_k16"]


def summarize_raw_allocation(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert additive allocation counters into the historical summary shape."""

    decoder = raw["decoder"]
    post = raw["post_resolution_adaptation"]
    decoder_observations = int(decoder["frame_observations"])
    post_observations = int(post["frame_observations"])
    histogram = {str(key): int(value) for key, value in decoder["spatial_action_histogram"].items()}
    if sum(histogram.values()) != decoder_observations:
        raise ValueError("Decoder histogram does not cover every frame observation")
    if sum(int(key) * value for key, value in histogram.items()) != int(decoder["spatial_actions_sum"]):
        raise ValueError("Decoder histogram does not match the spatial-action sum")
    valid_action_tokens = int(decoder["valid_action_tokens"])
    if valid_action_tokens != int(decoder["spatial_actions_sum"]) + int(decoder["eos_actions"]):
        raise ValueError("Decoder valid-token count does not match spatial plus EOS actions")
    return {
        "model_calls": int(raw["model_calls"]),
        "decoder": {
            "calls": int(decoder["calls"]),
            "frame_observations": decoder_observations,
            "padded_position_slots": int(decoder["padded_position_slots"]),
            "valid_action_tokens": valid_action_tokens,
            "spatial_actions_per_frame_mean": (
                int(decoder["spatial_actions_sum"]) / decoder_observations
                if decoder_observations
                else None
            ),
            "padded_position_slots_per_frame_mean": (
                int(decoder["padded_position_slots"]) / decoder_observations
                if decoder_observations
                else None
            ),
            "valid_action_tokens_per_frame_mean": (
                int(decoder["valid_action_tokens"]) / decoder_observations
                if decoder_observations
                else None
            ),
            "spatial_actions_per_frame_min": decoder["spatial_actions_min"],
            "spatial_actions_per_frame_max": decoder["spatial_actions_max"],
            "eos_action_rate": (
                int(decoder["eos_actions"]) / decoder_observations
                if decoder_observations
                else None
            ),
            "spatial_action_histogram": histogram,
        },
        "post_resolution_adaptation": {
            "frame_observations": post_observations,
            "padded_position_slots": int(post["padded_position_slots"]),
            "valid_retained_patches": int(post["retained_patches_sum"]),
            "retained_patches_per_frame_mean": (
                int(post["retained_patches_sum"]) / post_observations
                if post_observations
                else None
            ),
            "padded_position_slots_per_frame_mean": (
                int(post["padded_position_slots"]) / post_observations
                if post_observations
                else None
            ),
            "retained_patches_per_frame_min": post["retained_patches_min"],
            "retained_patches_per_frame_max": post["retained_patches_max"],
        },
    }


def merge_raw_allocations(weighted_records: Iterable[tuple[dict[str, Any], int]]) -> dict[str, Any]:
    """Merge per-scope raw counters, optionally weighting repeated identical scopes."""

    merged = {
        "schema_version": 1,
        "model_calls": 0,
        "decoder": {
            "calls": 0,
            "frame_observations": 0,
            "spatial_actions_sum": 0,
            "padded_position_slots": 0,
            "valid_action_tokens": 0,
            "spatial_actions_min": None,
            "spatial_actions_max": None,
            "eos_actions": 0,
            "spatial_action_histogram": {},
        },
        "post_resolution_adaptation": {
            "frame_observations": 0,
            "retained_patches_sum": 0,
            "padded_position_slots": 0,
            "retained_patches_min": None,
            "retained_patches_max": None,
        },
    }
    for raw, weight in weighted_records:
        if int(weight) != weight or weight <= 0:
            raise ValueError("Allocation replay weights must be positive integers")
        if int(raw.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported raw allocation schema")
        weight = int(weight)
        decoder = raw["decoder"]
        post = raw["post_resolution_adaptation"]
        merged["model_calls"] += int(raw["model_calls"]) * weight
        for key in (
            "calls",
            "frame_observations",
            "spatial_actions_sum",
            "padded_position_slots",
            "valid_action_tokens",
            "eos_actions",
        ):
            merged["decoder"][key] += int(decoder[key]) * weight
        for key in ("frame_observations", "retained_patches_sum", "padded_position_slots"):
            merged["post_resolution_adaptation"][key] += int(post[key]) * weight
        for value, count in decoder["spatial_action_histogram"].items():
            histogram = merged["decoder"]["spatial_action_histogram"]
            histogram[str(value)] = histogram.get(str(value), 0) + int(count) * weight
        for destination, source in (
            ("spatial_actions_min", decoder.get("spatial_actions_min")),
            ("retained_patches_min", post.get("retained_patches_min")),
        ):
            if source is not None:
                section = "decoder" if destination.startswith("spatial") else "post_resolution_adaptation"
                current = merged[section][destination]
                merged[section][destination] = int(source) if current is None else min(current, int(source))
        for destination, source in (
            ("spatial_actions_max", decoder.get("spatial_actions_max")),
            ("retained_patches_max", post.get("retained_patches_max")),
        ):
            if source is not None:
                section = "decoder" if destination.startswith("spatial") else "post_resolution_adaptation"
                current = merged[section][destination]
                merged[section][destination] = int(source) if current is None else max(current, int(source))
    summarize_raw_allocation(merged)
    return merged


@dataclass
class R2DHLVidCallStats:
    """Streaming decoder and post-resolution token-count diagnostics."""

    model_calls: int = 0
    decoder_calls: int = 0
    decoder_frame_observations: int = 0
    decoder_spatial_sum: int = 0
    decoder_position_slots: int = 0
    decoder_valid_action_tokens: int = 0
    decoder_spatial_min: int | None = None
    decoder_spatial_max: int | None = None
    decoder_eos_actions: int = 0
    decoder_spatial_histogram: Counter[int] = field(default_factory=Counter)
    post_frame_observations: int = 0
    post_retained_sum: int = 0
    post_position_slots: int = 0
    post_retained_min: int | None = None
    post_retained_max: int | None = None

    def reset(self) -> None:
        """Reset counters after a durable per-example/per-video snapshot."""

        self.model_calls = 0
        self.decoder_calls = 0
        self.decoder_frame_observations = 0
        self.decoder_spatial_sum = 0
        self.decoder_position_slots = 0
        self.decoder_valid_action_tokens = 0
        self.decoder_spatial_min = None
        self.decoder_spatial_max = None
        self.decoder_eos_actions = 0
        self.decoder_spatial_histogram.clear()
        self.post_frame_observations = 0
        self.post_retained_sum = 0
        self.post_position_slots = 0
        self.post_retained_min = None
        self.post_retained_max = None

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
            self.decoder_position_slots += positions.numel()
            self.decoder_valid_action_tokens += int((~padded).sum())
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
            self.post_position_slots += sum(counts)
        else:
            padded = torch.as_tensor(padded_value).detach().to(device="cpu", dtype=torch.bool)
            if padded.ndim != 2 or int(widths.sum()) != padded.shape[1]:
                raise ValueError("Post-resolution frame widths do not match the padding mask")
            self.post_position_slots += padded.numel()
            counts = []
            for frame_padded in padded.split(widths.tolist(), dim=1):
                counts.extend(int(value) for value in (~frame_padded).sum(dim=1).tolist())
        self.post_frame_observations += len(counts)
        self.post_retained_sum += sum(counts)
        self.post_retained_min, self.post_retained_max = self._update_range(
            self.post_retained_min, self.post_retained_max, counts
        )

    def as_raw_dict(self) -> dict[str, Any]:
        """Return additive integer counters suitable for resume-safe merging."""

        return {
            "schema_version": 1,
            "model_calls": self.model_calls,
            "decoder": {
                "calls": self.decoder_calls,
                "frame_observations": self.decoder_frame_observations,
                "spatial_actions_sum": self.decoder_spatial_sum,
                "padded_position_slots": self.decoder_position_slots,
                "valid_action_tokens": self.decoder_valid_action_tokens,
                "spatial_actions_min": self.decoder_spatial_min,
                "spatial_actions_max": self.decoder_spatial_max,
                "eos_actions": self.decoder_eos_actions,
                "spatial_action_histogram": {
                    str(key): value for key, value in sorted(self.decoder_spatial_histogram.items())
                },
            },
            "post_resolution_adaptation": {
                "frame_observations": self.post_frame_observations,
                "retained_patches_sum": self.post_retained_sum,
                "padded_position_slots": self.post_position_slots,
                "retained_patches_min": self.post_retained_min,
                "retained_patches_max": self.post_retained_max,
            },
        }

    def pop_raw_dict(self) -> dict[str, Any]:
        """Return the current additive counters and start a fresh scope."""

        raw = self.as_raw_dict()
        self.reset()
        return raw

    def as_dict(self) -> dict[str, Any]:
        return summarize_raw_allocation(self.as_raw_dict())


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
