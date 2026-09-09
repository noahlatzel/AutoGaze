"""Adapters and provenance helpers for fixed-budget HLVid evaluation."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


def _synchronize(device: torch.device | str | None) -> None:
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class FrameRead:
    """Observed result of one unique OpenCV frame seek/read."""

    requested_index: int
    set_succeeded: bool
    read_succeeded: bool
    reported_index: int | None


@dataclass(frozen=True)
class UniformDecodeTrace:
    """Auditable mapping from intended uniform indices to decoded frames."""

    video_path: str
    source_frame_count: int
    requested_indices: list[int]
    reads: list[FrameRead]
    failed_indices: list[int]
    mismatched_indices: list[dict[str, int | None]]
    padded_output_frames: int

    @property
    def exact(self) -> bool:
        return not self.failed_indices and not self.mismatched_indices and not self.padded_output_frames

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["unique_requested_indices"] = len(set(self.requested_indices))
        result["exact"] = self.exact
        return result


def _reported_frame_index(position_after_read: float) -> int | None:
    """Convert OpenCV's next-frame cursor to the index just returned."""

    if not math.isfinite(position_after_read):
        return None
    rounded = round(position_after_read)
    if abs(position_after_read - rounded) > 1e-3:
        return None
    return int(rounded) - 1


def inspect_uniform_decode(
    video_path: Path,
    num_frames: int,
    *,
    load_images: bool,
) -> tuple[list[Image.Image], UniformDecodeTrace]:
    """Run the established seek/read protocol and expose its actual mapping.

    ``load_images=True`` preserves the historical decoded tensors for healthy
    videos. Callers can fail closed from ``trace.exact`` instead of silently
    shifting a failed read and repeating the final successful frame.
    """

    import cv2

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if capture.grab():
            break
        frame_count -= 1
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Video has no decodable frames: {video_path}")

    requested = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(np.int64)
    frames_by_index: dict[int, Image.Image] = {}
    decoded: list[Image.Image] = []
    reads: list[FrameRead] = []
    try:
        for raw_index in requested:
            index = int(raw_index)
            if index in frames_by_index:
                decoded.append(frames_by_index[index])
                continue
            set_succeeded = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, index))
            ok, frame = capture.read()
            reported = (
                _reported_frame_index(float(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                if ok
                else None
            )
            reads.append(
                FrameRead(
                    requested_index=index,
                    set_succeeded=set_succeeded,
                    read_succeeded=bool(ok),
                    reported_index=reported,
                )
            )
            if not ok:
                continue
            if load_images:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames_by_index[index] = image
                decoded.append(image)
    finally:
        capture.release()

    failed = [read.requested_index for read in reads if not read.read_succeeded]
    mismatched = [
        {"requested_index": read.requested_index, "reported_index": read.reported_index}
        for read in reads
        if read.read_succeeded and read.reported_index != read.requested_index
    ]
    padded = num_frames - len(decoded) if load_images else len(failed)
    trace = UniformDecodeTrace(
        video_path=str(video_path),
        source_frame_count=frame_count,
        requested_indices=[int(index) for index in requested],
        reads=reads,
        failed_indices=failed,
        mismatched_indices=mismatched,
        padded_output_frames=max(0, padded),
    )
    return decoded, trace


def load_uniform_frames(
    video_path: Path,
    num_frames: int,
    *,
    strict: bool = True,
) -> tuple[list[Image.Image], UniformDecodeTrace]:
    """Decode uniform frames, failing closed on any unverified seek/read."""

    frames, trace = inspect_uniform_decode(video_path, num_frames, load_images=True)
    if strict and not trace.exact:
        raise ValueError(f"Non-exact frame decode for {video_path}: {trace.as_dict()}")
    if not frames:
        raise ValueError(f"Could not extract frames from video: {video_path}")
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames, trace


@dataclass
class FixedBudgetCallStats:
    """Streaming and per-example fixed-budget acquisition telemetry."""

    model_calls: int = 0
    frame_observations: int = 0
    retained_patch_sum: int = 0
    retained_patch_min: int | None = None
    retained_patch_max: int | None = None
    _active: bool = field(default=False, init=False, repr=False)
    _example_calls: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _pending_generations: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def start_example(self) -> None:
        if self._active:
            raise RuntimeError("Previous fixed-budget trace was not finished")
        if self._pending_generations:
            raise RuntimeError("Unpaired decoder-generation trace")
        self._active = True
        self._example_calls = []

    def record_generation(
        self,
        output: dict[str, Any],
        *,
        elapsed_seconds: float,
        actions_per_frame: int,
        exact_budget: int,
        fine_action_offset: int,
    ) -> None:
        positions = torch.as_tensor(output["gazing_pos"]).detach().to("cpu", dtype=torch.long)
        padded = torch.as_tensor(output["if_padded_gazing"]).detach().to("cpu", dtype=torch.bool)
        counts = torch.as_tensor(output["num_gazing_each_frame"]).detach().to("cpu", dtype=torch.long)
        if counts.ndim != 1:
            raise ValueError(f"Expected one decoder count per frame, got {tuple(counts.shape)}")
        if positions.ndim != 2 or padded.shape != positions.shape:
            raise ValueError("Decoder positions/padding must be matching rank-two tensors")
        if int(counts.sum().item()) != positions.shape[1]:
            raise ValueError("Decoder frame counts do not partition generated positions")

        action_batches: list[list[list[int]]] = []
        offset = 0
        for batch_index in range(positions.shape[0]):
            frames: list[list[int]] = []
            offset = 0
            for frame_index, count_tensor in enumerate(counts):
                count = int(count_tensor.item())
                frame_positions = positions[batch_index, offset : offset + count]
                frame_padded = padded[batch_index, offset : offset + count]
                actions = frame_positions[~frame_padded] - frame_index * actions_per_frame
                action_list = [int(value) for value in actions.tolist()]
                if len(action_list) != exact_budget:
                    raise ValueError(
                        f"Expected exactly {exact_budget} valid decoder actions, got {len(action_list)}"
                    )
                if len(set(action_list)) != exact_budget:
                    raise ValueError("Decoder repeated a fixed-budget action within one frame")
                if min(action_list) < fine_action_offset or max(action_list) >= actions_per_frame:
                    raise ValueError("Decoder emitted an action outside the fine-only vocabulary")
                frames.append(action_list)
                offset += count
            action_batches.append(frames)

        self._pending_generations.append(
            {
                "selector_seconds": elapsed_seconds,
                "decoder_batch_size": positions.shape[0],
                "decoder_frames_per_item": len(counts),
                "decoder_padded_lengths_per_frame": [int(value) for value in counts.tolist()],
                "decoder_action_ids": action_batches,
            }
        )

    def update(self, output: dict[str, Any], *, elapsed_seconds: float = 0.0) -> None:
        self.model_calls += 1
        counts = output.get("num_gazing_each_frame")
        if counts is None:
            return
        values = torch.as_tensor(counts).detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if values.numel() == 0:
            return
        current_min = int(values.min().item())
        current_max = int(values.max().item())
        self.frame_observations += int(values.numel())
        self.retained_patch_sum += int(values.sum().item())
        self.retained_patch_min = (
            current_min if self.retained_patch_min is None else min(self.retained_patch_min, current_min)
        )
        self.retained_patch_max = (
            current_max if self.retained_patch_max is None else max(self.retained_patch_max, current_max)
        )

        generation = self._pending_generations.pop(0) if self._pending_generations else None
        if self._active:
            padded = torch.as_tensor(output["if_padded_gazing"]).detach().to("cpu", dtype=torch.bool)
            shared_lengths = torch.as_tensor(output["num_gazing_each_frame"]).detach().to(
                "cpu", dtype=torch.long
            )
            segments = padded.split([int(value) for value in shared_lengths.tolist()], dim=1)
            valid_counts = torch.stack([(~segment).sum(dim=1) for segment in segments], dim=1)
            call = {
                "autogaze_total_seconds": elapsed_seconds,
                "post_recovery_padded_lengths_per_frame": [
                    int(value) for value in shared_lengths.tolist()
                ],
                "post_recovery_valid_counts_per_item_frame": valid_counts.tolist(),
            }
            if generation is not None:
                call.update(generation)
                call["autogaze_nonselector_seconds"] = max(
                    0.0, elapsed_seconds - float(generation["selector_seconds"])
                )
            self._example_calls.append(call)

    def finish_example(self) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("No fixed-budget trace is active")
        if self._pending_generations:
            raise RuntimeError("Unpaired decoder-generation trace")
        self._active = False
        calls = self._example_calls
        self._example_calls = []
        valid_counts = [
            count
            for call in calls
            for item in call["post_recovery_valid_counts_per_item_frame"]
            for count in item
        ]
        padded_lengths = [
            count
            for call in calls
            for count in call["post_recovery_padded_lengths_per_frame"]
        ]
        return {
            "model_calls": len(calls),
            "selector_seconds": sum(float(call.get("selector_seconds", 0.0)) for call in calls),
            "autogaze_total_seconds": sum(float(call["autogaze_total_seconds"]) for call in calls),
            "post_recovery_valid_patch_count": {
                "observations": len(valid_counts),
                "sum": sum(valid_counts),
                "min": min(valid_counts) if valid_counts else None,
                "max": max(valid_counts) if valid_counts else None,
                "mean": sum(valid_counts) / len(valid_counts) if valid_counts else None,
            },
            "post_recovery_padded_length": {
                "observations": len(padded_lengths),
                "sum": sum(padded_lengths),
                "min": min(padded_lengths) if padded_lengths else None,
                "max": max(padded_lengths) if padded_lengths else None,
                "mean": sum(padded_lengths) / len(padded_lengths) if padded_lengths else None,
            },
            "calls": calls,
        }

    def as_dict(self) -> dict[str, int | float | None]:
        mean = None
        if self.frame_observations:
            mean = self.retained_patch_sum / self.frame_observations
        return {
            "model_calls": self.model_calls,
            "frame_observations": self.frame_observations,
            "retained_patches_per_frame_mean": mean,
            "retained_patches_per_frame_min": self.retained_patch_min,
            "retained_patches_per_frame_max": self.retained_patch_max,
        }


def install_fixed_budget_forward(
    model: torch.nn.Module,
    *,
    exact_budget: int,
    fine_action_offset: int = 69,
    actions_per_frame: int = 265,
    static_fine_cells: Sequence[int] | None = None,
) -> FixedBudgetCallStats:
    """Force fine-only exact-K acquisition while retaining NVILA recovery.

    When ``static_fine_cells`` is supplied, those tile-local 14x14 cells are
    injected at the generation boundary before resolution recovery. The
    learned generator and its probability-rescoring pass are both skipped.
    """

    if exact_budget <= 0:
        raise ValueError("exact_budget must be positive")
    if fine_action_offset < 0 or fine_action_offset >= actions_per_frame:
        raise ValueError("fine_action_offset must lie inside the action vocabulary")
    if exact_budget > actions_per_frame - fine_action_offset:
        raise ValueError("exact_budget exceeds the number of available fine actions")
    if static_fine_cells is not None:
        static_fine_cells = tuple(int(value) for value in static_fine_cells)
        if len(static_fine_cells) != exact_budget or len(set(static_fine_cells)) != exact_budget:
            raise ValueError("Static fine cells must contain exactly K unique cells")
        fine_cell_count = actions_per_frame - fine_action_offset
        if min(static_fine_cells) < 0 or max(static_fine_cells) >= fine_cell_count:
            raise ValueError("Static fine cell lies outside the fine grid")

    allowed_token_ids = tuple(range(fine_action_offset, actions_per_frame))
    original_forward = model.forward
    original_generate = model.gazing_model.generate
    stats = FixedBudgetCallStats()

    def generated_or_static(_gazing_model, video, *args, **kwargs):
        device = video.device
        _synchronize(device)
        started = time.perf_counter()
        if static_fine_cells is None:
            output = original_generate(video, *args, **kwargs)
        else:
            batch_size, input_frames = video.shape[:2]
            output_frames = input_frames // int(_gazing_model.frame_sampling_rate)
            actions = torch.tensor(
                [fine_action_offset + cell for cell in static_fine_cells],
                device=device,
                dtype=torch.long,
            )
            positions = torch.cat(
                [actions + frame_index * actions_per_frame for frame_index in range(output_frames)]
            ).unsqueeze(0).expand(batch_size, -1).contiguous()
            output = {
                "gazing_pos": positions,
                "num_gazing_each_frame": torch.full(
                    (output_frames,), exact_budget, device=device, dtype=torch.long
                ),
                "if_padded_gazing": torch.zeros_like(positions, dtype=torch.bool),
                "task_loss_requirement": None,
            }
        _synchronize(device)
        elapsed = time.perf_counter() - started
        stats.record_generation(
            output,
            elapsed_seconds=elapsed,
            actions_per_frame=actions_per_frame,
            exact_budget=exact_budget,
            fine_action_offset=fine_action_offset,
        )
        return output

    def fixed_forward(_model, inputs, *args, **kwargs):
        kwargs["gazing_ratio"] = None
        kwargs["task_loss_requirement"] = None
        kwargs["max_gaze_tokens_each_frame"] = exact_budget
        kwargs["allowed_token_ids"] = allowed_token_ids
        kwargs["allow_eos"] = False
        if static_fine_cells is not None:
            kwargs["generate_only"] = True
        video = inputs.get("video")
        device = video.device if isinstance(video, torch.Tensor) else None
        _synchronize(device)
        started = time.perf_counter()
        output = original_forward(inputs, *args, **kwargs)
        _synchronize(device)
        stats.update(output, elapsed_seconds=time.perf_counter() - started)
        return output

    model.gazing_model.generate = MethodType(generated_or_static, model.gazing_model)
    model.forward = MethodType(fixed_forward, model)
    return stats
