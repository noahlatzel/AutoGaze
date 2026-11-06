"""Deterministic uniform-random exact-K fine-cell control for HLVid/NVILA (gate F1).

Random16 replaces learned AutoGaze generation at the same tile-local boundary as
Center16: before NVILA resolution recovery, with probability rescoring skipped.
Each (video, tile, frame) draws K distinct cells uniformly from the 14x14 fine
grid with its own RNG derived from a fixed seed, so selections are reproducible,
independent of call batching, and identical whenever a video recurs across
questions. The frozen fixed-budget adapter in ``hlvid.py`` is reused unchanged.
"""

from __future__ import annotations

import hashlib
from types import MethodType
from typing import Any

import numpy as np
import torch

from autogaze.human_gaze.hlvid import FixedBudgetCallStats, install_fixed_budget_forward

RANDOM16_RULE = (
    "per (video_key, tile_index, frame_index): numpy default_rng(SeedSequence(sha256("
    "'f1-random-cells|{seed}|{video_key}|{tile_index}|{frame_index}') as 4x uint64)).choice("
    "num_cells, K, replace=False), sorted ascending; tile_index enumerates AutoGaze tile "
    "batch items in processor order within one question; frame_index is the decoder frame"
)


def random_fine_cells(
    seed: int,
    video_key: str,
    tile_index: int,
    frame_index: int,
    *,
    budget: int = 16,
    num_cells: int = 196,
) -> tuple[int, ...]:
    """Return K distinct sorted cells in [0, num_cells) for one tile-frame."""

    if not 0 < budget <= num_cells:
        raise ValueError("budget must lie in [1, num_cells]")
    key = f"f1-random-cells|{int(seed)}|{video_key}|{int(tile_index)}|{int(frame_index)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    entropy = [int.from_bytes(digest[i : i + 8], "little") for i in range(0, 32, 8)]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    cells = rng.choice(num_cells, size=budget, replace=False)
    return tuple(sorted(int(value) for value in cells))


class RandomCellSelector:
    """Stateful replacement for ``gazing_model.generate`` within one question."""

    def __init__(
        self,
        *,
        seed: int,
        exact_budget: int,
        fine_action_offset: int = 69,
        actions_per_frame: int = 265,
    ) -> None:
        self.seed = int(seed)
        self.exact_budget = int(exact_budget)
        self.fine_action_offset = int(fine_action_offset)
        self.actions_per_frame = int(actions_per_frame)
        self.num_cells = self.actions_per_frame - self.fine_action_offset
        self.video_key: str | None = None
        self.next_tile_index = 0

    def start_example(self, video_key: str) -> None:
        self.video_key = str(video_key)
        self.next_tile_index = 0

    def finish_example(self) -> dict[str, Any]:
        if self.video_key is None:
            raise RuntimeError("No random-cell example is active")
        result = {"video_key": self.video_key, "tiles": self.next_tile_index}
        self.video_key = None
        self.next_tile_index = 0
        return result

    def generate(self, _gazing_model, video, *args, **kwargs) -> dict[str, Any]:
        if self.video_key is None:
            raise RuntimeError("Random-cell generation outside an active example")
        batch_size, input_frames = video.shape[:2]
        output_frames = input_frames // int(_gazing_model.frame_sampling_rate)
        rows = []
        for item in range(batch_size):
            tile_index = self.next_tile_index + item
            positions = []
            for frame_index in range(output_frames):
                cells = random_fine_cells(
                    self.seed,
                    self.video_key,
                    tile_index,
                    frame_index,
                    budget=self.exact_budget,
                    num_cells=self.num_cells,
                )
                offset = frame_index * self.actions_per_frame + self.fine_action_offset
                positions.extend(offset + cell for cell in cells)
            rows.append(positions)
        self.next_tile_index += batch_size
        positions = torch.tensor(rows, device=video.device, dtype=torch.long)
        return {
            "gazing_pos": positions,
            "num_gazing_each_frame": torch.full(
                (output_frames,), self.exact_budget, device=video.device, dtype=torch.long
            ),
            "if_padded_gazing": torch.zeros_like(positions, dtype=torch.bool),
            "task_loss_requirement": None,
        }


def install_random_fine_cells(
    model: torch.nn.Module,
    *,
    seed: int,
    exact_budget: int,
    fine_action_offset: int = 69,
    actions_per_frame: int = 265,
) -> tuple[FixedBudgetCallStats, RandomCellSelector]:
    """Install Random-K at the Center16 boundary while reusing the frozen adapter."""

    selector = RandomCellSelector(
        seed=seed,
        exact_budget=exact_budget,
        fine_action_offset=fine_action_offset,
        actions_per_frame=actions_per_frame,
    )

    def random_generate(_gazing_model, video, *args, **kwargs):
        return selector.generate(_gazing_model, video, *args, **kwargs)

    model.gazing_model.generate = MethodType(random_generate, model.gazing_model)
    # The adapter captures the replaced generator as its "learned" path, so its
    # exact-K/uniqueness/vocabulary validation and telemetry apply unchanged.
    stats = install_fixed_budget_forward(
        model,
        exact_budget=exact_budget,
        fine_action_offset=fine_action_offset,
        actions_per_frame=actions_per_frame,
        static_fine_cells=None,
    )
    adapter_forward = model.forward

    def generate_only_forward(_model, inputs, *args, **kwargs):
        # Same as Center16: no learned probability-rescoring pass.
        kwargs["generate_only"] = True
        return adapter_forward(inputs, *args, **kwargs)

    model.forward = MethodType(generate_only_forward, model)
    return stats, selector


__all__ = [
    "RANDOM16_RULE",
    "RandomCellSelector",
    "install_random_fine_cells",
    "random_fine_cells",
]
