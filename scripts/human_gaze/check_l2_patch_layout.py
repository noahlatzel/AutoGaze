#!/usr/bin/env python3
"""L2 CPU check: are the cells chosen by a fixed-budget policy the patches the tower encodes?

No GPU and no answerer weights. For each registered processor layout and each
policy kind, the real NVILA processor (with the real AutoGaze model and the
fixed-budget adapter) selects 16 fine cells per 224 px sub-tile on a synthetic
16-frame clip. The vision tower's real embedding module then gathers from
pixels that encode their own frame and position. The intended patches follow
from the recorded decoder actions; the gathered patches are read off the
pixels. The report gives, per layout and policy, the share of intended patches
that the tower gathers.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml
from PIL import Image

from autogaze.human_gaze.hlvid import install_fixed_budget_forward
from autogaze.human_gaze.random_cells import install_random_fine_cells
from scripts.human_gaze.evaluate_hlvid_nvila_fixed_budget import CENTER16_FINE_CELLS, sha256

FRAMES = 16
FINE_OFFSET = 69
ACTIONS_PER_FRAME = 265
SUBTILE_SIDE = 14


def synthetic_clip(seed: int, size: int = 448) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size // 16, size // 16, 3), dtype=np.uint8)
    base = np.asarray(Image.fromarray(base).resize((size, size), Image.BICUBIC)).copy()
    frames = []
    for index in range(FRAMES):
        frame = base.copy()
        x = 20 + 22 * index
        frame[100:180, x:x + 90] = rng.integers(0, 255, size=(80, 90, 3), dtype=np.uint8)
        frames.append(Image.fromarray(frame))
    return frames


def coded_pixels(items: int, size: int) -> torch.Tensor:
    """(items, FRAMES, 3, size, size): channel 0 frame, channel 1 row, channel 2 column."""
    ramp = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    pixels = torch.empty(items, FRAMES, 3, size, size)
    pixels[:, :, 0] = ((torch.arange(FRAMES, dtype=torch.float32) + 0.5) / FRAMES)[None, :, None, None]
    pixels[:, :, 1] = ramp[None, None, :, None]
    pixels[:, :, 2] = ramp[None, None, None, :]
    return pixels


def decode_patches(patches: torch.Tensor, tower_scales: list[int], patch: int) -> np.ndarray:
    """(n, 3*patch*patch) coded patches -> (n, 4) [frame, scale index, row, column]."""
    p = patches.reshape(-1, 3, patch, patch).double()
    frame = torch.round(p[:, 0].mean(dim=(1, 2)) * FRAMES - 0.5).long()
    step = (p[:, 1, 1:, :] - p[:, 1, :-1, :]).median(dim=1).values.median(dim=1).values
    scales = torch.tensor(tower_scales, dtype=torch.double)
    scale_index = ((1.0 / step)[:, None] - scales[None, :]).abs().argmin(dim=1)
    side = (scales[scale_index] / patch).long()
    row = torch.minimum((p[:, 1].mean(dim=(1, 2)) * side).floor().long(), side - 1)
    column = torch.minimum((p[:, 2].mean(dim=(1, 2)) * side).floor().long(), side - 1)
    return torch.stack([frame, scale_index, row, column], dim=1).numpy()


def intended_patches(calls: list[dict], fine_scale_index: int) -> set[tuple[int, int, int, int, int]]:
    """Decoder actions per sub-tile and frame -> (tile, frame, scale, row, column) in the tile's fine grid.

    AutoGaze splits each 448 px tile into 2x2 sub-tiles of 224 px in row-major
    order, so decoder item ``index`` is sub-tile (index // 2 % 2, index % 2) of
    tile ``index // 4``.
    """
    patches = set()
    item = 0
    for call in calls:
        for frames in call["decoder_action_ids"]:
            tile, sub = divmod(item, 4)
            sub_row, sub_column = divmod(sub, 2)
            for frame, actions in enumerate(frames):
                for action in actions:
                    row, column = divmod(int(action) - FINE_OFFSET, SUBTILE_SIDE)
                    patches.add((tile, frame, fine_scale_index,
                                 sub_row * SUBTILE_SIDE + row, sub_column * SUBTILE_SIDE + column))
            item += 1
    return patches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write the JSON report here (append-only).")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    model_path = Path(config["models"]["nvila"]["path"])
    autogaze_path = Path(config["models"]["pretrained_autogaze"]["path"])
    for name, expected in config["models"]["nvila"]["artifact_hashes"].items():
        if sha256(model_path / name) != expected:
            raise ValueError(f"pinned answerer file changed: {name}")

    from autogaze.vision_encoders.siglip.configuration_siglip import SiglipVisionConfig
    from autogaze.vision_encoders.siglip.modeling_siglip import SiglipVisionEmbeddings
    from transformers import AutoProcessor
    from transformers.modeling_utils import PreTrainedModel

    # The released processor moves AutoGaze to "cuda" unconditionally; this check has no GPU.
    original_to = PreTrainedModel.to

    def to_cpu(self, *to_args, **to_kwargs):
        to_args = tuple("cpu" if isinstance(value, str) and value.startswith("cuda") else value for value in to_args)
        return original_to(self, *to_args, **to_kwargs)

    PreTrainedModel.to = to_cpu
    seed = int(config["analysis"]["bootstrap_seed"])
    torch.manual_seed(seed)
    vision = json.loads((model_path / "config.json").read_text())["vision_config"]
    embeddings = SiglipVisionEmbeddings(SiglipVisionConfig(**vision)).eval()
    tower_scales = [int(scale) for scale in embeddings.scales]
    patch = int(embeddings.patch_size)
    tower_counts = [int(value) for value in embeddings.num_patch_each_scale]
    fine_scale_index = len(tower_scales) - 1

    layouts = {
        "consistent": dict(config["protocol"]["processor_layout"]),
        "released_default": {},
    }
    frames = synthetic_clip(seed)
    report = {
        "gate": "L2", "step": "cpu layout check of the fixed-budget path",
        "config_sha256": sha256(args.config),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
        "tower": {"scales": tower_scales, "patch_size": patch, "positions_per_scale": tower_counts},
        "clip": {"frames": FRAMES, "size": 448, "spatial_tiles": 1, "sub_tiles": 4}, "layouts": {},
    }
    for layout_name, overrides in layouts.items():
        entry = {}
        for policy in ("center16", "random16", "pretrained"):
            processor = AutoProcessor.from_pretrained(
                str(model_path), autogaze_model_id=str(autogaze_path),
                num_video_frames=FRAMES, num_video_frames_thumbnail=64, max_tiles_video=1,
                gazing_ratio_tile=[0.2] + [0.06] * 15, gazing_ratio_thumbnail=1,
                task_loss_requirement_tile=0.6, task_loss_requirement_thumbnail=None,
                max_batch_size_autogaze=16, trust_remote_code=True, use_fast=False, **overrides)
            selector = None
            if policy == "random16":
                stats, selector = install_random_fine_cells(
                    processor._autogaze_model, seed=seed, exact_budget=16,
                    fine_action_offset=FINE_OFFSET, actions_per_frame=ACTIONS_PER_FRAME)
            else:
                stats = install_fixed_budget_forward(
                    processor._autogaze_model, exact_budget=16, fine_action_offset=FINE_OFFSET,
                    actions_per_frame=ACTIONS_PER_FRAME,
                    static_fine_cells=CENTER16_FINE_CELLS if policy == "center16" else None)
            stats.start_example()
            if selector is not None:
                selector.start_example("layout-check")
            video_token = processor.tokenizer.video_token
            inputs = processor(text=f"{video_token}\n\nQuestion: layout check", videos=[frames], return_tensors="pt")
            acquisition = stats.finish_example()
            if selector is not None:
                selector.finish_example()
            info = inputs["gazing_info"]
            positions = info["gazing_pos_tiles"][0].long()
            padded = info["if_padded_gazing_tiles"][0].bool()
            size = int(inputs["pixel_values_videos_tiles"][0].shape[-1])
            with torch.inference_mode():
                gathered = embeddings.get_gazed_patches_and_pos_embeddings(
                    coded_pixels(positions.shape[0], size),
                    {"gazing_pos": positions, "if_padded_gazing": padded})["patches"]
            keep = ~padded
            decoded = decode_patches(gathered[keep], tower_scales, patch)
            item = np.repeat(np.arange(positions.shape[0]), keep.sum(dim=1).numpy())
            gathered_set = {(int(i), *map(int, row)) for i, row in zip(item, decoded)}
            intended = intended_patches(acquisition["calls"], fine_scale_index)
            processor_counts = [(int(scale) // int(processor.target_patch_size)) ** 2
                                for scale in sorted(processor.target_scales)]
            by_frame = {}
            for frame in range(FRAMES):
                wanted = {value for value in intended if value[1] == frame}
                by_frame[str(frame)] = len(wanted & gathered_set) / len(wanted)
            entry[policy] = {
                "processor_positions_per_scale": processor_counts,
                "layout_consistent": processor_counts == tower_counts,
                "intended_patches": len(intended),
                "gathered_patches": len(gathered_set),
                "intended_patches_gathered": len(intended & gathered_set) / len(intended),
                "gathered_patches_not_intended": len(gathered_set - intended) / len(gathered_set),
                "intended_patches_gathered_by_frame": by_frame,
                "patches_per_tile_frame": len(intended) / (positions.shape[0] * FRAMES),
            }
        report["layouts"][layout_name] = entry
    consistent = report["layouts"]["consistent"]
    report["conclusion"] = {
        "consistent_layout_delivers_selected_cells": all(
            value["layout_consistent"] and value["intended_patches_gathered"] == 1.0
            and value["gathered_patches_not_intended"] == 0.0 for value in consistent.values()),
        "released_default_delivers_selected_cells": all(
            value["intended_patches_gathered"] == 1.0 for value in report["layouts"]["released_default"].values()),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists() and args.output.read_text() != text:
            raise ValueError(f"append-only check output differs: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    if not report["conclusion"]["consistent_layout_delivers_selected_cells"]:
        raise SystemExit("registered consistent layout does not deliver the selected cells")


if __name__ == "__main__":
    main()
