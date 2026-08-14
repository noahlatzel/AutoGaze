# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose whether a learned exact-16 policy collapsed to a static center prior."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset
from autogaze.human_gaze.coverage import (
    CoverageAccumulator,
    center_order,
    global_positions_to_fine_cells,
    selected_coverage,
)
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor


def generate_fine(model, video, cfg, allowed):
    gaze = model(
        {"video": video},
        max_gaze_tokens_each_frame=int(cfg["exact_budget"]),
        allowed_token_ids=allowed,
        generate_only=True,
    )
    return global_positions_to_fine_cells(
        gaze["gazing_pos"],
        num_frames=int(cfg["clip_len"]),
        exact_budget=int(cfg["exact_budget"]),
        actions_per_frame=int(cfg["actions_per_frame"]),
        fine_action_offset=int(cfg["fine_action_offset"]),
    ).cpu()


def within_source_different_video_indices(records):
    """Pair each clip with a deterministic clip from another same-source video."""
    source_videos = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        source_videos[record["source"]][record["video_id"]].append(index)
    shuffled = [None] * len(records)
    for videos in source_videos.values():
        video_ids = sorted(videos)
        if len(video_ids) == 1:
            indices = videos[video_ids[0]]
            offset = max(1, len(indices) // 2)
            for position, index in enumerate(indices):
                shuffled[index] = indices[(position + offset) % len(indices)]
            continue
        for video_position, video_id in enumerate(video_ids):
            target = videos[video_ids[(video_position + 1) % len(video_ids)]]
            for clip_position, index in enumerate(videos[video_id]):
                shuffled[index] = target[clip_position % len(target)]
    if any(index is None for index in shuffled):
        raise AssertionError("Failed to assign a shuffled clip")
    return shuffled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    batch_size = args.batch_size or int(cfg["batch_size"])

    checkpoint = str(args.checkpoint)
    processor = AutoGazeImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoGaze.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    dataset = AVGazeStavisDataset(
        root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        split=args.split,
        load_rgb=True,
        load_heatmap=False,
        cell_mass_path=cfg["cell_mass_path"],
        image_processor=processor,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=int(cfg["num_workers"]),
        shuffle=False,
        pin_memory=True,
    )
    shuffled_indices = within_source_different_video_indices(dataset.records)
    shuffled_loader = DataLoader(
        Subset(dataset, shuffled_indices),
        batch_size=batch_size,
        num_workers=int(cfg["num_workers"]),
        shuffle=False,
        pin_memory=True,
    )

    grid_size = int(round(math.sqrt(int(cfg["actions_per_frame"]) - int(cfg["fine_action_offset"]))))
    num_cells = grid_size * grid_size
    budget = int(cfg["exact_budget"])
    center = center_order(grid_size)[:budget]
    center_set = set(center.tolist())
    allowed = list(range(int(cfg["fine_action_offset"]), int(cfg["actions_per_frame"])))
    accumulator = CoverageAccumulator()
    selection_counts = torch.zeros(num_cells, dtype=torch.long)
    center_overlap = []
    shuffled_overlap = []
    frame_masks = set()
    processed = 0

    with torch.inference_mode():
        for batch, shuffled_batch in zip(loader, shuffled_loader):
            if args.max_clips is not None and processed >= args.max_clips:
                break
            video = batch["video"].cuda(non_blocking=True)
            actual = generate_fine(model, video, cfg, allowed)
            shuffled = generate_fine(
                model, shuffled_batch["video"].cuda(non_blocking=True), cfg, allowed
            )
            for index in range(actual.shape[0]):
                if args.max_clips is not None and processed >= args.max_clips:
                    break
                cell_mass = batch["cell_mass"][index]
                source = batch["source"][index]
                video_id = batch["video_id"][index]
                accumulator.add(
                    "actual", budget, source, video_id, selected_coverage(cell_mass, actual[index])
                )
                accumulator.add(
                    "shuffled", budget, source, video_id, selected_coverage(cell_mass, shuffled[index])
                )
                accumulator.add(
                    "center", budget, source, video_id, selected_coverage(cell_mass, center)
                )
                for frame_actual, frame_shuffled in zip(actual[index], shuffled[index]):
                    selection_counts.scatter_add_(
                        0, frame_actual, torch.ones_like(frame_actual, dtype=torch.long)
                    )
                    actual_set = set(frame_actual.tolist())
                    shuffled_set = set(frame_shuffled.tolist())
                    center_overlap.append(len(actual_set & center_set) / budget)
                    shuffled_overlap.append(len(actual_set & shuffled_set) / budget)
                    frame_masks.add(tuple(sorted(actual_set)))
                processed += 1

    methods = ("actual", "shuffled", "center")
    coverage = accumulator.finalize(methods, [budget])
    total_frames = processed * int(cfg["clip_len"])
    probabilities = selection_counts.to(torch.float64) / selection_counts.sum()
    nonzero = probabilities[probabilities > 0]
    normalized_entropy = float(-(nonzero * nonzero.log()).sum() / math.log(num_cells))
    macro = {method: coverage[method][str(budget)]["macro_source_mean"] for method in methods}
    report = {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "split": args.split,
        "processed_clips": processed,
        "shuffle_strategy": "within_source_different_video",
        "coverage": coverage,
        "diagnostics": {
            "actual_minus_shuffled_macro": macro["actual"] - macro["shuffled"],
            "actual_minus_center_macro": macro["actual"] - macro["center"],
            "mean_center16_overlap_fraction": float(np.mean(center_overlap)),
            "mean_actual_shuffled_overlap_fraction": float(np.mean(shuffled_overlap)),
            "normalized_selection_entropy": normalized_entropy,
            "unique_frame_selection_sets": len(frame_masks),
            "unique_frame_selection_fraction": len(frame_masks) / total_frames,
            "selection_frequency_per_frame": (
                selection_counts.to(torch.float64) / total_frames
            ).tolist(),
            "center16_cells": center.tolist(),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    frequency = selection_counts.reshape(grid_size, grid_size).to(torch.float64) / total_frames
    figure, axis = plt.subplots(figsize=(6.5, 5.7), constrained_layout=True)
    image = axis.imshow(frequency.numpy(), cmap="magma", vmin=0, vmax=1)
    for cell in center.tolist():
        row, column = divmod(cell, grid_size)
        axis.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.2))
    axis.set_xlabel("Fine-grid column")
    axis.set_ylabel("Fine-grid row")
    axis.set_title("Learned exact-16 selection frequency (Center-16 outlined)")
    figure.colorbar(image, ax=axis, label="Fraction of validation frames selecting cell")
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_plot, dpi=180)
    plt.close(figure)

    print(json.dumps({"macro": macro, **report["diagnostics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
