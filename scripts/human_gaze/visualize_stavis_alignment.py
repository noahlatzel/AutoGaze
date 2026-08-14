# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one aligned STAViS frame, heatmap, overlay, and Oracle-16 cells."""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cell-mass", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--source", default="AVAD")
    parser.add_argument("--frame-position", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = AVGazeStavisDataset(
        root=args.root,
        manifest_path=args.manifest,
        split=args.split,
        load_rgb=True,
        load_heatmap=True,
        cell_mass_path=args.cell_mass,
    )
    index = next(
        index for index, record in enumerate(dataset.records) if record["source"] == args.source
    )
    item = dataset[index]
    position = args.frame_position
    rgb = item["video"][position].permute(1, 2, 0).numpy()
    heatmap = item["heatmap"][position].numpy()
    selected = torch.argsort(item["cell_mass"][position], descending=True)[:16].tolist()

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Aligned RGB 224×224")
    axes[1].imshow(heatmap, cmap="magma")
    axes[1].set_title("Normalized human heatmap")
    axes[2].imshow(rgb)
    axes[2].imshow(heatmap, cmap="magma", alpha=0.55)
    axes[2].set_title("Alignment overlay")
    axes[3].imshow(rgb)
    axes[3].imshow(heatmap, cmap="magma", alpha=0.4)
    cell_size = 224 // 14
    for cell in selected:
        row, column = divmod(cell, 14)
        axes[3].add_patch(
            patches.Rectangle(
                (column * cell_size, row * cell_size),
                cell_size,
                cell_size,
                linewidth=1.2,
                edgecolor="#35e7ff",
                facecolor="none",
            )
        )
    for coordinate in range(0, 225, cell_size):
        axes[3].axhline(coordinate - 0.5, color="white", linewidth=0.2, alpha=0.35)
        axes[3].axvline(coordinate - 0.5, color="white", linewidth=0.2, alpha=0.35)
    coverage = float(item["cell_mass"][position, selected].sum())
    axes[3].set_title(f"Oracle-16 cells ({coverage:.1%} mass)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"{item['source']} / {item['video_id']} / source frame "
        f"{int(item['frame_numbers'][position])}",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
