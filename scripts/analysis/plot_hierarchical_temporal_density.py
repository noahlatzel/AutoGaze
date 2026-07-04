# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot temporal density for hierarchical sampling dry-run outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Hierarchical dry-run JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--title-prefix", default="Hierarchical sampling")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    rows = load_rows(Path(args.results))
    if not rows:
        raise SystemExit(f"No rows found in {args.results}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = []
    labels = []
    for row in rows:
        indices = [idx for leaf in row["leaf_intervals"] for idx in leaf["sampled_indices"]]
        edges = np.linspace(0, row["frame_count"], args.bins + 1)
        counts, _ = np.histogram(indices, bins=edges)
        matrix.append(counts)
        labels.append(row["video_path"].replace(".mp4", "").replace("clip_", ""))
    matrix = np.asarray(matrix)

    fig_h = max(3.5, 0.55 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(f"{args.title_prefix}: selected frames per normalized time bin")
    ax.set_xlabel("Normalized video time bin")
    ax.set_ylabel("Video")
    ax.set_xticks(np.linspace(0, args.bins - 1, 9))
    ax.set_xticklabels([f"{value:.2f}" for value in np.linspace(0, 1, 9)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value:
                color = "white" if value > matrix.max() * 0.55 else "black"
                ax.text(j, i, str(value), ha="center", va="center", fontsize=6, color=color)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Selected frames")
    fig.tight_layout()
    heatmap_path = output_dir / "temporal_density_heatmap.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    max_depth = max(max(row["leaf_depths"]) for row in rows)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    cmap = plt.get_cmap("magma", max_depth + 1)
    for y, row in enumerate(rows):
        for leaf in row["leaf_intervals"]:
            start = leaf["start"] / row["frame_count"]
            width = (leaf["end"] - leaf["start"] + 1) / row["frame_count"]
            depth = int(leaf["depth"])
            ax.barh(y, width, left=start, height=0.66, color=cmap(depth), edgecolor="white", linewidth=0.4)
            if width > 0.035:
                ax.text(start + width / 2, y, str(depth), ha="center", va="center", fontsize=7, color="white")
    ax.set_xlim(0, 1)
    ax.set_title(f"{args.title_prefix}: leaf intervals over normalized video time")
    ax.set_xlabel("Normalized video time")
    ax.set_ylabel("Video")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_depth))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, ticks=range(max_depth + 1))
    cbar.set_label("Leaf depth")
    fig.tight_layout()
    timeline_path = output_dir / "leaf_depth_timeline.png"
    fig.savefig(timeline_path, dpi=180)
    plt.close(fig)

    summary = {
        "num_videos": len(rows),
        "bins": args.bins,
        "heatmap": str(heatmap_path),
        "timeline": str(timeline_path),
        "per_video": [],
    }
    for row, counts in zip(rows, matrix):
        lengths = [leaf["end"] - leaf["start"] + 1 for leaf in row["leaf_intervals"]]
        summary["per_video"].append(
            {
                "video_path": row["video_path"],
                "category": row["category"],
                "num_selected_frames": row["num_selected_frames"],
                "num_leaf_tiles": row["num_leaf_tiles"],
                "nonzero_bins": int((counts > 0).sum()),
                "max_bin_count": int(counts.max()),
                "bin_counts": counts.tolist(),
                "depth_counts": dict(sorted(Counter(row["leaf_depths"]).items())),
                "interval_length_min": int(min(lengths)),
                "interval_length_max": int(max(lengths)),
            }
        )
    summary_path = output_dir / "temporal_density_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
