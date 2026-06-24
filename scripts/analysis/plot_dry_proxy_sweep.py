# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot dry proxy-sweep diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from proxy_plot_common import METHOD_LABELS, proxy_identity, proxy_label, read_jsonl


DEFAULT_DRY_SUMMARY = "outputs/hlvid/proxy_dry_sweeps/full_128_log_fast/sweep_summary.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/dry_proxy_sweep"

DRY_LINE_METHODS = [
    ("mean_novelty", "pixel_mse", None),
    ("ssim_distance", None, None),
    ("semantic_mean_novelty", None, "siglip"),
    ("semantic_mean_novelty", None, "dinov2"),
    ("semantic_diversity", None, "dinov2"),
    ("siglip_qgain", None, None),
    ("question_relevance", None, None),
]

DELTA_DISTRIBUTION_RUNS = [
    "dinov2_semantic_mean_novelty_l0p002_log",
    "dinov2_semantic_mean_novelty_l0p05_log",
    "siglip_qgain_l0p002_log",
    "siglip_qgain_l0p01_log",
    "question_relevance_l0p01_log",
    "mean_novelty_pixel_mse_l0p01_log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-summary", default=DEFAULT_DRY_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--delta-distribution-runs",
        nargs="+",
        default=DELTA_DISTRIBUTION_RUNS,
        help="Representative dry run names used for the stacked delta-distribution plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.dry_summary))

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_lambda_vs_frames(rows, output_dir / "lambda_vs_frames.png")
    plot_delta_distribution(rows, args.delta_distribution_runs, output_dir / "delta_distribution_representative.png")
    plot_delta_heatmap(rows, output_dir / "delta_heatmap.png")
    plot_budget_coverage(rows, output_dir / "budget_coverage_scatter.png")
    write_plot_data(rows, output_dir)


def plot_lambda_vs_frames(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for identity in DRY_LINE_METHODS:
        values = sorted([row for row in rows if proxy_identity(row) == identity], key=lambda row: row["lambda_cost"])
        if not values:
            continue
        ax.plot(
            [row["lambda_cost"] for row in values],
            [row["mean_visited_frames"] for row in values],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS.get(identity, str(identity)),
        )
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlabel("Temporal cost weight lambda")
    ax.set_ylabel("Mean visited frames")
    ax.set_title("Dry Sweep: Temporal Cost Controls Frame Budget")
    ax.set_ylim(48, 116)
    ax.legend(frameon=True, fontsize=8)
    save_png_figure(fig, output_path)


def plot_delta_distribution(rows: list[dict], run_names: list[str], output_path: Path) -> None:
    by_name = {row["run_name"]: row for row in rows}
    selected_rows = [by_name[name] for name in run_names if name in by_name]
    if not selected_rows:
        return

    delta_values = ["1", "2", "4", "8", "16", "32", "64", "128", "192", "256", "320"]
    labels = []
    stacks = {delta: [] for delta in delta_values}
    for row in selected_rows:
        labels.append(short_run_label(row))
        counts = row["selected_delta_counts"]
        total = max(sum(counts.values()), 1)
        for delta in delta_values:
            stacks[delta].append(counts.get(delta, 0) / total)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bottoms = [0.0] * len(labels)
    colors = plt.cm.viridis([index / (len(delta_values) - 1) for index in range(len(delta_values))])
    for delta, color in zip(delta_values, colors):
        values = stacks[delta]
        ax.bar(labels, values, bottom=bottoms, label=delta, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_ylabel("Fraction of selected actions")
    ax.set_title("Dry Sweep: Representative Selected-Delta Distributions")
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(title="Delta", ncols=2, fontsize=8, frameon=True)
    save_png_figure(fig, output_path)


def plot_budget_coverage(rows: list[dict], output_path: Path) -> None:
    identities = sorted({proxy_identity(row) for row in rows}, key=lambda item: METHOD_LABELS.get(item, str(item)))
    y_positions = {identity: index for index, identity in enumerate(identities)}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(
        [row["mean_visited_frames"] for row in rows],
        [y_positions[proxy_identity(row)] for row in rows],
        c=[row["lambda_cost"] for row in rows],
        cmap="plasma",
        s=52,
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([METHOD_LABELS.get(identity, str(identity)) for identity in identities], fontsize=8)
    ax.set_xlabel("Mean visited frames")
    ax.set_title("Dry Sweep: Budget Coverage Across Proxy Policies")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("lambda")
    save_png_figure(fig, output_path)


def plot_delta_heatmap(rows: list[dict], output_path: Path) -> None:
    rows = [row for row in rows if row.get("selected_delta_counts")]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: (proxy_label(row), row["lambda_cost"], row["run_name"]))
    delta_values = sorted(
        {int(delta) for row in rows for delta in row["selected_delta_counts"]},
        key=lambda value: value,
    )
    heatmap = []
    labels = []
    for row in rows:
        counts = row["selected_delta_counts"]
        total = max(sum(counts.values()), 1)
        heatmap.append([counts.get(str(delta), 0) / total for delta in delta_values])
        labels.append(f"{proxy_label(row)} | lambda={row['lambda_cost']}")

    fig_height = max(6.0, 0.18 * len(rows))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    image = ax.imshow(np.array(heatmap), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(np.arange(len(delta_values)))
    ax.set_xticklabels([str(delta) for delta in delta_values])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Selected delta")
    ax.set_title("Dry Sweep: Selected-Delta Distribution Across All Runs")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Fraction of selected actions")
    save_png_figure(fig, output_path)


def short_run_label(row: dict) -> str:
    return f"{proxy_label(row)}\nlambda={row['lambda_cost']}"


def write_plot_data(rows: list[dict], output_dir: Path) -> None:
    (output_dir / "dry_sweep_plot_rows.json").write_text(json.dumps(rows, indent=2) + "\n")


def save_png_figure(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
