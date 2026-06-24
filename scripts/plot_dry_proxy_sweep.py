# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot dry proxy sweep diagnostics and completed HLVid benchmark frontiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_DRY_SUMMARY = "outputs/hlvid/proxy_dry_sweeps/full_128_log_fast/sweep_summary.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/dry_proxy_sweep"
DEFAULT_BENCHMARK_ROOTS = [
    "outputs/hlvid/proxy_sampling_v1_mtv1",
    "outputs/hlvid/proxy_sampling_v2_cost_sweep_mtv1",
    "outputs/hlvid/proxy_sampling_v3_semantic_mtv1",
]

METHOD_LABELS = {
    ("mean_novelty", "pixel_mse", None): "Pixel novelty",
    ("mean_novelty", "edge_l1", None): "Edge novelty",
    ("ssim_distance", None, None): "SSIM distance",
    ("semantic_mean_novelty", None, "siglip"): "SigLIP semantic novelty",
    ("semantic_mean_novelty", None, "dinov2"): "DINOv2 semantic novelty",
    ("semantic_diversity", None, "dinov2"): "DINOv2 semantic diversity",
    ("siglip_qgain", None, None): "SigLIP qgain",
    ("question_relevance", None, None): "Question relevance",
    ("siglip_cosine", None, None): "SigLIP cosine",
    ("dinov2_cosine", None, None): "DINOv2 cosine",
}

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
    parser.add_argument("--benchmark-roots", nargs="+", default=DEFAULT_BENCHMARK_ROOTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_rows = read_jsonl(Path(args.dry_summary))
    benchmark_rows = read_benchmarks([Path(root) for root in args.benchmark_roots])

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_lambda_vs_frames(dry_rows, output_dir / "lambda_vs_frames.png")
    plot_delta_distribution(dry_rows, output_dir / "delta_distribution_representative.png")
    plot_budget_coverage(dry_rows, output_dir / "budget_coverage_scatter.png")
    plot_benchmark_accuracy_vs_frames(benchmark_rows, output_dir / "benchmark_accuracy_vs_frames.png")
    plot_frontier(
        benchmark_rows,
        x_key="mean_visual_tokens_projected",
        x_label="Mean visual tokens",
        output_path=output_dir / "frontier_accuracy_vs_visual_tokens.png",
    )
    plot_frontier(
        benchmark_rows,
        x_key="mean_visited_frames",
        x_label="Mean visited frames",
        output_path=output_dir / "frontier_accuracy_vs_visited_frames.png",
    )
    plot_frontier(
        benchmark_rows,
        x_key="mean_end_to_end_seconds",
        x_label="Mean end-to-end seconds per example",
        output_path=output_dir / "frontier_accuracy_vs_time.png",
    )
    write_plot_data(dry_rows, benchmark_rows, output_dir)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_benchmarks(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        for summary_path in sorted(root.glob("*/summary.json")):
            summary = json.loads(summary_path.read_text())
            if summary.get("num_examples") != 268:
                continue
            if summary.get("accuracy") is None:
                continue
            row = {
                "run_name": summary_path.parent.name,
                "root": str(root),
                "accuracy": float(summary["accuracy"]),
                "mean_visited_frames": summary.get("mean_visited_frames"),
                "mean_visual_tokens_projected": summary.get("mean_visual_tokens_projected"),
                "mean_end_to_end_seconds": summary.get("mean_end_to_end_seconds"),
                "proxy_metric": summary.get("proxy_metric"),
                "proxy_base_metric": summary.get("proxy_base_metric"),
                "semantic_model": summary.get("semantic_model"),
                "lambda_cost": summary.get("lambda_cost"),
                "selection": summary.get("selection"),
                "sampling": summary.get("sampling"),
            }
            rows.append(row)

    deduped = {}
    for row in rows:
        deduped[row["run_name"]] = row
    return list(deduped.values())


def plot_lambda_vs_frames(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for identity in DRY_LINE_METHODS:
        values = sorted([row for row in rows if dry_identity(row) == identity], key=lambda row: row["lambda_cost"])
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
    save_figure(fig, output_path)


def plot_delta_distribution(rows: list[dict], output_path: Path) -> None:
    by_name = {row["run_name"]: row for row in rows}
    delta_values = ["1", "2", "4", "8", "16", "32", "64", "128", "192", "256", "320"]
    labels = []
    stacks = {delta: [] for delta in delta_values}
    for run_name in DELTA_DISTRIBUTION_RUNS:
        row = by_name[run_name]
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
    save_figure(fig, output_path)


def plot_budget_coverage(rows: list[dict], output_path: Path) -> None:
    identities = sorted({dry_identity(row) for row in rows}, key=lambda item: METHOD_LABELS.get(item, str(item)))
    y_positions = {identity: index for index, identity in enumerate(identities)}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(
        [row["mean_visited_frames"] for row in rows],
        [y_positions[dry_identity(row)] for row in rows],
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
    save_figure(fig, output_path)


def plot_benchmark_accuracy_vs_frames(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot_benchmark_scatter(ax, rows, "mean_visited_frames")
    ax.set_xlabel("Mean visited frames")
    ax.set_ylabel("HLVid accuracy")
    ax.set_title("Completed NVILA-HLVid Benchmark Points")
    annotate_top_points(ax, rows, "mean_visited_frames")
    save_figure(fig, output_path)


def plot_frontier(rows: list[dict], x_key: str, x_label: str, output_path: Path) -> None:
    rows = [row for row in rows if isinstance(row.get(x_key), (int, float))]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot_benchmark_scatter(ax, rows, x_key)
    frontier = pareto_frontier(rows, x_key)
    if frontier:
        ax.plot(
            [row[x_key] for row in frontier],
            [row["accuracy"] for row in frontier],
            color="black",
            linewidth=2,
            marker="o",
            label="Pareto frontier",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("HLVid accuracy")
    ax.set_title(f"Evaluated Policy Frontier: Accuracy vs {x_label}")
    ax.legend(frameon=True, fontsize=8)
    annotate_frontier(ax, frontier, x_key)
    save_figure(fig, output_path)


def plot_benchmark_scatter(ax, rows: list[dict], x_key: str) -> None:
    groups = {}
    for row in rows:
        groups.setdefault(benchmark_family(row), []).append(row)
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    for index, (family, values) in enumerate(sorted(groups.items())):
        values = [row for row in values if isinstance(row.get(x_key), (int, float))]
        if not values:
            continue
        ax.scatter(
            [row[x_key] for row in values],
            [row["accuracy"] for row in values],
            label=family,
            marker=markers[index % len(markers)],
            s=58,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )


def pareto_frontier(rows: list[dict], x_key: str) -> list[dict]:
    best_accuracy = -1.0
    frontier = []
    for row in sorted(rows, key=lambda item: (item[x_key], -item["accuracy"])):
        if row["accuracy"] > best_accuracy + 1e-12:
            frontier.append(row)
            best_accuracy = row["accuracy"]
    return frontier


def annotate_top_points(ax, rows: list[dict], x_key: str) -> None:
    for row in sorted(rows, key=lambda item: item["accuracy"], reverse=True)[:6]:
        if isinstance(row.get(x_key), (int, float)):
            ax.annotate(short_benchmark_label(row), (row[x_key], row["accuracy"]), fontsize=7, xytext=(4, 4), textcoords="offset points")


def annotate_frontier(ax, rows: list[dict], x_key: str) -> None:
    for row in rows:
        ax.annotate(short_benchmark_label(row), (row[x_key], row["accuracy"]), fontsize=7, xytext=(4, 4), textcoords="offset points")


def dry_identity(row: dict) -> tuple[str, str | None, str | None]:
    metric = row["proxy_metric"]
    semantic_model = row.get("semantic_model")
    if metric in ("siglip_qgain", "question_relevance", "siglip_cosine", "dinov2_cosine"):
        semantic_model = None
    return metric, row.get("proxy_base_metric"), semantic_model


def benchmark_family(row: dict) -> str:
    if row.get("selection") == "uniform":
        return "Uniform"
    identity = (row.get("proxy_metric"), row.get("proxy_base_metric"), row.get("semantic_model"))
    if row.get("proxy_metric") in ("siglip_qgain", "question_relevance", "siglip_cosine", "dinov2_cosine"):
        identity = (row.get("proxy_metric"), row.get("proxy_base_metric"), None)
    return METHOD_LABELS.get(identity, row.get("proxy_metric") or "Other")


def short_run_label(row: dict) -> str:
    label = METHOD_LABELS.get(dry_identity(row), row["proxy_metric"])
    return f"{label}\nlambda={row['lambda_cost']}"


def short_benchmark_label(row: dict) -> str:
    name = row["run_name"]
    replacements = {
        "semantic_mean_novelty": "semnovel",
        "question_relevance": "qrel",
        "siglip_": "siglip ",
        "dinov2_": "dino ",
        "_128_": " ",
        "_log": "",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.replace("_", " ")


def write_plot_data(dry_rows: list[dict], benchmark_rows: list[dict], output_dir: Path) -> None:
    (output_dir / "dry_sweep_plot_rows.json").write_text(json.dumps(dry_rows, indent=2) + "\n")
    (output_dir / "benchmark_plot_rows.json").write_text(json.dumps(benchmark_rows, indent=2) + "\n")


def save_figure(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
