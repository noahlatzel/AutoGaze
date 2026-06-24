# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot completed HLVid/NVILA proxy benchmark frontiers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from proxy_plot_common import METHOD_LABELS, proxy_identity


DEFAULT_OUTPUT_DIR = "outputs/hlvid_proxy_benchmarks"
DEFAULT_BENCHMARK_ROOTS = [
    "outputs/hlvid/proxy_sampling_v1_mtv1",
    "outputs/hlvid/proxy_sampling_v2_cost_sweep_mtv1",
    "outputs/hlvid/proxy_sampling_v3_semantic_mtv1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-roots", nargs="+", default=DEFAULT_BENCHMARK_ROOTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-num-examples", type=int, default=268)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_benchmarks([Path(root) for root in args.benchmark_roots], args.expected_num_examples)

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_frontier(
        rows,
        x_key="mean_visual_tokens_projected",
        x_label="Mean visual tokens",
        output_path=output_dir / "frontier_accuracy_vs_visual_tokens.png",
    )
    plot_frontier(
        rows,
        x_key="mean_visited_frames",
        x_label="Mean visited frames",
        output_path=output_dir / "frontier_accuracy_vs_visited_frames.png",
    )
    plot_frontier(
        rows,
        x_key="mean_end_to_end_seconds",
        x_label="Mean end-to-end seconds per example",
        output_path=output_dir / "frontier_accuracy_vs_time.png",
    )
    baseline = find_uniform_baseline(rows)
    if baseline is not None:
        plot_accuracy_delta_vs_uniform(rows, baseline, output_dir / "accuracy_delta_vs_uniform.png")
        plot_category_accuracy_delta_vs_uniform(rows, baseline, output_dir / "category_accuracy_delta_vs_uniform.png")
        plot_efficiency_vs_uniform(rows, baseline, output_dir / "efficiency_vs_uniform.png")
        plot_per_question_uniform_comparison(rows, baseline, output_dir / "per_question_uniform_comparison.png")
    plot_timing_breakdown(rows, baseline, output_dir / "timing_breakdown.png")
    write_plot_data(rows, output_dir)


def read_benchmarks(roots: list[Path], expected_num_examples: int | None) -> list[dict]:
    rows = []
    for root in roots:
        for summary_path in sorted(root.glob("*/summary.json")):
            summary = json.loads(summary_path.read_text())
            if expected_num_examples is not None and summary.get("num_examples") != expected_num_examples:
                continue
            if summary.get("accuracy") is None:
                continue
            rows.append(
                {
                    "run_name": summary_path.parent.name,
                    "root": str(root),
                    "accuracy": float(summary["accuracy"]),
                    "mean_visited_frames": summary.get("mean_visited_frames"),
                    "mean_visual_tokens_projected": summary.get("mean_visual_tokens_projected"),
                    "mean_end_to_end_seconds": summary.get("mean_end_to_end_seconds"),
                    "by_category": summary.get("by_category") or {},
                    "prediction_counts": summary.get("prediction_counts") or {},
                    "timing_summary_seconds": summary.get("timing_summary_seconds") or {},
                    "proxy_metric": summary.get("proxy_metric"),
                    "proxy_base_metric": summary.get("proxy_base_metric"),
                    "semantic_model": summary.get("semantic_model"),
                    "lambda_cost": summary.get("lambda_cost"),
                    "selection": summary.get("selection"),
                    "sampling": summary.get("sampling"),
                    "results_path": str(summary_path.parent / "results.jsonl"),
                }
            )

    deduped = {}
    for row in rows:
        deduped[row["run_name"]] = row
    return list(deduped.values())


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
    save_png_figure(fig, output_path)


def plot_accuracy_delta_vs_uniform(rows: list[dict], baseline: dict, output_path: Path) -> None:
    rows = [row for row in rows if row is not baseline]
    rows = sorted(rows, key=lambda row: row["accuracy"] - baseline["accuracy"])
    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.34 * len(rows))))
    labels = [short_benchmark_label(row) for row in rows]
    deltas = [(row["accuracy"] - baseline["accuracy"]) * 100.0 for row in rows]
    colors = ["#2c7fb8" if value >= 0 else "#d95f0e" for value in deltas]
    ax.barh(labels, deltas, color=colors, alpha=0.86)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Accuracy delta vs uniform_128 (percentage points)")
    ax.set_title("HLVid Proxy Accuracy Relative to Uniform 128")
    save_png_figure(fig, output_path)


def plot_category_accuracy_delta_vs_uniform(rows: list[dict], baseline: dict, output_path: Path) -> None:
    categories = sorted(baseline.get("by_category", {}))
    if not categories:
        return
    rows = select_representative_rows(rows, baseline, limit=10)
    labels = [short_benchmark_label(row) for row in rows]
    x = np.arange(len(labels))
    width = min(0.8 / max(len(categories), 1), 0.34)

    fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(labels)), 5.2))
    for index, category in enumerate(categories):
        base_accuracy = baseline["by_category"][category]["accuracy"]
        deltas = [
            (row.get("by_category", {}).get(category, {}).get("accuracy", base_accuracy) - base_accuracy) * 100.0
            for row in rows
        ]
        ax.bar(x + (index - (len(categories) - 1) / 2) * width, deltas, width=width, label=category)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Accuracy delta vs uniform_128 (percentage points)")
    ax.set_title("Category-Level Accuracy Delta")
    ax.legend(frameon=True)
    save_png_figure(fig, output_path)


def plot_efficiency_vs_uniform(rows: list[dict], baseline: dict, output_path: Path) -> None:
    rows = select_representative_rows(rows, baseline, limit=10)
    labels = [short_benchmark_label(row) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(3, 1, figsize=(max(9.0, 0.6 * len(rows)), 8.2), sharex=True)
    accuracy_delta = [(row["accuracy"] - baseline["accuracy"]) * 100.0 for row in rows]
    token_reduction = [relative_reduction(row, baseline, "mean_visual_tokens_projected") * 100.0 for row in rows]
    frame_reduction = [relative_reduction(row, baseline, "mean_visited_frames") * 100.0 for row in rows]
    time_ratio = [relative_ratio(row, baseline, "mean_end_to_end_seconds") for row in rows]

    axes[0].bar(x, accuracy_delta, color="#2c7fb8", alpha=0.86)
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("Accuracy delta\n(pp)")

    width = 0.38
    axes[1].bar(x - width / 2, token_reduction, width=width, color="#41ab5d", alpha=0.86, label="Visual tokens")
    axes[1].bar(x + width / 2, frame_reduction, width=width, color="#756bb1", alpha=0.86, label="Visited frames")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("Reduction\n(%)")
    axes[1].legend(frameon=True, fontsize=8)

    axes[2].bar(x, time_ratio, color="#d95f0e", alpha=0.86)
    axes[2].axhline(1.0, color="black", linewidth=1.0)
    axes[2].set_ylabel("Time ratio")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    axes[2].set_title("End-to-end time relative to uniform_128")

    fig.suptitle("Proxy Efficiency Relative to Uniform 128")
    save_png_figure(fig, output_path)


def plot_timing_breakdown(rows: list[dict], baseline: dict | None, output_path: Path) -> None:
    rows = select_representative_rows(rows, baseline, limit=10)
    rows = [row for row in rows if row.get("timing_summary_seconds")]
    if not rows:
        return
    labels = [short_benchmark_label(row) for row in rows]
    components = [
        ("frame_load", "Frame load"),
        ("proxy", "Proxy"),
        ("processor", "Processor"),
        ("llm_inference", "LLM"),
        ("system_overhead", "Overhead"),
    ]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(rows)), 5.5))
    for key, label in components:
        values = [
            float(row.get("timing_summary_seconds", {}).get(key, {}).get("mean", 0.0) or 0.0)
            for row in rows
        ]
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += np.array(values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean seconds per example")
    ax.set_title("HLVid Benchmark Timing Breakdown")
    ax.legend(frameon=True, fontsize=8)
    save_png_figure(fig, output_path)


def plot_per_question_uniform_comparison(rows: list[dict], baseline: dict, output_path: Path) -> None:
    baseline_results = read_results_by_question(Path(baseline["results_path"]))
    if not baseline_results:
        return
    rows = [row for row in select_representative_rows(rows, baseline, limit=10) if row is not baseline]
    comparison_rows = []
    for row in rows:
        results = read_results_by_question(Path(row["results_path"]))
        if not results:
            continue
        counts = Counter()
        for question_id, base_result in baseline_results.items():
            if question_id not in results:
                continue
            base_correct = bool(base_result.get("is_correct"))
            run_correct = bool(results[question_id].get("is_correct"))
            if base_correct and run_correct:
                counts["both_correct"] += 1
            elif run_correct:
                counts["proxy_only"] += 1
            elif base_correct:
                counts["uniform_only"] += 1
            else:
                counts["both_wrong"] += 1
        comparison_rows.append((row, counts))
    if not comparison_rows:
        return

    labels = [short_benchmark_label(row) for row, _ in comparison_rows]
    x = np.arange(len(comparison_rows))
    components = [
        ("both_correct", "Both correct", "#74c476"),
        ("proxy_only", "Proxy only", "#2b8cbe"),
        ("uniform_only", "Uniform only", "#fdae6b"),
        ("both_wrong", "Both wrong", "#bdbdbd"),
    ]
    bottom = np.zeros(len(comparison_rows))
    fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(comparison_rows)), 5.5))
    for key, label, color in components:
        values = [counts[key] for _, counts in comparison_rows]
        ax.bar(x, values, bottom=bottom, label=label, color=color, alpha=0.9)
        bottom += np.array(values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Number of questions")
    ax.set_title("Per-Question Correctness Compared with Uniform 128")
    ax.legend(frameon=True, fontsize=8)
    save_png_figure(fig, output_path)


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


def annotate_frontier(ax, rows: list[dict], x_key: str) -> None:
    for row in rows:
        ax.annotate(short_benchmark_label(row), (row[x_key], row["accuracy"]), fontsize=7, xytext=(4, 4), textcoords="offset points")


def find_uniform_baseline(rows: list[dict]) -> dict | None:
    uniform_rows = [row for row in rows if row.get("selection") == "uniform"]
    for row in uniform_rows:
        if row.get("sampling") == "128" or row.get("run_name") == "uniform_128":
            return row
    return uniform_rows[0] if uniform_rows else None


def select_representative_rows(rows: list[dict], baseline: dict | None, limit: int) -> list[dict]:
    selected = []
    if baseline is not None:
        selected.append(baseline)
    for row in pareto_frontier([row for row in rows if isinstance(row.get("mean_visual_tokens_projected"), (int, float))], "mean_visual_tokens_projected"):
        if row not in selected:
            selected.append(row)
    for row in sorted(rows, key=lambda item: item["accuracy"], reverse=True):
        if row not in selected:
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected[:limit]


def relative_reduction(row: dict, baseline: dict, key: str) -> float:
    base_value = baseline.get(key)
    value = row.get(key)
    if not isinstance(base_value, (int, float)) or not base_value:
        return 0.0
    if not isinstance(value, (int, float)):
        return 0.0
    return 1.0 - value / base_value


def relative_ratio(row: dict, baseline: dict, key: str) -> float:
    base_value = baseline.get(key)
    value = row.get(key)
    if not isinstance(base_value, (int, float)) or not base_value:
        return 0.0
    if not isinstance(value, (int, float)):
        return 0.0
    return value / base_value


def read_results_by_question(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row.get("question_id")
            if isinstance(question_id, int):
                rows[question_id] = row
    return rows


def benchmark_family(row: dict) -> str:
    if row.get("selection") == "uniform":
        return "Uniform"
    return METHOD_LABELS.get(proxy_identity(row), row.get("proxy_metric") or "Other")


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


def write_plot_data(rows: list[dict], output_dir: Path) -> None:
    (output_dir / "benchmark_plot_rows.json").write_text(json.dumps(rows, indent=2) + "\n")


def save_png_figure(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
