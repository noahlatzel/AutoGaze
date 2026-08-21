# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot learning and endpoint diagnostics for the R2d variable-budget gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_name(cfg, base_seed: int, training_seed: int) -> str:
    return cfg["run_template"].format(
        base_seed=base_seed,
        training_seed=training_seed,
    )


def style(axis) -> None:
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def complete_series(histories, key):
    by_step = defaultdict(list)
    for history in histories:
        for record in history:
            if key in record:
                by_step[int(record["train_step"])].append(float(record[key]))
    steps = np.array(
        sorted(step for step, values in by_step.items() if len(values) == len(histories))
    )
    values = [np.asarray(by_step[int(step)]) for step in steps]
    return steps, np.asarray([value.mean() for value in values]), np.asarray(
        [value.std() for value in values]
    )


def fixed_k16_endpoints(cfg, root: Path) -> list[float]:
    values = []
    for base_seed in cfg["base_seeds"]:
        path = root / cfg["fixed_k16_run_template"].format(
            base_seed=base_seed,
            continuation_seed=base_seed + 100000,
        ) / "validation_metrics.jsonl"
        values.append(float(read_history(path)[-1]["coverage_k16_macro_source"]))
    return values


def plot_learning(
    cfg, histories, fixed_values, baselines, output: Path, x_max: int | None = None
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True, constrained_layout=True)
    metrics = (
        ("coverage_macro_source", "Validation macro-source coverage"),
        ("mean_tokens_macro_source", "Validation macro-source mean K"),
        ("token_cost", "Adaptive token price λ"),
    )
    for axis, (metric, ylabel) in zip(axes, metrics):
        for seed, history in zip(cfg["training_seeds"], histories):
            axis.plot(
                [record["train_step"] for record in history if metric in record],
                [record[metric] for record in history if metric in record],
                alpha=0.30,
                linewidth=1,
                label=f"Seed {seed}" if metric == metrics[0][0] else None,
            )
        steps, means, stds = complete_series(histories, metric)
        axis.plot(steps, means, color="black", linewidth=2.3, label="Three-seed mean")
        axis.fill_between(steps, means - stds, means + stds, color="black", alpha=0.12)
        axis.set_ylabel(ylabel)
        style(axis)

    fixed_mean = float(np.mean(fixed_values))
    axes[0].axhline(fixed_mean, color="#1677b8", linestyle="--", label="Matched fixed K16")
    val = baselines["splits"]["val"]
    for method, label, color, linestyle in (
        ("center", "Center-16", "#666666", ":"),
        ("prior", "Prior-16", "#d95f02", "-."),
    ):
        axes[0].axhline(
            val[method]["16"]["macro_source_mean"],
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].set_title("R2d learned variable fine-token budget")
    axes[1].axhspan(
        float(cfg["matched_budget_min"]),
        float(cfg["matched_budget_max"]),
        color="#1b9e77",
        alpha=0.15,
        label="Matched-compute band",
    )
    axes[1].axhline(16, color="#1b9e77", linestyle="--")
    axes[1].legend(frameon=False)
    axes[-1].set_xlabel("Additional optimizer updates from fixed-K16 initialization")
    axes[-1].set_xlim(0, int(cfg["total_updates"]) if x_max is None else x_max)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_per_source(cfg, histories, baselines, output: Path) -> None:
    sources = list(baselines["splits"]["val"]["prior"]["16"]["per_source"])
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, constrained_layout=True)
    for axis, source in zip(axes.flat, sources):
        coverage_key = f"coverage_source_{source}"
        length_key = f"mean_tokens_source_{source}"
        steps, coverage_mean, coverage_std = complete_series(histories, coverage_key)
        axis.plot(steps, coverage_mean, color="#1677b8", linewidth=2, label="Coverage")
        axis.fill_between(
            steps, coverage_mean - coverage_std, coverage_mean + coverage_std,
            color="#1677b8", alpha=0.12,
        )
        axis.axhline(
            baselines["splits"]["val"]["prior"]["16"]["per_source"][source]["mean_video_coverage"],
            color="#d95f02", linestyle="-.", linewidth=1, label="Prior-16",
        )
        length_axis = axis.twinx()
        length_steps, length_mean, _ = complete_series(histories, length_key)
        length_axis.plot(length_steps, length_mean, color="#1b9e77", linestyle="--", linewidth=1.4, label="Mean K")
        length_axis.axhline(16, color="#1b9e77", linestyle=":", linewidth=1)
        length_axis.set_ylim(4, 36)
        axis.set_title(source)
        style(axis)
        if axis in axes[:, -1]:
            length_axis.set_ylabel("Mean K", color="#1b9e77")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean-video coverage")
    for axis in axes[-1]:
        axis.set_xlabel("Optimizer updates")
    axes.flat[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Variable-budget validation trajectories by source")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_endpoint(cfg, reports, fixed_values, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    methods = (
        ("variable", "Variable K", "#1677b8"),
        ("forced_k16", "Forced K16", "#666666"),
        ("actual_k_forced_order", "Actual K on forced order", "#e6ab02"),
        ("same_source_shuffled_k", "Shuffled K", "#7b3294"),
        ("oracle_variable", "Oracle variable", "#1b9e77"),
    )
    x = np.arange(len(reports))
    width = 0.15
    for index, (method, label, color) in enumerate(methods):
        values = [report["coverage"][method]["16"]["macro_source_mean"] for report in reports]
        axes[0, 0].bar(x + (index - 2) * width, values, width, label=label, color=color)
    axes[0, 0].plot(x, fixed_values, color="black", marker="o", linestyle="--", label="Matched fixed K16")
    axes[0, 0].set_xticks(x, [str(seed) for seed in cfg["training_seeds"]])
    axes[0, 0].set_ylabel("Validation macro-source coverage")
    axes[0, 0].set_title("Matched-compute endpoint controls")
    axes[0, 0].legend(frameon=False, fontsize=8)

    length_values = np.arange(int(cfg["min_budget"]), int(cfg["max_budget"]) + 1)
    for seed, report in zip(cfg["training_seeds"], reports):
        histogram = report["diagnostics"]["length_histogram"]
        counts = np.array([histogram.get(str(int(value)), 0) for value in length_values], dtype=float)
        axes[0, 1].plot(length_values, counts / counts.sum(), marker=".", label=str(seed))
    axes[0, 1].axvline(16, color="black", linestyle="--")
    axes[0, 1].set_xlabel("Selected fine tokens")
    axes[0, 1].set_ylabel("Fraction of validation frames")
    axes[0, 1].set_title("Learned length distributions")
    axes[0, 1].legend(frameon=False)

    for relationship, label, color in (
        ("entropy_bins", "Gaze entropy", "#d95f02"),
        ("center_mass_bins", "Center-16 mass", "#1b9e77"),
    ):
        for report in reports:
            bins = report["relationships"][relationship]
            axes[1, 0].plot(
                [item["feature_mean"] for item in bins],
                [item["mean_tokens"] for item in bins],
                color=color,
                alpha=0.35,
            )
        bins_by_index = list(zip(*(report["relationships"][relationship] for report in reports)))
        axes[1, 0].plot(
            [np.mean([item["feature_mean"] for item in values]) for values in bins_by_index],
            [np.mean([item["mean_tokens"] for item in values]) for values in bins_by_index],
            color=color,
            marker="o",
            linewidth=2.2,
            label=label,
        )
    axes[1, 0].axhline(16, color="black", linestyle="--")
    axes[1, 0].set_xlabel("Feature value (within-feature quantile bins)")
    axes[1, 0].set_ylabel("Mean K")
    axes[1, 0].set_title("What predicts allocated length?")
    axes[1, 0].legend(frameon=False)

    positions = np.arange(1, int(cfg["max_budget"]) + 1)
    for group, label, color in (
        ("all", "All frames", "black"),
        ("short_4_12", "Short", "#d95f02"),
        ("medium_13_20", "Medium", "#7570b3"),
        ("long_21_36", "Long", "#1b9e77"),
    ):
        arrays = [np.asarray(report["marginal_coverage_gain"][group]) for report in reports]
        if arrays[0].size:
            values = np.stack(arrays)
            axes[1, 1].plot(positions, values.mean(axis=0), color=color, label=label)
            axes[1, 1].fill_between(
                positions,
                values.mean(axis=0) - values.std(axis=0),
                values.mean(axis=0) + values.std(axis=0),
                color=color,
                alpha=0.10,
            )
    axes[1, 1].axvline(16, color="#777777", linestyle="--")
    axes[1, 1].set_xlabel("Fine-token position")
    axes[1, 1].set_ylabel("Mean marginal gaze mass")
    axes[1, 1].set_title("Marginal coverage by learned length group")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        style(axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--learning-only",
        action="store_true",
        help="Render available learning curves without requiring endpoint diagnostics.",
    )
    parser.add_argument("--x-max", type=int)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    histories = []
    for base_seed, training_seed in zip(cfg["base_seeds"], cfg["training_seeds"]):
        run = run_name(cfg, int(base_seed), int(training_seed))
        histories.append(read_history(args.runs_root / run / "validation_metrics.jsonl"))
    fixed_values = fixed_k16_endpoints(cfg, Path("."))
    baselines = read_json(Path(cfg["fixed_baselines"]))
    plot_learning(
        cfg, histories, fixed_values, baselines,
        args.output_dir / "r2d_variable_budget_learning.png",
        x_max=args.x_max,
    )
    if args.learning_only:
        return
    reports = []
    for base_seed, training_seed in zip(cfg["base_seeds"], cfg["training_seeds"]):
        run = run_name(cfg, int(base_seed), int(training_seed))
        reports.append(read_json(args.diagnostics_root / run / "variable_budget_val.json"))
    plot_per_source(
        cfg, histories, baselines,
        args.output_dir / "r2d_variable_budget_by_source.png",
    )
    plot_endpoint(
        cfg, reports, fixed_values,
        args.output_dir / "r2d_variable_budget_endpoint_controls.png",
    )
    print(json.dumps({
        "fixed_k16_mean": float(np.mean(fixed_values)),
        "variable_mean": float(np.mean([
            report["coverage"]["variable"]["16"]["macro_source_mean"]
            for report in reports
        ])),
        "val_mean_k": float(np.mean([
            report["length"]["variable"]["macro_source_mean"]
            for report in reports
        ])),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
