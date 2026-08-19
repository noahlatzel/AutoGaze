# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the matched K24/K36 fixed-budget training sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from scripts.human_gaze.plot_fresh20k_analysis import (
    complete_series,
    read_history,
    read_json,
    style_axis,
)


def format_run(template: str, budget: int, base_seed: int, continuation_seed: int) -> str:
    return template.format(
        budget=budget,
        base_seed=base_seed,
        continuation_seed=continuation_seed,
    )


def stitched_history(
    cfg: dict,
    root: Path,
    budget: int,
    base_seed: int,
    continuation_seed: int,
) -> list[dict]:
    paths = [
        root / format_run(cfg[key], budget, base_seed, continuation_seed)
        for key in ("stage1_run_template", "stage2_run_template", "stage3_run_template")
    ]
    records = []
    for record in read_history(paths[0] / "validation_metrics.jsonl"):
        records.append({**record, "cumulative_step": int(record["train_step"])})
    for record in read_history(paths[1] / "validation_metrics.jsonl"):
        train_step = int(record["train_step"])
        if train_step > 0:
            records.append({**record, "cumulative_step": cfg["stage1_updates"] + train_step})
    for record in read_history(paths[2] / "validation_metrics.jsonl"):
        train_step = int(record["train_step"])
        if train_step > cfg["stage2_updates"]:
            records.append({**record, "cumulative_step": cfg["stage1_updates"] + train_step})
    return sorted(records, key=lambda record: int(record["cumulative_step"]))


def all_histories(cfg: dict, root: Path, budget: int) -> list[list[dict]]:
    return [
        stitched_history(cfg, root, budget, int(base_seed), int(continuation_seed))
        for base_seed, continuation_seed in zip(cfg["base_seeds"], cfg["continuation_seeds"])
    ]


def pretrained_macro(cfg: dict, budget: int) -> float:
    if budget == 16:
        report = read_json(Path(cfg["pretrained_k16"]))
    else:
        report = read_json(Path(cfg["pretrained_template"].format(budget=budget)))
    return float(report["metrics"]["val"][str(budget)]["macro_source_mean"])


def plot_learning(
    cfg: dict,
    histories_by_budget: dict[int, list[list[dict]]],
    baselines: dict,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharex=True, constrained_layout=True)
    for axis, budget in zip(axes, cfg["budgets"]):
        histories = histories_by_budget[int(budget)]
        metric = f"coverage_k{budget}_macro_source"
        for seed, history in zip(cfg["base_seeds"], histories):
            axis.plot(
                [record["cumulative_step"] for record in history],
                [record[metric] for record in history],
                linewidth=1.0,
                alpha=0.34,
                label=f"Seed {seed}",
            )
        steps, means, stds = complete_series(histories, metric)
        axis.plot(steps, means, color="black", linewidth=2.4, label="Three-seed mean")
        axis.fill_between(steps, means - stds, means + stds, color="black", alpha=0.13)
        val = baselines["splits"]["val"]
        for value, label, linestyle in (
            (pretrained_macro(cfg, int(budget)), "Pretrained", "--"),
            (val["center"][str(budget)]["macro_source_mean"], f"Center-{budget}", ":"),
            (val["prior"][str(budget)]["macro_source_mean"], f"Prior-{budget}", "-."),
        ):
            axis.axhline(value, color="#555555", linestyle=linestyle, linewidth=1.1, label=label)
        for step in (cfg["stage1_updates"], 10000):
            axis.axvline(step, color="#999999", linestyle="--", linewidth=0.8)
        axis.set_title(f"Fixed K={budget}")
        axis.set_xlabel("Cumulative optimizer updates")
        axis.set_ylabel("Validation macro-source coverage")
        axis.set_xlim(0, cfg["total_updates"])
        style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Fresh decoder-only direct-coverage GRPO by fixed token budget", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_per_source(
    cfg: dict,
    histories: list[list[dict]],
    baselines: dict,
    budget: int,
    output: Path,
) -> None:
    val = baselines["splits"]["val"]
    prior = val["prior"][str(budget)]["per_source"]
    center = val["center"][str(budget)]["per_source"]
    sources = list(prior)
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.7), sharex=True, constrained_layout=True)
    for axis, source in zip(axes.flat, sources):
        metric = f"coverage_k{budget}_source_{source}"
        for history in histories:
            axis.plot(
                [record["cumulative_step"] for record in history],
                [record[metric] for record in history],
                linewidth=0.9,
                alpha=0.28,
            )
        steps, means, stds = complete_series(histories, metric)
        axis.plot(steps, means, color="black", linewidth=2.0, label="Three-seed mean")
        axis.fill_between(steps, means - stds, means + stds, color="black", alpha=0.12)
        axis.axhline(
            prior[source]["mean_video_coverage"],
            color="#d95f02",
            linestyle="-.",
            linewidth=1.1,
            label=f"Prior-{budget}",
        )
        axis.axhline(
            center[source]["mean_video_coverage"],
            color="#666666",
            linestyle=":",
            linewidth=1.1,
            label=f"Center-{budget}",
        )
        axis.axvline(10000, color="#999999", linestyle="--", linewidth=0.8)
        axis.set_title(source)
        axis.set_xlim(0, cfg["total_updates"])
        style_axis(axis)
    axes.flat[0].legend(frameon=False, fontsize=7, loc="lower right")
    for axis in axes[:, 0]:
        axis.set_ylabel("Validation mean-video coverage")
    for axis in axes[-1, :]:
        axis.set_xlabel("Cumulative optimizer updates")
    figure.suptitle(f"Validation K={budget} trajectories by source", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def diagnostic_report(
    cfg: dict,
    root: Path,
    budget: int,
    base_seed: int,
    continuation_seed: int,
    cumulative_step: int,
) -> dict:
    template = "stage2_run_template" if cumulative_step == 10000 else "stage3_run_template"
    run = format_run(cfg[template], budget, base_seed, continuation_seed)
    return read_json(root / run / f"center_collapse_val_step{cumulative_step}.json")


def load_diagnostics(cfg: dict, root: Path) -> dict[int, dict[int, list[dict]]]:
    result = {}
    for budget in cfg["budgets"]:
        by_step = {}
        for step in cfg["diagnostics"]["checkpoints_cumulative"]:
            by_step[int(step)] = [
                diagnostic_report(
                    cfg,
                    root,
                    int(budget),
                    int(base_seed),
                    int(continuation_seed),
                    int(step),
                )
                for base_seed, continuation_seed in zip(
                    cfg["base_seeds"], cfg["continuation_seeds"]
                )
            ]
        result[int(budget)] = by_step
    return result


def report_values(reports: list[dict], path: tuple[str, ...]) -> np.ndarray:
    values = []
    for report in reports:
        value = report
        for key in path:
            value = value[key]
        values.append(float(value))
    return np.array(values)


def plot_collapse(diagnostics: dict[int, dict[int, list[dict]]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, budget in enumerate(sorted(diagnostics)):
        by_step = diagnostics[budget]
        steps = np.array(sorted(by_step))
        static_method = f"selection_frequency_top{budget}"
        for method, label, color, linestyle in (
            ("actual", "Dynamic policy", "#1677b8", "-"),
            ("center", f"Center-{budget}", "#666666", ":"),
            (static_method, f"Static learned Top-{budget}", "#d95f02", "-."),
            ("shuffled", "Same-source shuffled video", "#7b3294", "--"),
        ):
            arrays = [
                report_values(by_step[int(step)], ("coverage", method, str(budget), "macro_source_mean"))
                for step in steps
            ]
            means = np.array([values.mean() for values in arrays])
            stds = np.array([values.std() for values in arrays])
            axes[row, 0].plot(steps, means, marker="o", color=color, linestyle=linestyle, label=label)
            axes[row, 0].fill_between(steps, means - stds, means + stds, color=color, alpha=0.10)
        for metric, label, color, linestyle in (
            ("mean_center_overlap_fraction", f"Center-{budget} overlap", "#d95f02", "-"),
            ("normalized_selection_entropy", "Normalized selection entropy", "#1b9e77", "--"),
            ("mean_actual_shuffled_overlap_fraction", "Actual/shuffled overlap", "#7b3294", ":"),
        ):
            arrays = [report_values(by_step[int(step)], ("diagnostics", metric)) for step in steps]
            means = np.array([values.mean() for values in arrays])
            stds = np.array([values.std() for values in arrays])
            axes[row, 1].plot(steps, means, marker="o", color=color, linestyle=linestyle, label=label)
            axes[row, 1].fill_between(steps, means - stds, means + stds, color=color, alpha=0.10)
        axes[row, 0].set_title(f"K={budget}: dynamic and static controls")
        axes[row, 1].set_title(f"K={budget}: concentration and content dependence")
        axes[row, 0].set_ylabel("Validation macro-source coverage")
        axes[row, 1].set_ylabel("Fraction / normalized entropy")
        for axis in axes[row]:
            axis.set_xlabel("Cumulative optimizer updates")
            axis.set_xticks(steps)
            axis.legend(frameon=False, fontsize=8)
            style_axis(axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def final_k16(cfg: dict, root: Path) -> tuple[float, float]:
    values = []
    for base_seed in cfg["reference_k16"]["base_seeds"]:
        continuation_seed = int(base_seed) + 100000
        run = cfg["reference_k16"]["stage3_run_template"].format(
            base_seed=base_seed,
            continuation_seed=continuation_seed,
        )
        records = read_history(root / run / "validation_metrics.jsonl")
        final = next(record for record in records if int(record["train_step"]) == cfg["stage3_final_step"])
        values.append(float(final["coverage_k16_macro_source"]))
    return float(np.mean(values)), float(np.std(values))


def plot_budget_summary(
    cfg: dict,
    histories_by_budget: dict[int, list[list[dict]]],
    baselines: dict,
    runs_root: Path,
    output: Path,
) -> None:
    budgets = np.array([16, *[int(value) for value in cfg["budgets"]]])
    learned_means = []
    learned_stds = []
    k16_mean, k16_std = final_k16(cfg, runs_root)
    learned_means.append(k16_mean)
    learned_stds.append(k16_std)
    for budget in cfg["budgets"]:
        histories = histories_by_budget[int(budget)]
        values = [
            next(
                float(record[f"coverage_k{budget}_macro_source"])
                for record in reversed(history)
                if int(record["cumulative_step"]) == cfg["total_updates"]
            )
            for history in histories
        ]
        learned_means.append(float(np.mean(values)))
        learned_stds.append(float(np.std(values)))
    val = baselines["splits"]["val"]
    figure, axis = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    axis.errorbar(
        budgets,
        learned_means,
        yerr=learned_stds,
        marker="o",
        linewidth=2.2,
        capsize=4,
        label="Learned at 20k",
    )
    for method, label, linestyle in (
        ("prior", "Source Prior-K", "-."),
        ("center", "Center-K", ":"),
        ("oracle", "Per-frame Oracle-K", "--"),
    ):
        axis.plot(
            budgets,
            [val[method][str(int(budget))]["macro_source_mean"] for budget in budgets],
            marker="o",
            linestyle=linestyle,
            label=label,
        )
    axis.plot(
        budgets,
        [pretrained_macro(cfg, int(budget)) for budget in budgets],
        marker="o",
        linestyle="--",
        label="Pretrained AutoGaze",
    )
    axis.set_xticks(budgets)
    axis.set_xlabel("Fixed fine tokens per frame")
    axis.set_ylabel("Validation macro-source coverage")
    axis.set_title("Coverage scaling with fixed gaze budget")
    axis.legend(frameon=False)
    style_axis(axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    histories_by_budget = {
        int(budget): all_histories(cfg, args.runs_root, int(budget))
        for budget in cfg["budgets"]
    }
    baselines = read_json(Path(cfg["static_baselines"]))
    diagnostics = load_diagnostics(cfg, args.diagnostics_root)
    plot_learning(
        cfg,
        histories_by_budget,
        baselines,
        args.output_dir / "r2c_fixed_budget_learning_curves.png",
    )
    for budget in cfg["budgets"]:
        plot_per_source(
            cfg,
            histories_by_budget[int(budget)],
            baselines,
            int(budget),
            args.output_dir / f"r2c_fixedk{budget}_validation_by_source.png",
        )
    plot_collapse(
        diagnostics,
        args.output_dir / "r2c_fixed_budget_center_collapse_longitudinal.png",
    )
    plot_budget_summary(
        cfg,
        histories_by_budget,
        baselines,
        args.runs_root,
        args.output_dir / "r2c_fixed_budget_coverage_scaling.png",
    )


if __name__ == "__main__":
    main()
