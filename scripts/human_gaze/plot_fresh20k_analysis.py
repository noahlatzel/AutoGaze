# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the preregistered aggregate views for the six-seed 20k continuation."""

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


def run_path(root: Path, template: str, base_seed: int, continuation_seed: int) -> Path:
    return root / template.format(
        base_seed=base_seed,
        continuation_seed=continuation_seed,
    )


def stitched_history(cfg: dict, root: Path, base_seed: int, continuation_seed: int) -> list[dict]:
    stage1 = run_path(root, cfg["stage1_run_template"], base_seed, continuation_seed)
    stage2 = run_path(root, cfg["stage2_run_template"], base_seed, continuation_seed)
    stage3 = run_path(root, cfg["stage3_run_template"], base_seed, continuation_seed)
    records = []
    for record in read_history(stage1 / "validation_metrics.jsonl"):
        records.append({**record, "cumulative_step": int(record["train_step"])})
    for record in read_history(stage2 / "validation_metrics.jsonl"):
        train_step = int(record["train_step"])
        if train_step > 0:
            records.append(
                {**record, "cumulative_step": int(cfg["stage1_updates"]) + train_step}
            )
    for record in read_history(stage3 / "validation_metrics.jsonl"):
        train_step = int(record["train_step"])
        if train_step > int(cfg["stage2_updates"]):
            records.append(
                {**record, "cumulative_step": int(cfg["stage1_updates"]) + train_step}
            )
    return sorted(records, key=lambda record: int(record["cumulative_step"]))


def complete_series(histories: list[list[dict]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_step = defaultdict(list)
    for history in histories:
        for record in history:
            if key in record:
                by_step[int(record["cumulative_step"])].append(float(record[key]))
    steps = np.array(
        sorted(step for step, values in by_step.items() if len(values) == len(histories)),
        dtype=int,
    )
    means = np.array([np.mean(by_step[int(step)]) for step in steps])
    stds = np.array([np.std(by_step[int(step)]) for step in steps])
    return steps, means, stds


def style_axis(axis) -> None:
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def plot_learning(cfg: dict, histories: list[list[dict]], output: Path) -> None:
    figure, (axis, gain_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        height_ratios=(3.3, 1.25),
        sharex=True,
        constrained_layout=True,
    )
    key = "coverage_k16_macro_source"
    for seed, history in zip(cfg["base_seeds"], histories):
        axis.plot(
            [record["cumulative_step"] for record in history],
            [record[key] for record in history],
            linewidth=1.05,
            alpha=0.34,
            label=f"Seed {seed}",
        )
    steps, means, stds = complete_series(histories, key)
    axis.plot(steps, means, color="black", linewidth=2.5, label="Six-seed mean")
    axis.fill_between(
        steps,
        means - stds,
        means + stds,
        color="black",
        alpha=0.13,
        label="Population SD",
    )
    for key_name, label, linestyle in (
        ("pretrained", "Pretrained", "--"),
        ("center16", "Center-16", ":"),
        ("prior16", "Prior-16", "-."),
    ):
        axis.axhline(
            float(cfg["baselines"][key_name]),
            color="#555555",
            linestyle=linestyle,
            linewidth=1.1,
            label=label,
        )
    for step, label in (
        (int(cfg["stage1_updates"]), "Adam reset; 3e-6"),
        (10000, "Optimizer-preserving continuation"),
    ):
        axis.axvline(step, color="#888888", linestyle="--", linewidth=0.9)
        axis.annotate(
            label,
            xy=(step, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(5, -2),
            textcoords="offset points",
            rotation=90,
            va="top",
            fontsize=8,
            color="#666666",
        )
    eligible = steps >= 1000
    gains = means[eligible] - np.interp(steps[eligible] - 1000, steps, means)
    gain_axis.plot(steps[eligible], gains, color="#1f77b4", linewidth=1.8)
    gain_axis.axhline(0, color="#555555", linewidth=0.9)
    gain_axis.axvline(10000, color="#888888", linestyle="--", linewidth=0.9)
    gain_axis.set_ylabel("Rolling 1k-step\ngain")
    gain_axis.set_xlabel("Cumulative optimizer updates")
    gain_axis.set_xlim(0, int(cfg["total_updates"]))
    axis.set_ylabel("Validation macro-source K=16 coverage")
    axis.set_title("Fresh decoder-only direct-coverage GRPO: 20k continuation")
    axis.legend(ncol=3, frameon=False, fontsize=8)
    style_axis(axis)
    style_axis(gain_axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_per_source(
    cfg: dict,
    histories: list[list[dict]],
    baselines: dict,
    output: Path,
) -> None:
    prior = baselines["splits"]["val"]["prior"]["16"]["per_source"]
    center = baselines["splits"]["val"]["center"]["16"]["per_source"]
    sources = list(prior)
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.7), sharex=True, constrained_layout=True)
    for axis, source in zip(axes.flat, sources):
        metric = f"coverage_k16_source_{source}"
        for history in histories:
            axis.plot(
                [record["cumulative_step"] for record in history],
                [record[metric] for record in history],
                linewidth=0.9,
                alpha=0.27,
            )
        steps, means, stds = complete_series(histories, metric)
        axis.plot(steps, means, color="black", linewidth=2.0, label="Six-seed mean")
        axis.fill_between(steps, means - stds, means + stds, color="black", alpha=0.12)
        axis.axhline(
            prior[source]["mean_video_coverage"],
            color="#d95f02",
            linestyle="-.",
            linewidth=1.1,
            label="Prior-16",
        )
        axis.axhline(
            center[source]["mean_video_coverage"],
            color="#666666",
            linestyle=":",
            linewidth=1.1,
            label="Center-16",
        )
        axis.axvline(10000, color="#999999", linestyle="--", linewidth=0.8)
        axis.set_title(source)
        axis.set_xlim(0, int(cfg["total_updates"]))
        style_axis(axis)
    for axis in axes[:, 0]:
        axis.set_ylabel("Validation mean-video coverage")
    for axis in axes[-1, :]:
        axis.set_xlabel("Cumulative optimizer updates")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="lower right")
    figure.suptitle("Validation K=16 trajectories by source", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def diagnostic_path(
    cfg: dict,
    root: Path,
    base_seed: int,
    continuation_seed: int,
    cumulative_step: int,
) -> Path:
    if cumulative_step == 10000:
        run = run_path(root, cfg["stage2_run_template"], base_seed, continuation_seed)
        return run / "center_collapse_val_within_source.json"
    run = run_path(root, cfg["stage3_run_template"], base_seed, continuation_seed)
    return run / f"center_collapse_val_step{cumulative_step}.json"


def load_diagnostics(cfg: dict, root: Path) -> dict[int, list[dict]]:
    result = defaultdict(list)
    for base_seed, continuation_seed in zip(cfg["base_seeds"], cfg["continuation_seeds"]):
        for step in cfg["diagnostics"]["checkpoints_cumulative"]:
            result[int(step)].append(
                read_json(
                    diagnostic_path(
                        cfg,
                        root,
                        int(base_seed),
                        int(continuation_seed),
                        int(step),
                    )
                )
            )
    return dict(result)


def diagnostic_values(reports: list[dict], path: tuple[str, ...]) -> np.ndarray:
    values = []
    for report in reports:
        value = report
        for key in path:
            value = value[key]
        values.append(float(value))
    return np.array(values)


def plot_collapse(diagnostics: dict[int, list[dict]], output: Path) -> None:
    steps = np.array(sorted(diagnostics))
    figure, (coverage_axis, structure_axis) = plt.subplots(
        1, 2, figsize=(12.8, 5.4), constrained_layout=True
    )
    for method, label, color, linestyle in (
        ("actual", "Dynamic policy", "#1677b8", "-"),
        ("center", "Center-16", "#666666", ":"),
        ("selection_frequency_top16", "Static learned Top-16", "#d95f02", "-."),
        ("shuffled", "Same-source shuffled video", "#7b3294", "--"),
    ):
        arrays = [
            diagnostic_values(
                diagnostics[int(step)],
                ("coverage", method, "16", "macro_source_mean"),
            )
            for step in steps
        ]
        means = np.array([values.mean() for values in arrays])
        stds = np.array([values.std() for values in arrays])
        coverage_axis.plot(steps, means, marker="o", color=color, linestyle=linestyle, label=label)
        coverage_axis.fill_between(steps, means - stds, means + stds, color=color, alpha=0.10)
    for metric, label, color, linestyle in (
        ("mean_center16_overlap_fraction", "Center-16 overlap", "#d95f02", "-"),
        ("normalized_selection_entropy", "Normalized selection entropy", "#1b9e77", "--"),
        ("mean_actual_shuffled_overlap_fraction", "Actual/shuffled overlap", "#7b3294", ":"),
    ):
        arrays = [
            diagnostic_values(diagnostics[int(step)], ("diagnostics", metric))
            for step in steps
        ]
        means = np.array([values.mean() for values in arrays])
        stds = np.array([values.std() for values in arrays])
        structure_axis.plot(steps, means, marker="o", color=color, linestyle=linestyle, label=label)
        structure_axis.fill_between(steps, means - stds, means + stds, color=color, alpha=0.10)
    coverage_axis.set_title("Dynamic and static coverage controls")
    coverage_axis.set_ylabel("Validation macro-source K=16 coverage")
    structure_axis.set_title("Spatial concentration and content dependence")
    structure_axis.set_ylabel("Fraction / normalized entropy")
    for axis in (coverage_axis, structure_axis):
        axis.set_xlabel("Cumulative optimizer updates")
        axis.set_xticks(steps)
        axis.legend(frameon=False, fontsize=8)
        style_axis(axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_per_video_delta(
    diagnostics: dict[int, list[dict]],
    baselines: dict,
    output: Path,
) -> None:
    final_reports = diagnostics[max(diagnostics)]
    prior = baselines["splits"]["val"]["prior"]["16"]["per_source"]
    sources = list(prior)
    deltas_by_source = {}
    for source in sources:
        prior_videos = prior[source]["video_coverage"]
        deltas = []
        for video_id, prior_value in prior_videos.items():
            learned = np.mean(
                [
                    report["coverage"]["actual"]["16"]["per_source"][source][
                        "video_coverage"
                    ][video_id]
                    for report in final_reports
                ]
            )
            deltas.append(float(learned - prior_value))
        deltas_by_source[source] = np.sort(np.array(deltas))
    max_abs = max(float(np.max(np.abs(values))) for values in deltas_by_source.values())
    x_limit = max(0.05, max_abs * 1.08)
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True, constrained_layout=True)
    for axis, source in zip(axes.flat, sources):
        deltas = deltas_by_source[source]
        probabilities = np.arange(1, len(deltas) + 1) / len(deltas)
        axis.step(deltas, probabilities, where="post", color="#1677b8", linewidth=1.8)
        axis.scatter(deltas, probabilities, color="#1677b8", s=22, zorder=3)
        axis.axvline(0, color="#555555", linewidth=0.9)
        axis.axvline(np.median(deltas), color="#d95f02", linestyle="--", linewidth=1.1)
        axis.set_title(f"{source} (n={len(deltas)})")
        axis.set_xlim(-x_limit, x_limit)
        axis.set_ylim(0, 1.04)
        style_axis(axis)
    for axis in axes[:, 0]:
        axis.set_ylabel("Fraction of validation videos")
    for axis in axes[-1, :]:
        axis.set_xlabel("20k learned coverage − Prior-16 coverage")
    figure.suptitle("Per-video improvement distribution at 20k", fontsize=14)
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
    histories = [
        stitched_history(cfg, args.runs_root, int(base_seed), int(continuation_seed))
        for base_seed, continuation_seed in zip(cfg["base_seeds"], cfg["continuation_seeds"])
    ]
    baselines = read_json(Path(cfg["baseline_metrics"]))
    diagnostics = load_diagnostics(cfg, args.diagnostics_root)
    plot_learning(cfg, histories, args.output_dir / "r2b_fresh20k_validation_macro_k16.png")
    plot_per_source(
        cfg,
        histories,
        baselines,
        args.output_dir / "r2b_fresh20k_validation_by_source.png",
    )
    plot_collapse(
        diagnostics,
        args.output_dir / "r2b_fresh20k_center_collapse_longitudinal.png",
    )
    plot_per_video_delta(
        diagnostics,
        baselines,
        args.output_dir / "r2b_fresh20k_per_video_delta_prior16.png",
    )


if __name__ == "__main__":
    main()
