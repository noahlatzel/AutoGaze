#!/usr/bin/env python3
"""Render the preregistered supervised K16 comparison figures from curation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def style(axis) -> None:
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def interval_arrays(rows: list[dict], x_key: str) -> tuple[np.ndarray, ...]:
    x = np.asarray([row[x_key] for row in rows], dtype=np.float64)
    mean = np.asarray([row["coverage"]["mean"] for row in rows])
    low = np.asarray([row["coverage"]["ci90_low"] for row in rows])
    high = np.asarray([row["coverage"]["ci90_high"] for row in rows])
    return x, mean, low, high


def plot_convergence(metrics: dict, output: Path) -> None:
    convergence = metrics["convergence"]
    supervised = convergence["supervised_fixed_checkpoints"]
    rl = convergence["rl_existing_actual_logged_updates"]
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    panels = (
        ("base_clip_presentations", "Base-clip presentations"),
        ("nominal_training_trajectory_action_rows", "Nominal trajectory action rows"),
        ("training_wall_seconds", "Training-process wall time (hours)"),
    )
    for axis, (key, label) in zip(axes, panels):
        for rows, method, color in (
            (rl, "RL K16", "#7b3294"),
            (supervised, "Supervised K16", "#1677b8"),
        ):
            if key == "training_wall_seconds":
                x = np.asarray([row[key]["mean"] / 3600 for row in rows])
                mean = np.asarray([row["coverage"]["mean"] for row in rows])
                low = np.asarray([row["coverage"]["ci90_low"] for row in rows])
                high = np.asarray([row["coverage"]["ci90_high"] for row in rows])
            else:
                x, mean, low, high = interval_arrays(rows, key)
            axis.plot(x, mean, color=color, linewidth=2, marker="o" if method.startswith("Supervised") else None, label=method)
            axis.fill_between(x, low, high, color=color, alpha=0.12)
        axis.set_xlabel(label)
        axis.set_ylabel("Validation macro-source K16 coverage")
        style(axis)
    axes[0].legend(frameon=False)
    figure.suptitle("Held-out validation convergence under matched base-clip exposure")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"supervised_k16_convergence.{suffix}", dpi=180)
    plt.close(figure)


def plot_endpoint(metrics: dict, output: Path) -> None:
    rows = [
        row for row in metrics["per_seed_checkpoint_rows"] if row["cumulative_update"] == 20000
    ]
    by_method = {
        method: sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["base_seed"],
        )
        for method in ("rl", "supervised")
    }
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "coverage_full_macro", "Full validation"),
        (axes[1], "coverage_offcenter_macro", "Human-only off-center quartile"),
    ):
        for rl, sl in zip(by_method["rl"], by_method["supervised"]):
            axis.plot([0, 1], [rl[key], sl[key]], color="#888888", alpha=0.55, marker="o")
        axis.set_xticks([0, 1], ["RL", "Supervised"])
        axis.set_ylabel("Macro-source K16 coverage")
        axis.set_title(title)
        style(axis)
    diagnostics = metrics["endpoint_diagnostics"]["supervised"]
    labels = ["Dynamic", "Static Top16", "Shuffled", "Center16", "Train prior"]
    keys = [
        "coverage_full",
        "coverage_static",
        "coverage_shuffled",
        "coverage_center",
        "coverage_train_prior",
    ]
    means = [diagnostics[key]["mean"] for key in keys]
    errors = [
        [diagnostics[key]["mean"] - diagnostics[key]["ci90_low"] for key in keys],
        [diagnostics[key]["ci90_high"] - diagnostics[key]["mean"] for key in keys],
    ]
    axes[2].bar(np.arange(len(labels)), means, color=("#1677b8", "#d95f02", "#7b3294", "#666666", "#1b9e77"))
    axes[2].errorbar(np.arange(len(labels)), means, yerr=errors, fmt="none", color="black", capsize=3)
    axes[2].set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    axes[2].set_ylabel("Validation macro-source K16 coverage")
    axes[2].set_title("Supervised content-dependence controls")
    style(axes[2])
    figure.suptitle("Fixed 20k endpoint comparison; all six matched seeds")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"supervised_k16_endpoint_controls.{suffix}", dpi=180)
    plt.close(figure)


def plot_agreement(metrics: dict, output: Path) -> None:
    agreement = metrics["agreement"]
    paired = agreement["paired_supervised_vs_rl_summary"]
    rl = agreement["rl_vs_rl_context"]["summary"]
    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    groups = (
        ("intersection_over_k", "Intersection / K", "intersection_over_k_full_macro"),
        ("set_jaccard", "Set Jaccard", "set_jaccard_full_macro"),
    )
    x = np.arange(len(groups))
    for position, (key, _label, rl_key) in enumerate(groups):
        sl = paired[key]["full_validation"]
        axis.errorbar(
            position - 0.12,
            sl["mean"],
            yerr=[[sl["mean"] - sl["ci90_low"]], [sl["ci90_high"] - sl["mean"]]],
            fmt="o",
            color="#1677b8",
            capsize=4,
            label="Paired SL–RL" if position == 0 else None,
        )
        context = rl[rl_key]
        axis.errorbar(
            position + 0.12,
            context["mean"],
            yerr=[[context["mean"] - context["min"]], [context["max"] - context["mean"]]],
            fmt="s",
            color="#7b3294",
            capsize=4,
            label="RL–RL pair range" if position == 0 else None,
        )
    axis.set_xticks(x, [label for _key, label, _rl in groups])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Full-validation source/video-macro agreement")
    axis.set_title("SL–RL agreement in the context of RL seed variability")
    axis.legend(frameon=False)
    style(axis)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"supervised_k16_agreement.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    with args.metrics.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("status") != "complete":
        raise ValueError("Figures require a complete six-seed curation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plot_convergence(metrics, args.output_dir)
    plot_endpoint(metrics, args.output_dir)
    plot_agreement(metrics, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
