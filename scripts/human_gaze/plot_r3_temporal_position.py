# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot and summarize the fixed paired R3 temporal-position experiment."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES


SEEDS = (440826, 440827, 440828)
TRAINING_SEED_OFFSET = 200000


def run_name(arm, seed):
    return f"r3b_temporal_{arm}_base{seed}_seed{seed + TRAINING_SEED_OFFSET}"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_runs(root):
    runs = {}
    for arm in ("control", "position"):
        runs[arm] = {}
        for seed in SEEDS:
            directory = root / run_name(arm, seed)
            validation = read_jsonl(directory / "validation_metrics.jsonl")
            training = read_jsonl(directory / "training_metrics.jsonl")
            endpoint = [row for row in validation if int(row["train_step"]) == 10000]
            if len(endpoint) != 1:
                raise ValueError(f"Expected one fixed endpoint in {directory}, found {len(endpoint)}")
            runs[arm][seed] = {
                "validation": validation,
                "training": training,
                "endpoint": endpoint[0],
            }
    return runs


def matched_curves(runs, key):
    step_sets = [
        {int(row["train_step"]) for row in runs[arm][seed]["validation"]}
        for arm in runs
        for seed in SEEDS
    ]
    steps = np.asarray(sorted(set.intersection(*step_sets)), dtype=np.int64)
    curves = {}
    for arm in runs:
        curves[arm] = np.asarray(
            [
                [
                    next(row[key] for row in runs[arm][seed]["validation"] if int(row["train_step"]) == step)
                    for step in steps
                ]
                for seed in SEEDS
            ],
            dtype=np.float64,
        )
    return steps, curves


def plot_learning(runs, output_path):
    steps, curves = matched_curves(runs, "coverage_k16_macro_source")
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    colors = {"control": "#4c78a8", "position": "#e45756"}
    for arm in ("control", "position"):
        for curve in curves[arm]:
            axes[0].plot(steps, curve, color=colors[arm], alpha=0.25, linewidth=1)
        mean = curves[arm].mean(0)
        std = curves[arm].std(0, ddof=1)
        axes[0].plot(steps, mean, color=colors[arm], linewidth=2.3, label=arm.title())
        axes[0].fill_between(steps, mean - std, mean + std, color=colors[arm], alpha=0.14)
    axes[0].axhline(0.4237799161677996, color="#555555", linestyle="--", label="Prior-16")
    axes[0].axhline(0.36891054740813767, color="#999999", linestyle=":", label="Center-16")
    axes[0].set_xlabel("Continuation updates")
    axes[0].set_ylabel("Validation macro-source K16 coverage")
    axes[0].set_title("Paired fixed-step learning curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    paired = curves["position"] - curves["control"]
    for seed, curve in zip(SEEDS, paired):
        axes[1].plot(steps, curve, alpha=0.45, linewidth=1, label=str(seed))
    axes[1].plot(steps, paired.mean(0), color="black", linewidth=2.4, label="Paired mean")
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].axhline(0.005, color="#2ca02c", linestyle="--", linewidth=1, label="Pass effect size")
    axes[1].set_xlabel("Continuation updates")
    axes[1].set_ylabel("Position − control macro K16")
    axes[1].set_title("Matched-seed treatment effect")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return steps, curves


def plot_sources_and_gate(runs, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(STAVIS_SOURCES))
    source_deltas = np.asarray(
        [
            [
                runs["position"][seed]["endpoint"][f"coverage_k16_source_{source}"]
                - runs["control"][seed]["endpoint"][f"coverage_k16_source_{source}"]
                for source in STAVIS_SOURCES
            ]
            for seed in SEEDS
        ]
    )
    axes[0].bar(x, source_deltas.mean(0), color="#e45756", alpha=0.8)
    for seed_index, seed in enumerate(SEEDS):
        axes[0].scatter(x, source_deltas[seed_index], color="black", s=20, alpha=0.65, label=str(seed) if seed_index == 0 else None)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x, STAVIS_SOURCES, rotation=25, ha="right")
    axes[0].set_ylabel("Position − control endpoint coverage")
    axes[0].set_title("Fixed endpoint paired effect by source")
    axes[0].grid(axis="y", alpha=0.25)

    gate_curves = []
    for seed in SEEDS:
        rows = runs["position"][seed]["validation"]
        steps = np.asarray([row["train_step"] for row in rows])
        gate = np.asarray([row["temporal_position_gate"] for row in rows])
        gate_curves.append(gate)
        axes[1].plot(steps, gate, alpha=0.38, linewidth=1, label=str(seed))
    axes[1].plot(steps, np.asarray(gate_curves).mean(0), color="black", linewidth=2.2, label="Mean")
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].set_xlabel("Continuation updates")
    axes[1].set_ylabel("Learned scalar gate (= signal/feature RMS with sign)")
    axes[1].set_title("Norm-controlled temporal contribution")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return source_deltas


def load_center_diagnostics(diagnostics_root):
    return {
        arm: {
            seed: json.loads(
                (diagnostics_root / run_name(arm, seed) / "center_collapse_val.json").read_text()
            )
            for seed in SEEDS
        }
        for arm in ("control", "position")
    }


def plot_controls(center, output_path):
    methods = ("actual", "center", "selection_frequency_top16", "shuffled")
    labels = ("Dynamic", "Center-16", "Static learned Top-16", "Shuffled video")
    colors = ("#4c78a8", "#999999", "#f58518", "#7b3294")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    x = np.arange(2)
    width = 0.19
    for method_index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means = [
            np.mean([center[arm][seed]["coverage"][method]["16"]["macro_source_mean"] for seed in SEEDS])
            for arm in ("control", "position")
        ]
        axes[0].bar(x + (method_index - 1.5) * width, means, width, label=label, color=color)
    axes[0].set_xticks(x, ("Control", "Temporal position"))
    axes[0].set_ylabel("Endpoint macro-source K16 coverage")
    axes[0].set_title("Dynamic and static endpoint controls")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    diagnostic_keys = ("mean_center16_overlap_fraction", "mean_actual_shuffled_overlap_fraction")
    diagnostic_labels = ("Center-16 overlap", "Shuffled-video overlap")
    for index, (key, label) in enumerate(zip(diagnostic_keys, diagnostic_labels)):
        means = [np.mean([center[arm][seed]["diagnostics"][key] for seed in SEEDS]) for arm in ("control", "position")]
        axes[1].bar(x + (index - 0.5) * 0.34, means, 0.34, label=label)
    axes[1].set_xticks(x, ("Control", "Temporal position"))
    axes[1].set_ylabel("Mean selection overlap fraction")
    axes[1].set_title("Center bias and current-video dependence")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def summarize(runs, steps, curves, source_deltas, center):
    endpoint_pairs = curves["position"][:, -1] - curves["control"][:, -1]
    auc_pairs = np.trapezoid(curves["position"] - curves["control"], steps, axis=1) / (steps[-1] - steps[0])
    dynamic_minus_static = {
        arm: [
            center[arm][seed]["diagnostics"]["actual_minus_static_topk_macro"]
            for seed in SEEDS
        ]
        for arm in center
    }
    mean_delta = float(endpoint_pairs.mean())
    positive = int((endpoint_pairs > 0).sum())
    dynamic_positive = all(value > 0 for value in dynamic_minus_static["position"])
    if mean_delta >= 0.005 and positive >= 2 and dynamic_positive:
        decision = "pass"
    elif mean_delta <= 0 and int((endpoint_pairs <= 0).sum()) >= 2:
        decision = "fail"
    else:
        decision = "inconclusive"
    return {
        "schema_version": 1,
        "fixed_endpoint_step": 10000,
        "endpoint_paired_delta": dict(zip(map(str, SEEDS), endpoint_pairs.tolist())),
        "endpoint_mean_paired_delta": mean_delta,
        "positive_seed_pairs": positive,
        "matched_curve_auc_paired_delta": dict(zip(map(str, SEEDS), auc_pairs.tolist())),
        "source_mean_paired_delta": dict(zip(STAVIS_SOURCES, source_deltas.mean(0).tolist())),
        "treatment_final_gate": {
            str(seed): runs["position"][seed]["endpoint"]["temporal_position_gate"]
            for seed in SEEDS
        },
        "treatment_final_signal_to_feature_rms": {
            str(seed): runs["position"][seed]["endpoint"]["temporal_signal_to_feature_rms"]
            for seed in SEEDS
        },
        "dynamic_minus_static": dynamic_minus_static,
        "decision": decision,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("outputs/human_gaze/grpo"))
    parser.add_argument("--diagnostics-root", type=Path, default=Path("outputs/human_gaze/diagnostics"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.run_root)
    steps, curves = plot_learning(runs, args.output_dir / "r3b_temporal_position_learning_curves.png")
    source_deltas = plot_sources_and_gate(runs, args.output_dir / "r3b_temporal_position_sources_and_gate.png")
    center = load_center_diagnostics(args.diagnostics_root)
    plot_controls(center, args.output_dir / "r3b_temporal_position_endpoint_controls.png")
    summary = summarize(runs, steps, curves, source_deltas, center)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
