# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot and summarize the fixed paired R5 feature-transport experiment."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES

SEEDS = (440826, 440827, 440828)
OFFSET = 200000
T95_DF2 = 4.302652729911275


def name(arm, seed):
    if arm == "control":
        return f"r3b_temporal_control_base{seed}_seed{seed + OFFSET}"
    if arm == "r4_difference":
        return f"r4_causal_difference_base{seed}_seed{seed + OFFSET}"
    return f"r5_feature_transport_base{seed}_seed{seed + OFFSET}"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_runs(root):
    runs = {}
    for arm in ("control", "transport", "r4_difference"):
        runs[arm] = {}
        for seed in SEEDS:
            directory = root / name(arm, seed)
            validation = read_jsonl(directory / "validation_metrics.jsonl")
            endpoint = [row for row in validation if int(row["train_step"]) == 10000]
            if len(endpoint) != 1:
                raise ValueError(f"Expected one fixed endpoint in {directory}")
            runs[arm][seed] = {"validation": validation, "endpoint": endpoint[0]}
    return runs


def matched_curves(runs, key):
    steps = np.asarray(
        sorted(
            set.intersection(
                *[
                    {int(row["train_step"]) for row in runs[arm][seed]["validation"]}
                    for arm in runs
                    for seed in SEEDS
                ]
            )
        )
    )
    values = {}
    for arm in runs:
        values[arm] = np.asarray(
            [
                [
                    next(
                        row[key]
                        for row in runs[arm][seed]["validation"]
                        if int(row["train_step"]) == step
                    )
                    for step in steps
                ]
                for seed in SEEDS
            ],
            dtype=float,
        )
    return steps, values


def paired_summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    half_width = T95_DF2 * standard_error
    return {
        "by_seed": dict(zip(map(str, SEEDS), values.tolist())),
        "mean": mean,
        "standard_error": standard_error,
        "t95_ci": [mean - half_width, mean + half_width],
    }


def plot_learning(steps, curves, output):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    colors = {"control": "#4c78a8", "transport": "#54a24b"}
    for arm in ("control", "transport"):
        for curve in curves[arm]:
            axes[0].plot(steps, curve, color=colors[arm], alpha=0.22, linewidth=1)
        mean = curves[arm].mean(0)
        std = curves[arm].std(0, ddof=1)
        axes[0].plot(steps, mean, color=colors[arm], linewidth=2.3, label=arm.title())
        axes[0].fill_between(steps, mean - std, mean + std, color=colors[arm], alpha=0.13)
    axes[0].axhline(0.4237799161677996, color="#555", linestyle="--", label="Prior-16")
    axes[0].axhline(0.36891054740813767, color="#999", linestyle=":", label="Center-16")
    axes[0].set(xlabel="Continuation updates", ylabel="Validation macro-source K16 coverage", title="R5 fixed-step learning curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    delta = curves["transport"] - curves["control"]
    for seed, line in zip(SEEDS, delta):
        axes[1].plot(steps, line, alpha=0.42, linewidth=1, label=str(seed))
    axes[1].plot(steps, delta.mean(0), color="black", linewidth=2.4, label="Paired mean")
    axes[1].axhline(0, color="#777", linewidth=1)
    axes[1].axhline(0.005, color="#2ca02c", linestyle="--", linewidth=1, label="Pass effect size")
    axes[1].set(xlabel="Continuation updates", ylabel="Transport − control macro K16", title="Matched-seed treatment effect")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_sources_gate(runs, output):
    source_delta = np.asarray(
        [
            [
                runs["transport"][seed]["endpoint"][f"coverage_k16_source_{source}"]
                - runs["control"][seed]["endpoint"][f"coverage_k16_source_{source}"]
                for source in STAVIS_SOURCES
            ]
            for seed in SEEDS
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(STAVIS_SOURCES))
    axes[0].bar(x, source_delta.mean(0), color="#54a24b", alpha=0.82)
    for values in source_delta:
        axes[0].scatter(x, values, color="black", s=19, alpha=0.6)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].axhline(-0.005, color="#d62728", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, STAVIS_SOURCES, rotation=25, ha="right")
    axes[0].set(ylabel="Transport − control endpoint coverage", title="Paired endpoint effect by source")
    axes[0].grid(axis="y", alpha=0.25)

    gates = []
    for seed in SEEDS:
        rows = runs["transport"][seed]["validation"]
        step = np.asarray([row["train_step"] for row in rows])
        gate = np.asarray([row["feature_transport_logit_gate"] for row in rows])
        gates.append(gate)
        axes[1].plot(step, gate, alpha=0.4, linewidth=1, label=str(seed))
    axes[1].plot(step, np.asarray(gates).mean(0), color="black", linewidth=2.3, label="Mean")
    axes[1].axhline(0, color="#777", linewidth=1)
    axes[1].set(xlabel="Continuation updates", ylabel="Learned transport logit gate", title="Direct correspondence contribution")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return source_delta


def load_center(root):
    return {
        arm: {
            seed: json.loads((root / name(arm, seed) / "center_collapse_val.json").read_text())
            for seed in SEEDS
        }
        for arm in ("control", "transport")
    }


def plot_controls(center, output):
    methods = ("actual", "center", "selection_frequency_top16", "shuffled")
    labels = ("Dynamic", "Center-16", "Static learned Top-16", "Shuffled video")
    colors = ("#4c78a8", "#999", "#f58518", "#7b3294")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    x = np.arange(2)
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means = [
            np.mean([center[arm][seed]["coverage"][method]["16"]["macro_source_mean"] for seed in SEEDS])
            for arm in ("control", "transport")
        ]
        axes[0].bar(x + (index - 1.5) * 0.19, means, 0.19, label=label, color=color)
    axes[0].set_xticks(x, ("Control", "Feature transport"))
    axes[0].set(ylabel="Endpoint macro-source K16 coverage", title="Dynamic and static endpoint controls")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    for index, (key, label) in enumerate((
        ("mean_center16_overlap_fraction", "Center-16 overlap"),
        ("mean_actual_shuffled_overlap_fraction", "Shuffled-video overlap"),
    )):
        means = [np.mean([center[arm][seed]["diagnostics"][key] for seed in SEEDS]) for arm in ("control", "transport")]
        axes[1].bar(x + (index - 0.5) * 0.34, means, 0.34, label=label)
    axes[1].set_xticks(x, ("Control", "Feature transport"))
    axes[1].set(ylabel="Mean selection overlap fraction", title="Center bias and current-video dependence")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("outputs/human_gaze/grpo"))
    parser.add_argument("--diagnostics-root", type=Path, default=Path("outputs/human_gaze/diagnostics"))
    parser.add_argument("--transport-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.run_root)
    steps, curves = matched_curves(runs, "coverage_k16_macro_source")
    plot_learning(steps, curves, args.output_dir / "r5_feature_transport_learning.png")
    source_delta = plot_sources_gate(runs, args.output_dir / "r5_feature_transport_sources_gate.png")
    center = load_center(args.diagnostics_root)
    plot_controls(center, args.output_dir / "r5_feature_transport_controls.png")

    endpoint = curves["transport"][:, -1] - curves["control"][:, -1]
    auc = np.trapezoid(curves["transport"] - curves["control"], steps, axis=1) / (steps[-1] - steps[0])
    versus_r4 = curves["transport"][:, -1] - curves["r4_difference"][:, -1]
    dynamic_static = np.asarray([
        center["transport"][seed]["diagnostics"]["actual_minus_static_topk_macro"]
        for seed in SEEDS
    ])
    source_means = source_delta.mean(0)
    gates = np.asarray([
        runs["transport"][seed]["endpoint"]["feature_transport_logit_bias_rms"]
        for seed in SEEDS
    ])
    passed = endpoint.mean() >= 0.005 and (endpoint > 0).sum() >= 2 and (source_means > -0.005).all() and (dynamic_static > 0).all() and np.isfinite(gates).all() and (gates <= 1.0).all()
    failed = endpoint.mean() <= 0 and (endpoint <= 0).sum() >= 2
    decision = "pass" if passed else "fail" if failed else "inconclusive"
    summary = {
        "schema_version": 1,
        "fixed_endpoint_step": 10000,
        "primary_paired_transport_minus_control": paired_summary(endpoint),
        "matched_curve_auc_transport_minus_control": paired_summary(auc),
        "descriptive_transport_minus_r4_endpoint": paired_summary(versus_r4),
        "source_paired_deltas": {
            source: {"by_seed": dict(zip(map(str, SEEDS), source_delta[:, index].tolist())), "mean": float(source_means[index])}
            for index, source in enumerate(STAVIS_SOURCES)
        },
        "dynamic_minus_static_by_seed": dict(zip(map(str, SEEDS), dynamic_static.tolist())),
        "endpoint_absolute_logit_gate_by_seed": dict(zip(map(str, SEEDS), gates.tolist())),
        "transport_prior_diagnostics": json.loads(args.transport_json.read_text()),
        "decision": decision,
        "test_split_opened": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "endpoint_mean": float(endpoint.mean())}, sort_keys=True))


if __name__ == "__main__":
    main()
