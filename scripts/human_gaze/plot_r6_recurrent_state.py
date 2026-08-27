# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot and summarize the fixed paired R6 recurrent-state experiment."""

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
    return f"r6_recurrent_state_base{seed}_seed{seed + OFFSET}"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_runs(root):
    runs = {}
    for arm in ("control", "recurrent"):
        runs[arm] = {}
        for seed in SEEDS:
            directory = root / name(arm, seed)
            validation = read_jsonl(directory / "validation_metrics.jsonl")
            endpoint = [row for row in validation if int(row["train_step"]) == 10000]
            if len(endpoint) != 1:
                raise ValueError(f"Expected one fixed endpoint in {directory}")
            runs[arm][seed] = {
                "validation": validation,
                "training": read_jsonl(directory / "training_metrics.jsonl") if arm == "recurrent" else [],
                "endpoint": endpoint[0],
            }
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
    values = {
        arm: np.asarray(
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
        for arm in runs
    }
    return steps, values


def paired_summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    half_width = T95_DF2 * standard_error
    return {
        "by_seed": dict(zip(map(str, SEEDS), values.tolist())),
        "mean": mean,
        "sample_std": float(values.std(ddof=1)),
        "standard_error": standard_error,
        "t95_ci": [mean - half_width, mean + half_width],
    }


def plot_learning(steps, curves, output):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    colors = {"control": "#4c78a8", "recurrent": "#e45756"}
    for arm in ("control", "recurrent"):
        for curve in curves[arm]:
            axes[0].plot(steps, curve, color=colors[arm], alpha=0.22, linewidth=1)
        mean = curves[arm].mean(0)
        std = curves[arm].std(0, ddof=1)
        axes[0].plot(steps, mean, color=colors[arm], linewidth=2.3, label=arm.title())
        axes[0].fill_between(steps, mean - std, mean + std, color=colors[arm], alpha=0.13)
    axes[0].axhline(0.4237799161677996, color="#555", linestyle="--", label="Prior-16")
    axes[0].axhline(0.36891054740813767, color="#999", linestyle=":", label="Center-16")
    axes[0].set(
        xlabel="Continuation updates",
        ylabel="Validation macro-source K16 coverage",
        title="R6 fixed-step learning curves",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    delta = curves["recurrent"] - curves["control"]
    for seed, line in zip(SEEDS, delta):
        axes[1].plot(steps, line, alpha=0.42, linewidth=1, label=str(seed))
    axes[1].plot(steps, delta.mean(0), color="black", linewidth=2.4, label="Paired mean")
    axes[1].axhline(0, color="#777", linewidth=1)
    axes[1].axhline(0.005, color="#2ca02c", linestyle="--", linewidth=1, label="Pass effect")
    axes[1].set(
        xlabel="Continuation updates",
        ylabel="Recurrent − control macro K16",
        title="Matched-seed treatment effect",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_sources_state(runs, output):
    source_delta = np.asarray(
        [
            [
                runs["recurrent"][seed]["endpoint"][f"coverage_k16_source_{source}"]
                - runs["control"][seed]["endpoint"][f"coverage_k16_source_{source}"]
                for source in STAVIS_SOURCES
            ]
            for seed in SEEDS
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(STAVIS_SOURCES))
    axes[0].bar(x, source_delta.mean(0), color="#e45756", alpha=0.82)
    for values in source_delta:
        axes[0].scatter(x, values, color="black", s=19, alpha=0.6)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].axhline(-0.005, color="#d62728", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, STAVIS_SOURCES, rotation=25, ha="right")
    axes[0].set(
        ylabel="Recurrent − control endpoint coverage",
        title="Paired endpoint effect by source",
    )
    axes[0].grid(axis="y", alpha=0.25)
    gates = []
    for seed in SEEDS:
        rows = runs["recurrent"][seed]["validation"]
        steps = np.asarray([row["train_step"] for row in rows])
        gate = np.asarray([row["recurrent_state_logit_gate"] for row in rows])
        gates.append(gate)
        axes[1].plot(steps, gate, alpha=0.4, linewidth=1, label=str(seed))
    axes[1].plot(steps, np.asarray(gates).mean(0), color="black", linewidth=2.3, label="Mean")
    axes[1].axhline(0, color="#777", linewidth=1)
    axes[1].set(
        xlabel="Continuation updates",
        ylabel="Recurrent logit gate",
        title="Learned trajectory-state contribution",
    )
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
        for arm in ("control", "recurrent")
    }


def plot_controls(center, output):
    methods = ("actual", "center", "selection_frequency_top16", "shuffled")
    labels = ("Dynamic", "Center-16", "Static Top-16", "Shuffled video")
    colors = ("#4c78a8", "#999", "#f58518", "#7b3294")
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    x = np.arange(2)
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means = [
            np.mean(
                [center[arm][seed]["coverage"][method]["16"]["macro_source_mean"] for seed in SEEDS]
            )
            for arm in ("control", "recurrent")
        ]
        axis.bar(x + (index - 1.5) * 0.19, means, 0.19, label=label, color=color)
    axis.set_xticks(x, ("Control", "Recurrent state"))
    axis.set(
        ylabel="Endpoint macro-source K16 coverage",
        title="Dynamic and collapse controls",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def global_history_mean(report, metric):
    return np.mean(
        [
            report["seeds"][str(seed)]["per_source"][source][metric]["mean"]
            for seed in SEEDS
            for source in STAVIS_SOURCES
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("outputs/human_gaze/grpo"))
    parser.add_argument("--diagnostics-root", type=Path, default=Path("outputs/human_gaze/diagnostics"))
    parser.add_argument("--recurrent-json", type=Path, required=True)
    parser.add_argument("--control-history-json", type=Path, required=True)
    parser.add_argument("--verification-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.run_root)
    recurrent = json.loads(args.recurrent_json.read_text())
    control_history = json.loads(args.control_history_json.read_text())
    verification = json.loads(args.verification_json.read_text())
    steps, curves = matched_curves(runs, "coverage_k16_macro_source")
    plot_learning(steps, curves, args.output_dir / "r6_recurrent_state_learning.png")
    source_delta = plot_sources_state(runs, args.output_dir / "r6_recurrent_state_sources_gate.png")
    center = load_center(args.diagnostics_root)
    plot_controls(center, args.output_dir / "r6_recurrent_state_controls.png")

    endpoint = curves["recurrent"][:, -1] - curves["control"][:, -1]
    auc = np.trapezoid(curves["recurrent"] - curves["control"], steps, axis=1) / (
        steps[-1] - steps[0]
    )
    source_means = source_delta.mean(0)
    dynamic_center = np.asarray(
        [center["recurrent"][seed]["diagnostics"]["actual_minus_center_macro"] for seed in SEEDS]
    )
    dynamic_static = np.asarray(
        [center["recurrent"][seed]["diagnostics"]["actual_minus_static_topk_macro"] for seed in SEEDS]
    )
    dynamic_shuffle = np.asarray(
        [center["recurrent"][seed]["diagnostics"]["actual_minus_shuffled_macro"] for seed in SEEDS]
    )
    gates = np.asarray(
        [abs(runs["recurrent"][seed]["endpoint"]["recurrent_state_logit_gate"]) for seed in SEEDS]
    )
    reset_jaccard = global_history_mean(recurrent, "state_normal_vs_reset_jaccard")
    treatment_overlap = global_history_mean(recurrent, "previous_frame_selection_jaccard")
    control_overlap = global_history_mean(control_history, "normal_set_jaccard_step")
    treatment_velocity = global_history_mean(recurrent, "centroid_velocity_error")
    control_velocity = global_history_mean(control_history, "centroid_velocity_error")
    blind_copy = treatment_overlap >= 0.95 or (
        treatment_overlap - control_overlap >= 0.10
        and treatment_velocity >= control_velocity
    )
    inert = bool(np.any(gates < 1e-4) or reset_jaccard >= 0.999)
    stable = verification["status"] == "verified"
    collapsed = bool(
        np.any(dynamic_center <= 0)
        or np.any(dynamic_static <= 0)
        or np.any(dynamic_shuffle <= 0)
    )
    passed = bool(
        endpoint.mean() >= 0.005
        and (endpoint > 0).sum() >= 2
        and (source_means > -0.005).all()
        and not collapsed
        and not blind_copy
        and not inert
        and stable
    )
    failed = bool((endpoint.mean() <= 0 and (endpoint <= 0).sum() >= 2) or collapsed or not stable)
    decision = "pass" if passed else "fail" if failed else "inconclusive"
    summary = {
        "schema_version": 1,
        "fixed_endpoint_step": 10000,
        "primary_paired_recurrent_minus_control": paired_summary(endpoint),
        "matched_curve_auc_recurrent_minus_control": paired_summary(auc),
        "source_paired_deltas": {
            source: {
                "by_seed": dict(zip(map(str, SEEDS), source_delta[:, index].tolist())),
                "mean": float(source_means[index]),
            }
            for index, source in enumerate(STAVIS_SOURCES)
        },
        "collapse_controls": {
            "dynamic_minus_center_by_seed": dict(zip(map(str, SEEDS), dynamic_center.tolist())),
            "dynamic_minus_static_by_seed": dict(zip(map(str, SEEDS), dynamic_static.tolist())),
            "dynamic_minus_shuffle_by_seed": dict(zip(map(str, SEEDS), dynamic_shuffle.tolist())),
            "collapsed": collapsed,
        },
        "recurrent_path": {
            "endpoint_absolute_gate_by_seed": dict(zip(map(str, SEEDS), gates.tolist())),
            "state_normal_reset_jaccard_macro": float(reset_jaccard),
            "inert": inert,
            "stable": stable,
        },
        "trajectory_following": {
            "previous_frame_selection_jaccard": float(treatment_overlap),
            "matched_control_previous_frame_selection_jaccard": float(control_overlap),
            "centroid_velocity_error": float(treatment_velocity),
            "matched_control_centroid_velocity_error": float(control_velocity),
            "blind_copy": blind_copy,
            "recurrent_diagnostics": recurrent,
        },
        "decision": decision,
        "test_split_opened": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "endpoint_mean": float(endpoint.mean())}, sort_keys=True))


if __name__ == "__main__":
    main()
