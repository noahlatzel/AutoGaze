# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot validation macro coverage for human-gaze GRPO runs."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINES = {
    "Pretrained": 0.1190542392324096,
    "Center-16": 0.368911,
    "Prior-16": 0.423780,
}


def read_history(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*_5ep_seed*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    histories = defaultdict(list)
    for run_dir in sorted(args.runs_root.glob(args.pattern)):
        metrics_path = run_dir / "validation_metrics.jsonl"
        if not metrics_path.exists():
            continue
        records = read_history(metrics_path)
        if not records:
            continue
        arm = "Small KL (0.01)" if "small_kl" in run_dir.name else "No KL"
        match = re.search(r"seed(\d+)", run_dir.name)
        seed = match.group(1) if match else run_dir.name
        histories[arm].append((seed, records))

    if not histories:
        raise ValueError("No validation histories matched")

    colors = {"No KL": "#1677b8", "Small KL (0.01)": "#d95f02"}
    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for arm, runs in histories.items():
        color = colors[arm]
        by_step = defaultdict(list)
        for seed, records in runs:
            steps = [record["train_step"] for record in records]
            values = [record["coverage_k16_macro_source"] for record in records]
            axis.plot(steps, values, color=color, alpha=0.28, linewidth=1.2)
            for step, value in zip(steps, values):
                by_step[step].append(value)
        complete_steps = sorted(step for step, values in by_step.items() if len(values) == len(runs))
        means = np.array([np.mean(by_step[step]) for step in complete_steps])
        stds = np.array([np.std(by_step[step]) for step in complete_steps])
        axis.plot(complete_steps, means, color=color, linewidth=2.5, label=f"{arm} mean")
        axis.fill_between(complete_steps, means - stds, means + stds, color=color, alpha=0.15)

    baseline_styles = ["--", ":", "-."]
    for (label, value), style in zip(BASELINES.items(), baseline_styles):
        axis.axhline(value, color="#444444", linestyle=style, linewidth=1.1, label=label)
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Validation macro-source K=16 coverage")
    axis.set_ylim(0.10, 0.46)
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, frameon=False)
    axis.set_title("Direct human-coverage GRPO validation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
