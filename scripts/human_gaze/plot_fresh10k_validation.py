# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the stitched six-seed validation trajectory for the fresh 10k gate."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


def read_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stitched_history(stage1: Path, stage2: Path, offset: int) -> list[tuple[int, float]]:
    first = [
        (int(record["train_step"]), float(record["coverage_k16_macro_source"]))
        for record in read_history(stage1 / "validation_metrics.jsonl")
    ]
    second = [
        (
            offset + int(record["train_step"]),
            float(record["coverage_k16_macro_source"]),
        )
        for record in read_history(stage2 / "validation_metrics.jsonl")
        if int(record["train_step"]) > 0
    ]
    return first + second


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    histories = []
    for base_seed, continuation_seed in zip(
        cfg["base_seeds"], cfg["continuation_seeds"]
    ):
        stage1 = args.runs_root / cfg["stage1_run_template"].format(
            base_seed=base_seed
        )
        stage2 = args.runs_root / cfg["stage2_run_template"].format(
            base_seed=base_seed,
            continuation_seed=continuation_seed,
        )
        histories.append(
            (
                str(base_seed),
                stitched_history(stage1, stage2, int(cfg["stage1_updates"])),
            )
        )

    by_step = defaultdict(list)
    figure, axis = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    for seed, history in histories:
        steps = [step for step, _ in history]
        values = [value for _, value in history]
        axis.plot(steps, values, linewidth=1.15, alpha=0.38, label=f"Seed {seed}")
        for step, value in history:
            by_step[step].append(value)

    complete_steps = sorted(
        step for step, values in by_step.items() if len(values) == len(histories)
    )
    means = np.array([np.mean(by_step[step]) for step in complete_steps])
    stds = np.array([np.std(by_step[step]) for step in complete_steps])
    axis.plot(complete_steps, means, color="black", linewidth=2.6, label="Six-seed mean")
    axis.fill_between(
        complete_steps,
        means - stds,
        means + stds,
        color="black",
        alpha=0.13,
        label="Population SD",
    )

    styles = (("pretrained", "Pretrained", "--"), ("center16", "Center-16", ":"), ("prior16", "Prior-16", "-."))
    for key, label, linestyle in styles:
        axis.axhline(
            float(cfg["baselines"][key]),
            color="#555555",
            linestyle=linestyle,
            linewidth=1.1,
            label=label,
        )
    axis.axvline(
        int(cfg["stage1_updates"]),
        color="#888888",
        linestyle="--",
        linewidth=0.9,
    )
    axis.text(
        int(cfg["stage1_updates"]) + 80,
        0.13,
        "optimizer reset; LR = 3e-6",
        color="#666666",
        fontsize=9,
    )
    axis.set_xlim(0, int(cfg["total_updates"]))
    axis.set_ylim(0.10, 0.50)
    axis.set_xlabel("Cumulative optimizer updates")
    axis.set_ylabel("Validation macro-source K=16 coverage")
    axis.set_title("Fresh decoder-only direct-coverage GRPO (six seeds)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, frameon=False, fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
