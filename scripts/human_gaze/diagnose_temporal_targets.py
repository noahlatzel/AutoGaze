# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure temporal structure in aligned STAViS gaze targets."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES, read_manifest
from autogaze.human_gaze.temporal import (
    build_video_series,
    deterministic_nonadjacent_pairs,
    distribution_pair_metrics,
    lagged_pairs,
)


def summarize(values):
    values = np.concatenate(values) if values else np.asarray([], dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "p90": float(np.quantile(values, 0.9)) if values.size else None,
        "p95": float(np.quantile(values, 0.95)) if values.size else None,
    }


def evaluate_split(records, cell_mass, lags, topk):
    series = build_video_series(records, cell_mass)
    output = {"lags": {}, "control": {}}
    raw_adjacent_jsd = defaultdict(list)
    for lag in lags:
        by_source = {}
        for source in STAVIS_SOURCES:
            collected = defaultdict(list)
            for (cur_source, _), video in series.items():
                if cur_source != source:
                    continue
                left, right = lagged_pairs(video, lag)
                if len(left):
                    for name, values in distribution_pair_metrics(left, right, topk).items():
                        collected[name].append(values)
                        if lag == 1 and name == "jsd":
                            raw_adjacent_jsd[source].append(values)
            by_source[source] = {name: summarize(values) for name, values in collected.items()}
            if by_source[source].get("centroid_cells", {}).get("mean") is not None:
                by_source[source]["centroid_cells_per_second"] = {
                    **by_source[source]["centroid_cells"],
                    "mean": by_source[source]["centroid_cells"]["mean"] * 3.0 / lag,
                    "median": by_source[source]["centroid_cells"]["median"] * 3.0 / lag,
                    "p90": by_source[source]["centroid_cells"]["p90"] * 3.0 / lag,
                    "p95": by_source[source]["centroid_cells"]["p95"] * 3.0 / lag,
                }
        output["lags"][str(lag)] = by_source
    for source in STAVIS_SOURCES:
        collected = defaultdict(list)
        for (cur_source, _), video in series.items():
            if cur_source != source:
                continue
            left, right = deterministic_nonadjacent_pairs(video, minimum_lag=max(lags))
            if len(left):
                for name, values in distribution_pair_metrics(left, right, topk).items():
                    collected[name].append(values)
        output["control"][source] = {name: summarize(values) for name, values in collected.items()}
    return output, raw_adjacent_jsd


def plot_report(report, output_path):
    lags = [int(value) for value in report["config"]["lags"]]
    metrics = (("cosine", "Cosine similarity"), ("jsd", "Jensen–Shannon divergence"),
               ("centroid_cells", "Centroid displacement (cells)"),
               ("top16_jaccard", "GT Top-16 Jaccard"))
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for source in STAVIS_SOURCES:
            values = [report["splits"]["val"]["lags"][str(lag)][source][metric]["mean"] for lag in lags]
            axis.plot(lags, values, marker="o", label=source)
        axis.set_title(title)
        axis.set_xlabel("Lag (3 Hz samples)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-mass", type=Path, required=True)
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--topk", nargs="+", type=int, default=[16, 24, 32])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    cell_mass = np.load(args.cell_mass, mmap_mode="r")
    splits = {}
    adjacent = {}
    for split in ("train", "val"):
        splits[split], adjacent[split] = evaluate_split(
            read_manifest(args.manifest, split), cell_mass, args.lags, args.topk
        )
    train_jsd = np.concatenate([value for values in adjacent["train"].values() for value in values])
    threshold = float(np.quantile(train_jsd, 0.95))
    abrupt = {}
    for split in ("train", "val"):
        abrupt[split] = {
            source: {
                "count": int(sum(len(value) for value in adjacent[split][source])),
                "fraction_above_train_p95": float(
                    np.mean(np.concatenate(adjacent[split][source]) >= threshold)
                ),
            }
            for source in STAVIS_SOURCES
        }
    report = {
        "schema_version": 1,
        "config": {"lags": args.lags, "topk": args.topk, "target_fps": 3.0},
        "abrupt_transition": {"definition": "adjacent JSD >= train pooled p95", "train_p95_jsd": threshold, "splits": abrupt},
        "splits": splits,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plot_report(report, args.output_plot)
    print(json.dumps({"train_p95_jsd": threshold, "val_abrupt": abrupt["val"]}, sort_keys=True))


if __name__ == "__main__":
    main()
