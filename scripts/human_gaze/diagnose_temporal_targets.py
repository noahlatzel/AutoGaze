# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure temporal structure in aligned STAViS gaze targets."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def evaluate_image_change(records, cell_mass, dataset_root, image_size):
    """Pair adjacent gaze JSD with a read-only low-resolution RGB-change proxy."""
    videos = defaultdict(list)
    for record in records:
        videos[(record["source"], record["video_id"])].append(record)
    output = defaultdict(lambda: {"rgb_change": [], "jsd": []})
    for (source, video_id), video_records in videos.items():
        previous_image = None
        previous_mass = None
        previous_index = None
        for record in sorted(video_records, key=lambda value: value["target_start_index"]):
            mass = np.asarray(cell_mass[record["cell_mass_index"]], dtype=np.float64)
            for offset, frame_number in enumerate(record["frame_numbers"]):
                target_index = int(record["target_start_index"]) + offset
                image_path = (
                    dataset_root
                    / "video_frames"
                    / source
                    / video_id
                    / f"img_{int(frame_number):05d}.jpg"
                )
                with Image.open(image_path) as image:
                    current_image = np.asarray(
                        image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR),
                        dtype=np.float32,
                    ) / 255.0
                current_mass = mass[offset]
                if previous_index is not None and target_index == previous_index + 1:
                    output[source]["rgb_change"].append(float(np.abs(current_image - previous_image).mean()))
                    output[source]["jsd"].append(float(distribution_pair_metrics(previous_mass[None], current_mass[None], [16])["jsd"][0]))
                previous_image = current_image
                previous_mass = current_mass
                previous_index = target_index
    return output


def summarize_image_change(train_values, val_values, gaze_jsd_threshold):
    pooled_train = np.concatenate(
        [np.asarray(values["rgb_change"], dtype=np.float64) for values in train_values.values()]
    )
    threshold = float(np.quantile(pooled_train, 0.95))
    sources = {}
    for source in STAVIS_SOURCES:
        rgb = np.asarray(val_values[source]["rgb_change"], dtype=np.float64)
        jsd = np.asarray(val_values[source]["jsd"], dtype=np.float64)
        high = rgb >= threshold
        correlation = float(np.corrcoef(_rank(rgb), _rank(jsd))[0, 1])
        sources[source] = {
            "count": int(len(rgb)),
            "spearman_rgb_change_vs_jsd": correlation,
            "high_change_fraction": float(high.mean()),
            "mean_jsd_high_change": float(jsd[high].mean()) if high.any() else None,
            "mean_jsd_other": float(jsd[~high].mean()) if (~high).any() else None,
            "abrupt_gaze_fraction_high_change": float((jsd[high] >= gaze_jsd_threshold).mean()) if high.any() else None,
            "abrupt_gaze_fraction_other": float((jsd[~high] >= gaze_jsd_threshold).mean()) if (~high).any() else None,
        }
    return {"definition": "mean absolute RGB change after 64x64 bilinear resize", "train_p95_rgb_change": threshold, "val": sources}


def plot_image_change(report, output_path):
    sources = list(STAVIS_SOURCES)
    values = report["image_change"]["val"]
    x = np.arange(len(sources))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    width = 0.36
    other = [values[source]["mean_jsd_other"] for source in sources]
    high = [
        np.nan
        if values[source]["mean_jsd_high_change"] is None
        else values[source]["mean_jsd_high_change"]
        for source in sources
    ]
    axes[0].bar(x - width / 2, other, width, label="Other adjacent frames")
    axes[0].bar(x + width / 2, high, width, label="RGB change ≥ train p95")
    axes[0].set_xticks(x, sources, rotation=25, ha="right")
    axes[0].set_ylabel("Adjacent gaze-map JSD")
    axes[0].set_title("Gaze change at high visual-change transitions")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].bar(x, [values[source]["spearman_rgb_change_vs_jsd"] for source in sources])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, sources, rotation=25, ha="right")
    axes[1].set_ylabel("Spearman correlation")
    axes[1].set_title("RGB-change vs gaze-change association")
    axes[1].grid(axis="y", alpha=0.25)
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
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--image-change-plot", type=Path)
    args = parser.parse_args()
    cell_mass = np.load(args.cell_mass, mmap_mode="r")
    splits = {}
    adjacent = {}
    records_by_split = {}
    for split in ("train", "val"):
        records_by_split[split] = read_manifest(args.manifest, split)
        splits[split], adjacent[split] = evaluate_split(
            records_by_split[split], cell_mass, args.lags, args.topk
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
    if args.dataset_root is not None:
        image_change = {
            split: evaluate_image_change(records_by_split[split], cell_mass, args.dataset_root, 64)
            for split in ("train", "val")
        }
        report["image_change"] = summarize_image_change(
            image_change["train"], image_change["val"], threshold
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plot_report(report, args.output_plot)
    if args.image_change_plot is not None:
        if "image_change" not in report:
            raise ValueError("--image-change-plot requires --dataset-root")
        plot_image_change(report, args.image_change_plot)
    print(json.dumps({"train_p95_jsd": threshold, "val_abrupt": abrupt["val"]}, sort_keys=True))


if __name__ == "__main__":
    main()
