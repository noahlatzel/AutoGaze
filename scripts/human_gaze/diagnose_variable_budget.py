# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate and diagnose a fine-only variable-token AutoGaze policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset, STAVIS_SOURCES
from autogaze.human_gaze.coverage import (
    CoverageAccumulator,
    center_order,
    global_positions_to_fine_cells,
)
from autogaze.human_gaze.variable_budget import (
    coverage_for_padded_variable_cells,
    coverage_for_variable_lengths,
    variable_global_positions_to_fine_cells,
)
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor


def source_balanced_calibration_indices(records, per_source: int, seed: int) -> list[int]:
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        key = f"{seed}|{record['clip_id']}".encode("utf-8")
        grouped[record["source"]].append((hashlib.sha256(key).hexdigest(), index))
    indices = []
    for source in STAVIS_SOURCES:
        candidates = sorted(grouped[source])
        if len(candidates) < per_source:
            raise ValueError(f"Calibration source {source} has too few clips")
        indices.extend(index for _, index in candidates[:per_source])
    return indices


def within_source_different_video_indices(records) -> list[int]:
    source_videos = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        source_videos[record["source"]][record["video_id"]].append(index)
    shuffled = [None] * len(records)
    for videos in source_videos.values():
        video_ids = sorted(videos)
        for video_position, video_id in enumerate(video_ids):
            target_id = video_ids[(video_position + 1) % len(video_ids)]
            target = videos[target_id]
            for clip_position, index in enumerate(videos[video_id]):
                shuffled[index] = target[clip_position % len(target)]
    if any(value is None for value in shuffled):
        raise AssertionError("Failed to construct the length shuffle")
    return [int(value) for value in shuffled]


def variable_generate(model, video, cfg):
    return model(
        {"video": video},
        max_gaze_tokens_each_frame=int(cfg["max_budget"]),
        min_gaze_tokens_each_frame=int(cfg["min_budget"]),
        allowed_token_ids=list(
            range(int(cfg["fine_action_offset"]), int(cfg["actions_per_frame"]) + 1)
        ),
        allow_eos=True,
        generate_only=True,
    )


def forced_generate(model, video, cfg, exact_budget: int) -> torch.Tensor:
    gaze = model(
        {"video": video},
        max_gaze_tokens_each_frame=int(exact_budget),
        allowed_token_ids=list(
            range(int(cfg["fine_action_offset"]), int(cfg["actions_per_frame"]))
        ),
        generate_only=True,
    )
    return global_positions_to_fine_cells(
        gaze["gazing_pos"],
        num_frames=int(cfg["clip_len"]),
        exact_budget=int(exact_budget),
        actions_per_frame=int(cfg["actions_per_frame"]),
        fine_action_offset=int(cfg["fine_action_offset"]),
    )


@torch.inference_mode()
def mean_calibration_length(model, loader, cfg) -> float:
    total = 0
    count = 0
    for batch in loader:
        gaze = variable_generate(model, batch["video"].cuda(non_blocking=True), cfg)
        _, lengths, _ = variable_global_positions_to_fine_cells(
            gaze["gazing_pos"],
            gaze["if_padded_gazing"],
            gaze["num_gazing_each_frame"],
            num_frames=int(cfg["clip_len"]),
            max_budget=int(cfg["max_budget"]),
            actions_per_frame=int(cfg["actions_per_frame"]),
            fine_action_offset=int(cfg["fine_action_offset"]),
        )
        total += int(lengths.sum())
        count += lengths.numel()
    return total / count


def calibrate_eos_bias(model, loader, cfg) -> dict:
    decoder = model.gazing_model.gaze_decoder
    eos_id = int(cfg["actions_per_frame"])
    target = float(cfg["target_mean_budget"])
    lower = float(cfg["calibration_bias_min"])
    upper = float(cfg["calibration_bias_max"])
    trials = []

    def evaluate(bias: float) -> float:
        decoder.set_output_token_logit_bias(eos_id, bias)
        mean_length = mean_calibration_length(model, loader, cfg)
        trials.append({"bias": bias, "mean_tokens": mean_length})
        return mean_length

    lower_length = evaluate(lower)
    upper_length = evaluate(upper)
    if lower_length < target or upper_length > target:
        raise ValueError(
            "EOS bias interval does not bracket the target mean: "
            f"{lower_length:.3f} at {lower}, {upper_length:.3f} at {upper}"
        )
    for _ in range(int(cfg["calibration_iterations"])):
        midpoint = (lower + upper) / 2
        midpoint_length = evaluate(midpoint)
        if midpoint_length > target:
            lower = midpoint
        else:
            upper = midpoint
    best = min(trials, key=lambda item: abs(item["mean_tokens"] - target))
    decoder.set_output_token_logit_bias(eos_id, float(best["bias"]))
    return {
        "target_mean_tokens": target,
        "selected_bias": float(best["bias"]),
        "selected_mean_tokens": float(best["mean_tokens"]),
        "trials": trials,
    }


def aggregate_scalar(records, values: torch.Tensor) -> dict:
    videos = defaultdict(lambda: defaultdict(list))
    for record, clip_values in zip(records, values):
        videos[record["source"]][record["video_id"]].append(
            float(clip_values.float().mean())
        )
    per_source = {}
    source_means = []
    for source in STAVIS_SOURCES:
        video_means = {
            video: float(np.mean(clips))
            for video, clips in sorted(videos[source].items())
        }
        source_mean = float(np.mean(list(video_means.values())))
        source_means.append(source_mean)
        per_source[source] = {
            "mean_video_value": source_mean,
            "num_videos": len(video_means),
            "video_values": video_means,
        }
    return {
        "macro_source_mean": float(np.mean(source_means)),
        "pooled_frame_mean": float(values.float().mean()),
        "per_source": per_source,
    }


def oracle_lengths(cell_mass: torch.Tensor, records, cfg) -> torch.Tensor:
    minimum = int(cfg["min_budget"])
    maximum = int(cfg["max_budget"])
    target = int(cfg["target_mean_budget"])
    result = torch.full(cell_mass.shape[:2], minimum, dtype=torch.long)
    for source in STAVIS_SOURCES:
        indices = [i for i, record in enumerate(records) if record["source"] == source]
        masses = cell_mass[indices]
        ordered = torch.sort(masses, dim=-1, descending=True, stable=True).values
        candidates = ordered[..., minimum:maximum]
        extra = (target - minimum) * candidates.shape[0] * candidates.shape[1]
        chosen = torch.zeros_like(candidates, dtype=torch.bool).flatten()
        chosen[torch.topk(candidates.flatten(), extra).indices] = True
        counts = chosen.reshape(candidates.shape).sum(dim=-1)
        result[indices] += counts
    return result


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_np = left.double().flatten().numpy()
    right_np = right.double().flatten().numpy()
    if np.std(left_np) == 0 or np.std(right_np) == 0:
        return float("nan")
    return float(np.corrcoef(left_np, right_np)[0, 1])


def binned_relation(feature: torch.Tensor, lengths: torch.Tensor, bins: int = 5) -> list[dict]:
    order = torch.argsort(feature.flatten(), stable=True)
    chunks = torch.tensor_split(order, bins)
    flat_feature = feature.flatten()
    flat_lengths = lengths.flatten()
    return [
        {
            "feature_mean": float(flat_feature[chunk].mean()),
            "mean_tokens": float(flat_lengths[chunk].float().mean()),
            "num_frames": int(chunk.numel()),
        }
        for chunk in chunks
        if chunk.numel()
    ]


def selection_frequency(
    ranked: torch.Tensor,
    lengths: torch.Tensor,
    num_cells: int,
) -> torch.Tensor:
    counts = torch.zeros(num_cells, dtype=torch.long)
    for cells, length in zip(ranked.reshape(-1, ranked.shape[-1]), lengths.flatten()):
        selected = cells[: int(length)]
        counts.scatter_add_(0, selected, torch.ones_like(selected))
    return counts.double() / lengths.numel()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    batch_size = args.batch_size or int(cfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["num_workers"])

    processor = AutoGazeImageProcessor.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    model = AutoGaze.from_pretrained(
        args.checkpoint, local_files_only=True
    ).cuda().eval()

    train_dataset = AVGazeStavisDataset(
        root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        split="train",
        load_rgb=True,
        load_heatmap=False,
        cell_mass_path=cfg["cell_mass_path"],
        image_processor=processor,
    )
    calibration_indices = source_balanced_calibration_indices(
        train_dataset.records,
        int(cfg["calibration_clips_per_source"]),
        int(cfg["calibration_seed"]),
    )
    calibration_loader = DataLoader(
        Subset(train_dataset, calibration_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    if args.skip_calibration:
        calibration = {
            "target_mean_tokens": float(cfg["target_mean_budget"]),
            "selected_bias": float(
                model.gazing_model.gaze_decoder.output_token_logit_bias[
                    int(cfg["actions_per_frame"])
                ]
            ),
            "selected_mean_tokens": mean_calibration_length(
                model, calibration_loader, cfg
            ),
            "trials": [],
        }
    else:
        calibration = calibrate_eos_bias(model, calibration_loader, cfg)

    dataset = AVGazeStavisDataset(
        root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        split="val",
        load_rgb=True,
        load_heatmap=False,
        cell_mass_path=cfg["cell_mass_path"],
        image_processor=processor,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    masses = []
    variable_rankings = []
    forced_k16_rankings = []
    forced_k36_rankings = []
    lengths = []
    eos_flags = []
    prefix_match_count = 0
    prefix_comparison_count = 0
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].cuda(non_blocking=True)
            variable = variable_generate(model, video, cfg)
            variable_cells, variable_lengths, emitted_eos = (
                variable_global_positions_to_fine_cells(
                    variable["gazing_pos"],
                    variable["if_padded_gazing"],
                    variable["num_gazing_each_frame"],
                    num_frames=int(cfg["clip_len"]),
                    max_budget=int(cfg["max_budget"]),
                    actions_per_frame=int(cfg["actions_per_frame"]),
                    fine_action_offset=int(cfg["fine_action_offset"]),
                )
            )
            forced_k16 = forced_generate(
                model, video, cfg, int(cfg["target_mean_budget"])
            )
            forced_k36 = forced_generate(model, video, cfg, int(cfg["max_budget"]))
            prefix_matches = []
            for cells, ordered, frame_lengths in zip(
                variable_cells, forced_k36, variable_lengths
            ):
                for selected, ranking, length in zip(cells, ordered, frame_lengths):
                    prefix_matches.append(
                        torch.equal(selected[: int(length)], ranking[: int(length)])
                    )
            masses.append(batch["cell_mass"].cpu())
            variable_rankings.append(variable_cells.cpu())
            forced_k16_rankings.append(forced_k16.cpu())
            forced_k36_rankings.append(forced_k36.cpu())
            lengths.append(variable_lengths.cpu())
            eos_flags.append(emitted_eos.cpu())

            prefix_match_count += sum(prefix_matches)
            prefix_comparison_count += len(prefix_matches)

    cell_mass = torch.cat(masses)
    variable_cells = torch.cat(variable_rankings)
    forced_k16_cells = torch.cat(forced_k16_rankings)
    forced_k36_cells = torch.cat(forced_k36_rankings)
    actual_lengths = torch.cat(lengths)
    emitted_eos = torch.cat(eos_flags)
    shuffled_indices = within_source_different_video_indices(dataset.records)
    shuffled_lengths = actual_lengths[shuffled_indices]
    fixed_lengths = torch.full_like(actual_lengths, int(cfg["target_mean_budget"]))
    oracle_k = oracle_lengths(cell_mass, dataset.records, cfg)
    oracle_ranking = torch.argsort(cell_mass, dim=-1, descending=True, stable=True)[
        ..., : int(cfg["max_budget"])
    ]

    frame_values = {
        "variable": coverage_for_padded_variable_cells(
            cell_mass, variable_cells, actual_lengths
        ),
        "forced_k16": coverage_for_variable_lengths(
            cell_mass, forced_k16_cells, fixed_lengths
        ),
        "actual_k_forced_order": coverage_for_variable_lengths(
            cell_mass, forced_k36_cells, actual_lengths
        ),
        "same_source_shuffled_k": coverage_for_variable_lengths(
            cell_mass, forced_k36_cells, shuffled_lengths
        ),
        "oracle_variable": coverage_for_variable_lengths(
            cell_mass, oracle_ranking, oracle_k
        ),
    }
    accumulator = CoverageAccumulator()
    for method, values in frame_values.items():
        for record, clip_values in zip(dataset.records, values):
            accumulator.add(
                method,
                16,
                record["source"],
                record["video_id"],
                clip_values,
            )
    coverage = accumulator.finalize(tuple(frame_values), [16])
    length_metrics = {
        "variable": aggregate_scalar(dataset.records, actual_lengths),
        "same_source_shuffled_k": aggregate_scalar(
            dataset.records, shuffled_lengths
        ),
        "oracle_variable": aggregate_scalar(dataset.records, oracle_k),
    }

    grid_size = int(round(math.sqrt(cell_mass.shape[-1])))
    center16 = center_order(grid_size)[:16]
    center_mass = cell_mass[..., center16].sum(dim=-1)
    normalized_entropy = -(
        cell_mass.clamp_min(1e-12) * cell_mass.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(cell_mass.shape[-1])
    selected_mass = cell_mass.gather(2, forced_k36_cells)
    length_groups = {
        "short_4_12": actual_lengths <= 12,
        "medium_13_20": (actual_lengths >= 13) & (actual_lengths <= 20),
        "long_21_36": actual_lengths >= 21,
    }
    marginal = {
        "all": selected_mass.mean(dim=(0, 1)).tolist(),
    }
    group_frequency = {}
    for name, mask in length_groups.items():
        marginal[name] = (
            selected_mass[mask].mean(dim=0).tolist() if mask.any() else []
        )
        group_frequency[name] = (
            selection_frequency(
                variable_cells[mask].reshape(1, -1, variable_cells.shape[-1]),
                actual_lengths[mask].reshape(1, -1),
                cell_mass.shape[-1],
            ).tolist()
            if mask.any()
            else [0.0] * cell_mass.shape[-1]
        )

    frequency = selection_frequency(
        variable_cells, actual_lengths, cell_mass.shape[-1]
    )
    center_membership = torch.zeros(cell_mass.shape[-1], dtype=torch.bool)
    center_membership[center16] = True
    active = torch.arange(int(cfg["max_budget"])).reshape(1, 1, -1) < actual_lengths.unsqueeze(-1)
    center_overlap = (
        (center_membership[variable_cells.clamp(min=0)] & active).sum(dim=-1)
        / actual_lengths.clamp(min=1)
    ).float()
    histogram = Counter(int(value) for value in actual_lengths.flatten())
    report = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint),
        "split": "val",
        "processed_clips": len(dataset),
        "calibration": {
            **calibration,
            "split": "train_source_balanced_fixed_subset",
            "clips_per_source": int(cfg["calibration_clips_per_source"]),
            "seed": int(cfg["calibration_seed"]),
            "clip_ids": [
                train_dataset.records[index]["clip_id"]
                for index in calibration_indices
            ],
        },
        "coverage": coverage,
        "length": length_metrics,
        "diagnostics": {
            "length_histogram": {
                str(key): value for key, value in sorted(histogram.items())
            },
            "length_std_frames": float(actual_lengths.float().std(unbiased=False)),
            "eos_rate": float(emitted_eos.float().mean()),
            "min_rate": float(
                (actual_lengths == int(cfg["min_budget"])).float().mean()
            ),
            "cap_rate": float(
                (actual_lengths == int(cfg["max_budget"])).float().mean()
            ),
            "mean_center16_overlap_fraction": float(center_overlap.mean()),
            "variable_forced_k36_prefix_match_rate": (
                prefix_match_count / prefix_comparison_count
            ),
            "length_entropy_correlation": correlation(
                actual_lengths, normalized_entropy
            ),
            "length_center_mass_correlation": correlation(
                actual_lengths, center_mass
            ),
            "selection_frequency_per_frame": frequency.tolist(),
            "selection_frequency_by_length_group": group_frequency,
            "center16_cells": center16.tolist(),
        },
        "relationships": {
            "entropy_bins": binned_relation(
                normalized_entropy, actual_lengths
            ),
            "center_mass_bins": binned_relation(center_mass, actual_lengths),
        },
        "marginal_coverage_gain": marginal,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 10), constrained_layout=True)
    image = axes[0, 0].imshow(
        frequency.reshape(grid_size, grid_size), cmap="magma", vmin=0, vmax=1
    )
    for cell in center16.tolist():
        row, column = divmod(cell, grid_size)
        axes[0, 0].add_patch(
            Rectangle(
                (column - 0.5, row - 0.5), 1, 1,
                fill=False, edgecolor="cyan", linewidth=1.0,
            )
        )
    axes[0, 0].set_title("Variable-K selection frequency (Center-16 outlined)")
    axes[0, 0].set_xlabel("Fine-grid column")
    axes[0, 0].set_ylabel("Fine-grid row")
    figure.colorbar(image, ax=axes[0, 0], label="Fraction of validation frames")

    xs = np.arange(int(cfg["min_budget"]), int(cfg["max_budget"]) + 1)
    axes[0, 1].bar(xs, [histogram.get(int(value), 0) for value in xs], color="#1677b8")
    axes[0, 1].axvline(16, color="black", linestyle="--", label="Target mean K16")
    axes[0, 1].set_title("Validation frame-length distribution")
    axes[0, 1].set_xlabel("Selected fine tokens")
    axes[0, 1].set_ylabel("Frames")
    axes[0, 1].legend(frameon=False)

    methods = (
        ("variable", "Variable K", "#1677b8"),
        ("forced_k16", "Forced K16", "#666666"),
        ("actual_k_forced_order", "Actual K on forced order", "#e6ab02"),
        ("same_source_shuffled_k", "Shuffled K", "#7b3294"),
        ("oracle_variable", "Oracle variable", "#1b9e77"),
    )
    x = np.arange(len(STAVIS_SOURCES))
    width = 0.16
    for method_index, (method, label, color) in enumerate(methods):
        values = [
            coverage[method]["16"]["per_source"][source][
                "mean_video_coverage"
            ]
            for source in STAVIS_SOURCES
        ]
        axes[1, 0].bar(
            x + (method_index - 2) * width,
            values,
            width,
            label=label,
            color=color,
        )
    axes[1, 0].set_xticks(x, STAVIS_SOURCES, rotation=25, ha="right")
    axes[1, 0].set_ylabel("Validation mean-video coverage")
    axes[1, 0].set_title("Adaptive-length controls by source")
    axes[1, 0].legend(frameon=False, fontsize=8)

    positions = np.arange(1, int(cfg["max_budget"]) + 1)
    axes[1, 1].plot(positions, marginal["all"], color="black", linewidth=2, label="All frames")
    for name, color in (
        ("short_4_12", "#d95f02"),
        ("medium_13_20", "#7570b3"),
        ("long_21_36", "#1b9e77"),
    ):
        if marginal[name]:
            axes[1, 1].plot(positions, marginal[name], label=name.replace("_", " "), color=color)
    axes[1, 1].axvline(16, color="#777777", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Fine-token position")
    axes[1, 1].set_ylabel("Mean marginal gaze mass")
    axes[1, 1].set_title("Marginal coverage along forced-K36 ordering")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_plot, dpi=180)
    plt.close(figure)

    summary = {
        "selected_bias": calibration["selected_bias"],
        "train_calibration_mean_k": calibration["selected_mean_tokens"],
        "val_mean_k": length_metrics["variable"]["macro_source_mean"],
        "variable_macro": coverage["variable"]["16"]["macro_source_mean"],
        "forced_k16_macro": coverage["forced_k16"]["16"]["macro_source_mean"],
        "actual_k_forced_order_macro": coverage[
            "actual_k_forced_order"
        ]["16"]["macro_source_mean"],
        "shuffled_k_macro": coverage["same_source_shuffled_k"]["16"]["macro_source_mean"],
        "oracle_variable_macro": coverage["oracle_variable"]["16"]["macro_source_mean"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
