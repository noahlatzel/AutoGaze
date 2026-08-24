# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit whether exact-K gaze policies use causal visual history."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset, STAVIS_SOURCES
from autogaze.human_gaze.coverage import global_positions_to_fine_cells
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from scripts.human_gaze.diagnose_center_collapse import (
    within_source_different_video_indices,
)


def generate_cells(model, video, budget, allowed, actions_per_frame, fine_offset):
    output = model(
        {"video": video},
        max_gaze_tokens_each_frame=budget,
        allowed_token_ids=allowed,
        generate_only=True,
    )
    return global_positions_to_fine_cells(
        output["gazing_pos"],
        num_frames=video.shape[1],
        exact_budget=budget,
        actions_per_frame=actions_per_frame,
        fine_action_offset=fine_offset,
    ).cpu()


def centroid(cells, grid_size):
    rows = torch.div(cells, grid_size, rounding_mode="floor").to(torch.float64)
    columns = torch.remainder(cells, grid_size).to(torch.float64)
    return torch.stack((columns.mean(-1), rows.mean(-1)), dim=-1)


def target_centroid(cell_mass, grid_size):
    indices = torch.arange(grid_size * grid_size, dtype=torch.float64)
    x = torch.remainder(indices, grid_size)
    y = torch.div(indices, grid_size, rounding_mode="floor")
    mass = cell_mass.to(torch.float64)
    return torch.stack(((mass * x).sum(-1), (mass * y).sum(-1)), dim=-1)


def set_jaccard(left, right):
    return torch.tensor(
        [
            len(set(a.tolist()) & set(b.tolist())) / len(set(a.tolist()) | set(b.tolist()))
            for a, b in zip(left, right)
        ],
        dtype=torch.float64,
    )


def selected_coverage_batch(cell_mass, cells):
    return cell_mass.gather(-1, cells).sum(-1).to(torch.float64)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
    }


def choose_trajectory_examples(dataset, grid_size):
    best = {}
    for index, record in enumerate(dataset.records):
        mass = torch.from_numpy(np.array(dataset.cell_mass[record["cell_mass_index"]], copy=True))
        points = target_centroid(mass, grid_size)
        path_length = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum().item()
        if record["source"] not in best or path_length > best[record["source"]][0]:
            best[record["source"]] = (path_length, index, points.numpy())
    return {source: value[1:] for source, value in best.items()}


def plot_history(report, output_path):
    sources = [
        source
        for source in STAVIS_SOURCES
        if all(seed["per_source"][source] for seed in report["seeds"].values())
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    x = np.arange(len(sources))
    for offset, intervention in ((-0.18, "reset"), (0.18, "alternative")):
        means = []
        errors = []
        for source in sources:
            seed_values = [
                1.0 - seed["per_source"][source][f"normal_vs_{intervention}_jaccard"]["mean"]
                for seed in report["seeds"].values()
            ]
            means.append(np.mean(seed_values))
            errors.append(np.std(seed_values, ddof=1) if len(seed_values) > 1 else 0.0)
        axes[0].bar(
            x + offset,
            means,
            width=0.34,
            yerr=errors,
            capsize=3,
            label=f"{intervention.title()} past",
        )
    axes[0].set_xticks(x, sources, rotation=25, ha="right")
    axes[0].set_ylabel("Selection change (1 − Jaccard)")
    axes[0].set_title("Current frame fixed; causal past intervened")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    gt = []
    predicted = []
    for source in sources:
        gt.append(np.mean([seed["per_source"][source]["gt_centroid_step"]["mean"] for seed in report["seeds"].values()]))
        predicted.append(np.mean([seed["per_source"][source]["normal_centroid_step"]["mean"] for seed in report["seeds"].values()]))
    axes[1].scatter(gt, predicted, s=55)
    limit = max(gt + predicted) * 1.08
    axes[1].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    for source, x_value, y_value in zip(sources, gt, predicted):
        axes[1].annotate(source, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlim(0, limit)
    axes[1].set_ylim(0, limit)
    axes[1].set_xlabel("GT centroid displacement / frame (cells)")
    axes[1].set_ylabel("Policy displacement / frame (cells)")
    axes[1].set_title("Normal policy trajectory dynamics")
    axes[1].grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_trajectory_examples(examples, trajectories, output_path, grid_size):
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=True)
    for axis, source in zip(axes.flat, STAVIS_SOURCES):
        _, gt = examples[source]
        axis.plot(gt[:, 0], gt[:, 1], "-o", color="black", linewidth=2, markersize=3, label="GT")
        for seed, points in trajectories[source].items():
            axis.plot(points[:, 0], points[:, 1], alpha=0.42, linewidth=1, label=str(seed))
        axis.scatter(gt[0, 0], gt[0, 1], color="limegreen", edgecolor="black", s=45, zorder=5)
        axis.scatter(gt[-1, 0], gt[-1, 1], color="red", marker="X", s=50, zorder=5)
        axis.set_title(source)
        axis.set_xlim(-0.5, grid_size - 0.5)
        axis.set_ylim(grid_size - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle("High-motion validation examples: GT and six policy centroid paths\n(start = green, end = red)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--trajectory-plot", type=Path, required=True)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    seeds = args.seeds or [int(value) for value in config["diagnostics"]["k16_seeds"]]
    frame_index = int(config["diagnostics"]["history_audit_frame_index"])
    if frame_index != int(config["dataset"]["clip_len"]) - 1:
        raise ValueError("This audit requires the final frame so all preceding frames are causal history")

    checkpoint_template = config["diagnostics"]["checkpoint_template"]
    first_checkpoint = checkpoint_template.format(seed=seeds[0], continuation_seed=seeds[0] + 100000)
    processor = AutoGazeImageProcessor.from_pretrained(first_checkpoint, local_files_only=True)
    dataset = AVGazeStavisDataset(
        root=config["dataset"]["root"],
        manifest_path=config["dataset"]["manifest"],
        split=config["dataset"]["evaluation_split"],
        load_rgb=True,
        load_heatmap=False,
        cell_mass_path=config["dataset"]["cell_mass"],
        image_processor=processor,
    )
    indices = list(range(len(dataset)))
    if args.max_clips is not None:
        indices = indices[: args.max_clips]
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    shuffled = within_source_different_video_indices(dataset.records)
    alternative_loader = DataLoader(Subset(dataset, [shuffled[index] for index in indices]), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    actions_per_frame = 265
    fine_offset = 69
    budget = int(config["architecture"]["budget"])
    allowed = list(range(fine_offset, actions_per_frame))
    grid_size = int(round(math.sqrt(actions_per_frame - fine_offset)))
    examples = choose_trajectory_examples(dataset, grid_size)
    example_by_index = {value[0]: source for source, value in examples.items()}
    trajectories = {source: {} for source in STAVIS_SOURCES}
    report_seeds = {}

    for seed in seeds:
        checkpoint = checkpoint_template.format(seed=seed, continuation_seed=seed + 100000)
        model = AutoGaze.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
        values = defaultdict(lambda: defaultdict(list))
        processed = 0
        with torch.inference_mode():
            for batch, alternative in zip(loader, alternative_loader):
                video = batch["video"].cuda(non_blocking=True)
                alternative_video = torch.cat((alternative["video"][:, :-1], batch["video"][:, -1:]), dim=1).cuda(non_blocking=True)
                normal = generate_cells(model, video, budget, allowed, actions_per_frame, fine_offset)
                changed = generate_cells(model, alternative_video, budget, allowed, actions_per_frame, fine_offset)
                reset = generate_cells(model, video[:, -1:], budget, allowed, actions_per_frame, fine_offset)
                mass = batch["cell_mass"]
                normal_points = centroid(normal, grid_size)
                gt_points = target_centroid(mass, grid_size)
                for batch_index, source in enumerate(batch["source"]):
                    final_normal = normal[batch_index, -1:]
                    final_changed = changed[batch_index, -1:]
                    final_reset = reset[batch_index]
                    final_mass = mass[batch_index, -1:]
                    values[source]["normal_vs_alternative_jaccard"].extend(set_jaccard(final_normal, final_changed).tolist())
                    values[source]["normal_vs_reset_jaccard"].extend(set_jaccard(final_normal, final_reset).tolist())
                    values[source]["alternative_centroid_shift"].extend(torch.linalg.vector_norm(centroid(final_normal, grid_size) - centroid(final_changed, grid_size), dim=-1).tolist())
                    values[source]["reset_centroid_shift"].extend(torch.linalg.vector_norm(centroid(final_normal, grid_size) - centroid(final_reset, grid_size), dim=-1).tolist())
                    normal_coverage = selected_coverage_batch(final_mass, final_normal)
                    values[source]["alternative_minus_normal_coverage"].extend((selected_coverage_batch(final_mass, final_changed) - normal_coverage).tolist())
                    values[source]["reset_minus_normal_coverage"].extend((selected_coverage_batch(final_mass, final_reset) - normal_coverage).tolist())
                    values[source]["normal_centroid_step"].extend(torch.linalg.vector_norm(normal_points[batch_index, 1:] - normal_points[batch_index, :-1], dim=-1).tolist())
                    values[source]["gt_centroid_step"].extend(torch.linalg.vector_norm(gt_points[batch_index, 1:] - gt_points[batch_index, :-1], dim=-1).tolist())
                    global_index = indices[processed + batch_index]
                    if global_index in example_by_index:
                        trajectories[example_by_index[global_index]][seed] = normal_points[batch_index].numpy()
                processed += video.shape[0]
        report_seeds[str(seed)] = {
            "checkpoint": checkpoint,
            "processed_clips": processed,
            "per_source": {
                source: {metric: summarize(metric_values) for metric, metric_values in values[source].items()}
                for source in STAVIS_SOURCES
            },
        }
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"completed_seed": seed, "processed_clips": processed}), flush=True)

    report = {
        "schema_version": 1,
        "split": config["dataset"]["evaluation_split"],
        "history_intervention_frame_index": frame_index,
        "current_frame_invariant": True,
        "reset_definition": "the current RGB frame alone as a one-frame clip",
        "alternative_definition": "frames 0-14 from a deterministic same-source different-video clip; actual frame 15 retained",
        "seeds": report_seeds,
        "trajectory_examples": {source: dataset.records[value[0]]["clip_id"] for source, value in examples.items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plot_history(report, args.output_plot)
    if args.max_clips is None:
        plot_trajectory_examples(examples, trajectories, args.trajectory_plot, grid_size)


if __name__ == "__main__":
    main()
