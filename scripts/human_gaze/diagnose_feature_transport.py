# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure what the R5 correspondence prior transports on validation clips."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset, STAVIS_SOURCES
from autogaze.human_gaze.coverage import global_positions_to_fine_cells
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from scripts.human_gaze.diagnose_temporal_policy_history import target_centroid

METHODS = ("actual", "same_coordinate", "feature_match_top16", "learned_bias_top16")


def video_source_macro(records):
    source_means = {}
    for source in STAVIS_SOURCES:
        videos = defaultdict(list)
        for record in records:
            if record["source"] == source:
                videos[record["video_id"]].append(record["value"])
        source_means[source] = float(
            np.mean([np.mean(values) for values in videos.values()])
        )
    return {
        "macro_source_mean": float(np.mean(list(source_means.values()))),
        "per_source": source_means,
    }


def summarize(values):
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def appearance_features(gaze_model, video):
    batch, frames = video.shape[:2]
    resized = rearrange(video, "b t c h w -> (b t) c h w")
    resized = F.interpolate(
        resized,
        size=(gaze_model.input_img_size, gaze_model.input_img_size),
        mode="bicubic",
        align_corners=False,
    )
    resized = rearrange(resized, "(b t) c h w -> b t c h w", b=batch, t=frames)
    embeddings, _, _, _, _ = gaze_model.embed(video=resized)
    return torch.stack(
        [frame - gaze_model.connector.pos_embed[None] for frame in embeddings],
        dim=1,
    )


def evaluate_seed(seed, checkpoint, cfg, loader):
    model = AutoGaze.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    gaze_model = model.gazing_model
    module = gaze_model.feature_transport_bias
    fine_offset = int(cfg["architecture"]["fine_action_offset"])
    actions_per_frame = int(cfg["architecture"]["actions_per_frame"])
    budget = int(cfg["architecture"]["budget"])
    allowed = list(range(fine_offset, actions_per_frame))
    method_records = {method: [] for method in METHODS}
    transitions = []

    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].cuda(non_blocking=True)
            gaze = model(
                {"video": video},
                max_gaze_tokens_each_frame=budget,
                allowed_token_ids=allowed,
                generate_only=True,
            )
            actual = global_positions_to_fine_cells(
                gaze["gazing_pos"],
                num_frames=int(cfg["dataset"]["clip_len"]),
                exact_budget=budget,
                actions_per_frame=actions_per_frame,
                fine_action_offset=fine_offset,
            )
            appearance = appearance_features(gaze_model, video)
            mass = batch["cell_mass"].to(video.device)
            target = torch.stack(
                [
                    target_centroid(item.cpu(), int(cfg["architecture"]["grid_size"]))
                    for item in mass
                ]
            ).to(video.device)
            target_motion = torch.linalg.vector_norm(target[:, 1:] - target[:, :-1], dim=-1)

            match_frames = []
            learned_frames = []
            for frame in range(1, actual.shape[1]):
                scores = module.normalized_scores(
                    appearance[:, frame - 1],
                    appearance[:, frame],
                    actual[:, frame - 1] + fine_offset,
                    allowed,
                )
                match_frames.append(scores.topk(budget, dim=-1).indices)
                learned_frames.append((scores * module.gate).topk(budget, dim=-1).indices)
            match = torch.stack(match_frames, dim=1)
            learned = torch.stack(learned_frames, dim=1)
            selections = {
                "actual": actual[:, 1:],
                "same_coordinate": actual[:, :-1],
                "feature_match_top16": match,
                "learned_bias_top16": learned,
            }
            coverage = {
                method: mass[:, 1:].gather(-1, cells).sum(-1)
                for method, cells in selections.items()
            }
            for index in range(video.shape[0]):
                source = batch["source"][index]
                video_id = batch["video_id"][index]
                for method in METHODS:
                    method_records[method].append(
                        {
                            "source": source,
                            "video_id": video_id,
                            "value": float(coverage[method][index].mean()),
                        }
                    )
                for frame in range(target_motion.shape[1]):
                    transitions.append(
                        {
                            "motion": float(target_motion[index, frame]),
                            **{
                                method: float(coverage[method][index, frame])
                                for method in METHODS
                            },
                        }
                    )

    motion_threshold = float(np.median([record["motion"] for record in transitions]))
    stratified = {}
    for stratum, predicate in (
        ("low_motion", lambda value: value < motion_threshold),
        ("high_motion", lambda value: value >= motion_threshold),
    ):
        subset = [record for record in transitions if predicate(record["motion"])]
        stratified[stratum] = {
            method: summarize([record[method] for record in subset]) for method in METHODS
        }
    result = {
        "seed": seed,
        "checkpoint": checkpoint,
        "gate": float(module.gate.detach()),
        "clip_count": len(method_records["actual"]),
        "evaluated_frames": "1-15",
        "coverage": {
            method: video_source_macro(method_records[method]) for method in METHODS
        },
        "motion_median_threshold_cells": motion_threshold,
        "motion_stratified_frame_coverage": stratified,
    }
    del model
    torch.cuda.empty_cache()
    return result


def plot_report(report, output):
    seeds = list(report["seeds"])
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(METHODS))
    values = np.asarray(
        [
            [report["seeds"][seed]["coverage"][method]["macro_source_mean"] for method in METHODS]
            for seed in seeds
        ]
    )
    axes[0].bar(x, values.mean(0), color=("#4c78a8", "#999", "#f58518", "#54a24b"))
    for row in values:
        axes[0].scatter(x, row, color="black", s=20, alpha=0.6)
    axes[0].set_xticks(x, ("Policy", "Same coordinates", "Feature match", "Learned bias"), rotation=20, ha="right")
    axes[0].set(ylabel="Validation macro-source K16 coverage", title="What the causal transport prior carries")
    axes[0].grid(axis="y", alpha=0.25)

    strata = ("low_motion", "high_motion")
    delta = np.asarray(
        [
            [
                report["seeds"][seed]["motion_stratified_frame_coverage"][stratum]["feature_match_top16"]["mean"]
                - report["seeds"][seed]["motion_stratified_frame_coverage"][stratum]["same_coordinate"]["mean"]
                for stratum in strata
            ]
            for seed in seeds
        ]
    )
    axes[1].bar(np.arange(2), delta.mean(0), color=("#72b7b2", "#e45756"))
    for row in delta:
        axes[1].scatter(np.arange(2), row, color="black", s=22, alpha=0.65)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(np.arange(2), ("Low target motion", "High target motion"))
    axes[1].set(ylabel="Feature-match − same-coordinate coverage", title="Does correspondence help when gaze moves?")
    axes[1].grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[440826, 440827, 440828])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    template = cfg["diagnostics"]["checkpoint_template"]
    first_checkpoint = template.format(
        seed=args.seeds[0], continuation_seed=args.seeds[0] + 200000
    )
    processor = AutoGazeImageProcessor.from_pretrained(first_checkpoint, local_files_only=True)
    dataset = AVGazeStavisDataset(
        root=cfg["dataset"]["root"],
        manifest_path=cfg["dataset"]["manifest"],
        split=cfg["dataset"]["evaluation_split"],
        load_rgb=True,
        load_heatmap=False,
        cell_mass_path=cfg["dataset"]["cell_mass"],
        image_processor=processor,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    report = {"schema_version": 1, "split": "val", "test_split_opened": False, "seeds": {}}
    for seed in args.seeds:
        checkpoint = template.format(seed=seed, continuation_seed=seed + 200000)
        report["seeds"][str(seed)] = evaluate_seed(seed, checkpoint, cfg, loader)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plot_report(report, args.output_plot)
    print(json.dumps({"output": str(args.output_json), "seeds": args.seeds}, sort_keys=True))


if __name__ == "__main__":
    main()
