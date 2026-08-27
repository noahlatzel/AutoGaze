# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit R6 recurrent-state sensitivity, persistence, and target following."""

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
from torch.utils.data import DataLoader, Subset

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset, STAVIS_SOURCES
from autogaze.human_gaze.coverage import global_positions_to_fine_cells
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from scripts.human_gaze.diagnose_temporal_policy_history import (
    adjacent_jsd,
    centroid,
    selected_coverage_batch,
    set_jaccard,
    summarize,
    target_centroid,
)


def next_same_source_indices(records):
    by_source = defaultdict(list)
    for index, record in enumerate(records):
        by_source[record["source"]].append(index)
    result = [None] * len(records)
    for indices in by_source.values():
        for position, index in enumerate(indices):
            result[index] = indices[(position + 1) % len(indices)]
    return result


def generate(model, video, budget, allowed, recurrent_frame_biases_override=None):
    return model(
        {"video": video},
        max_gaze_tokens_each_frame=budget,
        allowed_token_ids=allowed,
        generate_only=True,
        recurrent_frame_biases_override=recurrent_frame_biases_override,
    )


def local_tokens(gaze, frames, budget, actions_per_frame):
    split = gaze["gazing_pos"].split([budget] * frames, dim=1)
    return [split[frame] - actions_per_frame * frame for frame in range(frames)]


def connector_features(gaze_model, video):
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
    return torch.stack(embeddings, dim=1)


def sequence_biases(model, video, gaze, allowed, frames, budget, actions_per_frame):
    gaze_model = model.gazing_model
    features = connector_features(gaze_model, video)
    biases, states = gaze_model.recurrent_state_bias.sequence_biases(
        features,
        local_tokens(gaze, frames, budget, actions_per_frame),
        allowed,
        gaze_model.gaze_decoder_config.vocab_size,
    )
    return biases, states


def video_source_macro(records):
    per_source = {}
    for source in STAVIS_SOURCES:
        videos = defaultdict(list)
        for record in records:
            if record["source"] == source:
                videos[record["video_id"]].append(record["value"])
        per_source[source] = float(
            np.mean([np.mean(values) for values in videos.values()])
        )
    return {
        "macro_source_mean": float(np.mean(list(per_source.values()))),
        "per_source": per_source,
    }


def evaluate_seed(seed, checkpoint, config, loader, alternative_loader, abrupt_threshold):
    model = AutoGaze.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    module = model.gazing_model.recurrent_state_bias
    frames = int(config["dataset"]["clip_len"])
    budget = int(config["architecture"]["budget"])
    actions_per_frame = int(config["architecture"]["actions_per_frame"])
    fine_offset = int(config["architecture"]["fine_action_offset"])
    grid_size = int(config["architecture"]["grid_size"])
    allowed = list(range(fine_offset, actions_per_frame))
    per_source = defaultdict(lambda: defaultdict(list))
    actual_records = []
    carry_records = []
    transition_records = []
    state_rms = []
    state_saturation = []
    bias_rms = []
    state_absolute_max = 0.0
    processed = 0

    with torch.inference_mode():
        for batch, alternative in zip(loader, alternative_loader):
            video = batch["video"].cuda(non_blocking=True)
            alternative_video = alternative["video"].cuda(non_blocking=True)
            normal_gaze = generate(model, video, budget, allowed)
            alternative_gaze = generate(model, alternative_video, budget, allowed)
            normal_biases, normal_states = sequence_biases(
                model, video, normal_gaze, allowed, frames, budget, actions_per_frame
            )
            alternative_biases, _ = sequence_biases(
                model,
                alternative_video,
                alternative_gaze,
                allowed,
                frames,
                budget,
                actions_per_frame,
            )
            reset_override = normal_biases.clone()
            reset_override[:, -1] = 0
            alternative_override = normal_biases.clone()
            alternative_override[:, -1] = alternative_biases[:, -1]
            reset_gaze = generate(
                model,
                video,
                budget,
                allowed,
                recurrent_frame_biases_override=reset_override,
            )
            changed_gaze = generate(
                model,
                video,
                budget,
                allowed,
                recurrent_frame_biases_override=alternative_override,
            )

            normal = global_positions_to_fine_cells(
                normal_gaze["gazing_pos"], frames, budget, actions_per_frame, fine_offset
            ).cpu()
            reset = global_positions_to_fine_cells(
                reset_gaze["gazing_pos"], frames, budget, actions_per_frame, fine_offset
            ).cpu()
            changed = global_positions_to_fine_cells(
                changed_gaze["gazing_pos"], frames, budget, actions_per_frame, fine_offset
            ).cpu()
            if not torch.equal(normal[:, :-1], reset[:, :-1]):
                raise AssertionError("Reset intervention changed pre-final decoder history")
            if not torch.equal(normal[:, :-1], changed[:, :-1]):
                raise AssertionError("Alternative intervention changed pre-final decoder history")

            mass = batch["cell_mass"]
            normal_points = centroid(normal, grid_size)
            target_points = target_centroid(mass, grid_size)
            target_motion = torch.linalg.vector_norm(
                target_points[:, 1:] - target_points[:, :-1], dim=-1
            )
            policy_velocity = normal_points[:, 1:] - normal_points[:, :-1]
            target_velocity = target_points[:, 1:] - target_points[:, :-1]
            velocity_error = torch.linalg.vector_norm(
                policy_velocity - target_velocity, dim=-1
            )
            adjacent_overlap = torch.stack(
                [set_jaccard(item[:-1], item[1:]) for item in normal]
            )
            normal_coverage = selected_coverage_batch(mass, normal)
            carry_coverage = selected_coverage_batch(mass[:, 1:], normal[:, :-1])
            gaze_jsd = torch.stack([adjacent_jsd(item) for item in mass])
            state_rms.extend(
                normal_states.float().square().mean(dim=-1).sqrt().flatten().cpu().tolist()
            )
            state_absolute_max = max(
                state_absolute_max,
                float(normal_states.float().abs().max()),
            )
            state_saturation.extend(
                (normal_states.float().abs() >= 0.99).float().flatten().cpu().tolist()
            )
            bias_rms.extend(
                normal_biases[:, :, allowed].float().square().mean(dim=-1).sqrt().flatten().cpu().tolist()
            )

            for batch_index, source in enumerate(batch["source"]):
                video_id = batch["video_id"][batch_index]
                final_normal = normal[batch_index, -1:]
                final_reset = reset[batch_index, -1:]
                final_changed = changed[batch_index, -1:]
                final_mass = mass[batch_index, -1:]
                per_source[source]["state_normal_vs_reset_jaccard"].extend(
                    set_jaccard(final_normal, final_reset).tolist()
                )
                per_source[source]["state_normal_vs_alternative_jaccard"].extend(
                    set_jaccard(final_normal, final_changed).tolist()
                )
                final_coverage = selected_coverage_batch(final_mass, final_normal)
                per_source[source]["state_reset_minus_normal_coverage"].extend(
                    (selected_coverage_batch(final_mass, final_reset) - final_coverage).tolist()
                )
                per_source[source]["state_alternative_minus_normal_coverage"].extend(
                    (selected_coverage_batch(final_mass, final_changed) - final_coverage).tolist()
                )
                per_source[source]["policy_centroid_step"].extend(
                    torch.linalg.vector_norm(policy_velocity[batch_index], dim=-1).tolist()
                )
                per_source[source]["target_centroid_step"].extend(
                    target_motion[batch_index].tolist()
                )
                per_source[source]["centroid_velocity_error"].extend(
                    velocity_error[batch_index].tolist()
                )
                abrupt = gaze_jsd[batch_index] >= abrupt_threshold
                per_source[source]["centroid_velocity_error_abrupt"].extend(
                    velocity_error[batch_index][abrupt].tolist()
                )
                per_source[source]["centroid_velocity_error_other"].extend(
                    velocity_error[batch_index][~abrupt].tolist()
                )
                per_source[source]["previous_frame_selection_jaccard"].extend(
                    adjacent_overlap[batch_index].tolist()
                )
                actual_records.append(
                    {
                        "source": source,
                        "video_id": video_id,
                        "value": float(normal_coverage[batch_index, 1:].mean()),
                    }
                )
                carry_records.append(
                    {
                        "source": source,
                        "video_id": video_id,
                        "value": float(carry_coverage[batch_index].mean()),
                    }
                )
                for frame in range(frames - 1):
                    transition_records.append(
                        {
                            "source": source,
                            "motion": float(target_motion[batch_index, frame]),
                            "actual": float(normal_coverage[batch_index, frame + 1]),
                            "carry": float(carry_coverage[batch_index, frame]),
                        }
                    )
            processed += video.shape[0]

    motion_threshold = float(np.median([row["motion"] for row in transition_records]))
    motion_stratified = {}
    for label, predicate in (
        ("low_motion", lambda value: value < motion_threshold),
        ("high_motion", lambda value: value >= motion_threshold),
    ):
        selected = [row for row in transition_records if predicate(row["motion"])]
        motion_stratified[label] = {
            "actual": summarize([row["actual"] for row in selected]),
            "same_coordinate_carryover": summarize([row["carry"] for row in selected]),
        }

    result = {
        "seed": seed,
        "checkpoint": checkpoint,
        "processed_clips": processed,
        "gate": float(module.gate.detach()),
        "state": {
            "rms": summarize(state_rms),
            "absolute_max": state_absolute_max,
            "saturation_fraction": float(np.mean(state_saturation)),
            "allowed_action_bias_rms": summarize(bias_rms),
            "all_finite": bool(
                np.isfinite(state_rms).all()
                and np.isfinite(state_saturation).all()
                and np.isfinite(bias_rms).all()
            ),
        },
        "coverage_frames_1_to_15": {
            "policy": video_source_macro(actual_records),
            "same_coordinate_carryover": video_source_macro(carry_records),
        },
        "motion_median_threshold_cells": motion_threshold,
        "motion_stratified_coverage": motion_stratified,
        "per_source": {
            source: {
                metric: summarize(values)
                for metric, values in per_source[source].items()
            }
            for source in STAVIS_SOURCES
        },
    }
    del model
    torch.cuda.empty_cache()
    return result


def plot_report(report, output):
    seeds = list(report["seeds"])
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    reset_change = []
    alternative_change = []
    for seed in seeds:
        source_rows = report["seeds"][seed]["per_source"]
        reset_change.append(
            np.mean(
                [
                    1 - source_rows[source]["state_normal_vs_reset_jaccard"]["mean"]
                    for source in STAVIS_SOURCES
                ]
            )
        )
        alternative_change.append(
            np.mean(
                [
                    1 - source_rows[source]["state_normal_vs_alternative_jaccard"]["mean"]
                    for source in STAVIS_SOURCES
                ]
            )
        )
    x = np.arange(len(seeds))
    axes[0].bar(x - 0.18, reset_change, 0.36, label="Reset state")
    axes[0].bar(x + 0.18, alternative_change, 0.36, label="Alternative state")
    axes[0].set_xticks(x, seeds)
    axes[0].set(
        xlabel="Base seed",
        ylabel="Final-frame selection change (1 − Jaccard)",
        title="State-only intervention; decoder history fixed",
    )
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    strata = ("low_motion", "high_motion")
    actual = np.asarray(
        [
            [report["seeds"][seed]["motion_stratified_coverage"][key]["actual"]["mean"] for key in strata]
            for seed in seeds
        ]
    )
    carry = np.asarray(
        [
            [report["seeds"][seed]["motion_stratified_coverage"][key]["same_coordinate_carryover"]["mean"] for key in strata]
            for seed in seeds
        ]
    )
    positions = np.arange(2)
    axes[1].bar(positions - 0.18, actual.mean(0), 0.36, label="R6 policy")
    axes[1].bar(positions + 0.18, carry.mean(0), 0.36, label="Same coordinates")
    for row in actual:
        axes[1].scatter(positions - 0.18, row, color="black", s=18, alpha=0.55)
    for row in carry:
        axes[1].scatter(positions + 0.18, row, color="black", s=18, alpha=0.55)
    axes[1].set_xticks(positions, ("Low target motion", "High target motion"))
    axes[1].set(
        ylabel="Validation frame coverage",
        title="Target following versus coordinate persistence",
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[440826, 440827, 440828])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument(
        "--target-metrics",
        type=Path,
        default=Path("outputs/human_gaze/r3a_temporal_diagnostics/target_temporal_metrics.json"),
    )
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    target_metrics = json.loads(args.target_metrics.read_text())
    abrupt_threshold = float(target_metrics["abrupt_transition"]["train_p95_jsd"])
    template = config["diagnostics"]["checkpoint_template"]
    first_checkpoint = template.format(
        seed=args.seeds[0], continuation_seed=args.seeds[0] + 200000
    )
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
    alternative_indices = next_same_source_indices(dataset.records)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    alternative_loader = DataLoader(
        Subset(dataset, alternative_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    report = {
        "schema_version": 1,
        "split": "val",
        "test_split_opened": False,
        "state_intervention_frame": 15,
        "decoder_rgb_and_action_history_fixed_through_frame": 14,
        "alternative_definition": "next validation clip in stable same-source manifest order",
        "abrupt_transition_definition": f"adjacent target JSD >= pooled train p95 ({abrupt_threshold})",
        "seeds": {},
    }
    for seed in args.seeds:
        checkpoint = template.format(seed=seed, continuation_seed=seed + 200000)
        report["seeds"][str(seed)] = evaluate_seed(
            seed,
            checkpoint,
            config,
            loader,
            alternative_loader,
            abrupt_threshold,
        )
        print(json.dumps({"completed_seed": seed}), flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plot_report(report, args.output_plot)


if __name__ == "__main__":
    main()
