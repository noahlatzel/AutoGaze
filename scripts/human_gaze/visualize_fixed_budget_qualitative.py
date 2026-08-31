# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize learned fixed-budget gaze tokens on off-center validation clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from PIL import Image

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset, STAVIS_SOURCES
from autogaze.human_gaze.coverage import (
    global_positions_to_fine_cells,
    selected_coverage,
)
from autogaze.human_gaze.qualitative import (
    select_off_center_examples,
    select_off_center_examples_per_source,
)
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor


POLICIES = (
    ("k16", 16, "K16 policy", "#00d4ff"),
    ("k24", 24, "K24 policy", "#72d572"),
    ("k32", 32, "K32 inference (K36-trained)", "#ffb000"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cell-mass", required=True)
    parser.add_argument("--k16-checkpoint", type=Path, required=True)
    parser.add_argument("--k24-checkpoint", type=Path, required=True)
    parser.add_argument("--k32-checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-clips", type=int, default=3)
    parser.add_argument("--clips-per-source", type=int)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--center-budget", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def preprocess_video(video: torch.Tensor, processor, device: torch.device) -> torch.Tensor:
    frames = [
        np.rint(frame.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        for frame in video
    ]
    values = processor(frames, return_tensors="pt").pixel_values
    if values.ndim == 5:
        values = values[0]
    if values.ndim != 4:
        raise ValueError(f"Unexpected processed video shape: {tuple(values.shape)}")
    return values.unsqueeze(0).to(device)


def generate_cells(
    model: AutoGaze,
    video: torch.Tensor,
    budget: int,
    actions_per_frame: int = 265,
    fine_action_offset: int = 69,
) -> torch.Tensor:
    gaze = model(
        {"video": video},
        max_gaze_tokens_each_frame=budget,
        allowed_token_ids=list(range(fine_action_offset, actions_per_frame)),
        generate_only=True,
    )
    if gaze["if_padded_gazing"].any():
        raise AssertionError("Fine-only fixed-budget generation unexpectedly returned padding/EOS")
    expected = torch.full_like(gaze["num_gazing_each_frame"], budget)
    if not torch.equal(gaze["num_gazing_each_frame"], expected):
        raise AssertionError(f"Model did not return exactly {budget} actions per frame")
    return global_positions_to_fine_cells(
        gaze["gazing_pos"],
        num_frames=video.shape[1],
        exact_budget=budget,
        actions_per_frame=actions_per_frame,
        fine_action_offset=fine_action_offset,
    )[0].cpu()


def load_native_frame(
    root: Path,
    source: str,
    video_id: str,
    frame_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB and gaze at native display aspect without changing model input."""
    rgb_path = root / "video_frames" / source / video_id / f"img_{frame_number:05d}.jpg"
    heatmap_path = (
        root
        / "annotations"
        / source
        / video_id
        / "maps"
        / f"eyeMap_{frame_number:05d}.jpg"
    )
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(heatmap_path) as image:
        heatmap_image = image.convert("L")
        if heatmap_image.size != (rgb.shape[1], rgb.shape[0]):
            heatmap_image = heatmap_image.resize(
                (rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.BILINEAR
            )
        heatmap = np.asarray(heatmap_image, dtype=np.float32).copy()
    return rgb, heatmap


def add_heatmap(axis, heatmap: np.ndarray, max_alpha: float) -> None:
    peak = float(heatmap.max())
    normalized = heatmap / peak if peak > 0 else heatmap
    axis.imshow(
        normalized,
        cmap="magma",
        vmin=0,
        vmax=1,
        alpha=np.clip(np.sqrt(normalized) * max_alpha, 0, max_alpha),
    )
    if peak > 0:
        axis.contour(
            normalized,
            levels=(0.35, 0.7),
            colors=("#ff4fd8", "white"),
            linewidths=(1.15, 1.5),
            alpha=min(1.0, max_alpha + 0.18),
        )


def add_prediction(
    axis,
    cells: torch.Tensor,
    color: str,
    image_height: int,
    image_width: int,
) -> None:
    cell_height = image_height / 14
    cell_width = image_width / 14
    for cell in cells.tolist():
        row, column = divmod(cell, 14)
        axis.add_patch(
            Rectangle(
                (column * cell_width, row * cell_height),
                cell_width,
                cell_height,
                facecolor=to_rgba(color, 0.30),
                edgecolor=color,
                linewidth=1.65,
            )
        )


def gaze_centroid(
    cell_mass: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[float, float]:
    cell_height = image_height / 14
    cell_width = image_width / 14
    rows = torch.arange(14).repeat_interleave(14).to(cell_mass)
    columns = torch.arange(14).repeat(14).to(cell_mass)
    x = float(((columns + 0.5) * cell_width * cell_mass).sum())
    y = float(((rows + 0.5) * cell_height * cell_mass).sum())
    return x, y


def render_clip(
    root: Path,
    item: dict,
    example: dict,
    predictions: dict[str, torch.Tensor],
    frames_per_clip: int,
    output_path: Path,
) -> list[dict]:
    start = int(example["window_start"])
    positions = list(range(start, start + frames_per_clip))
    first_rgb, _ = load_native_frame(
        root,
        item["source"],
        item["video_id"],
        int(item["frame_numbers"][positions[0]]),
    )
    display_aspect = first_rgb.shape[0] / first_rgb.shape[1]
    figure, axes = plt.subplots(
        frames_per_clip,
        4,
        figsize=(15.2, (3.45 * display_aspect + 0.42) * frames_per_clip),
        constrained_layout=True,
    )
    if frames_per_clip == 1:
        axes = axes[np.newaxis, :]
    column_titles = ("Ground-truth gaze",) + tuple(policy[2] for policy in POLICIES)
    frame_reports = []
    for row_index, position in enumerate(positions):
        frame_number = int(item["frame_numbers"][position])
        rgb, heatmap = load_native_frame(
            root, item["source"], item["video_id"], frame_number
        )
        image_height, image_width = rgb.shape[:2]
        mass = item["cell_mass"][position]
        centroid = gaze_centroid(mass, image_height, image_width)
        frame_report = {
            "clip_position": position,
            "source_frame": frame_number,
            "timestamp_seconds": float(item["timestamps_seconds"][position]),
            "native_display_size": [image_height, image_width],
            "outside_center32_mass": float(example["outside_center_mass"][position]),
            "coverage": {},
            "selected_cells": {},
        }
        for column_index, axis in enumerate(axes[row_index]):
            axis.imshow(rgb)
            add_heatmap(axis, heatmap, max_alpha=0.92 if column_index == 0 else 0.40)
            axis.scatter(
                [centroid[0]],
                [centroid[1]],
                marker="+",
                s=125,
                linewidths=4.2,
                color="black",
                zorder=19,
            )
            axis.scatter(
                [centroid[0]],
                [centroid[1]],
                marker="+",
                s=125,
                linewidths=2.2,
                color="#ff4fd8",
                zorder=20,
            )
            if column_index > 0:
                key, budget, _, color = POLICIES[column_index - 1]
                cells = predictions[key][position]
                coverage = float(selected_coverage(mass.unsqueeze(0), cells).item())
                add_prediction(axis, cells, color, image_height, image_width)
                axis.text(
                    0.98,
                    0.025,
                    f"GT mass {coverage:.1%}",
                    ha="right",
                    va="bottom",
                    transform=axis.transAxes,
                    fontsize=8.5,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.62, "pad": 2, "edgecolor": "none"},
                )
                frame_report["coverage"][key] = coverage
                frame_report["selected_cells"][key] = cells.tolist()
            if row_index == 0:
                axis.set_title(column_titles[column_index], fontsize=11.5, fontweight="bold")
            if column_index == 0:
                axis.text(
                    -0.035,
                    0.5,
                    f"frame {int(item['frame_numbers'][position])}\n"
                    f"t={float(item['timestamps_seconds'][position]):.2f}s",
                    ha="right",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=9,
                )
            axis.set_xlim(-0.5, image_width - 0.5)
            axis.set_ylim(image_height - 0.5, -0.5)
            axis.axis("off")
        frame_reports.append(frame_report)

    figure.suptitle(
        f"Off-center validation example — {item['source']} / {item['video_id']}\n"
        f"selection score: {example['off_center_score']:.1%} mean GT mass outside Center-32",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return frame_reports


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    selection_dataset = AVGazeStavisDataset(
        root=args.root,
        manifest_path=args.manifest,
        split=args.split,
        load_rgb=False,
        load_heatmap=False,
        cell_mass_path=args.cell_mass,
    )
    if args.clips_per_source is None:
        examples = select_off_center_examples(
            selection_dataset.records,
            selection_dataset.cell_mass,
            num_clips=args.num_clips,
            frame_count=args.frames_per_clip,
            center_budget=args.center_budget,
        )
    else:
        examples = select_off_center_examples_per_source(
            selection_dataset.records,
            selection_dataset.cell_mass,
            sources=STAVIS_SOURCES,
            clips_per_source=args.clips_per_source,
            frame_count=args.frames_per_clip,
            center_budget=args.center_budget,
        )
    raw_dataset = AVGazeStavisDataset(
        root=args.root,
        manifest_path=args.manifest,
        split=args.split,
        load_rgb=True,
        load_heatmap=False,
        cell_mass_path=args.cell_mass,
    )

    checkpoints = {
        "k16": args.k16_checkpoint,
        "k24": args.k24_checkpoint,
        "k32": args.k32_checkpoint,
    }
    processor = AutoGazeImageProcessor.from_pretrained(
        checkpoints["k16"], local_files_only=True
    )
    models = {
        key: AutoGaze.from_pretrained(path, local_files_only=True).to(device).eval()
        for key, path in checkpoints.items()
    }

    report_examples = []
    with torch.inference_mode():
        for order, example in enumerate(examples, start=1):
            item = raw_dataset[int(example["dataset_index"])]
            video = preprocess_video(item["video"], processor, device)
            predictions = {
                key: generate_cells(models[key], video, budget)
                for key, budget, _, _ in POLICIES
            }
            if args.clips_per_source is None:
                slug = f"{order:02d}_{item['source']}_{item['video_id']}".replace("/", "-")
                output_path = args.output_dir / f"offcenter_{slug}.png"
            else:
                source_slug = item["source"].replace("/", "-")
                video_slug = item["video_id"].replace("/", "-").replace(" ", "_")
                output_path = (
                    args.output_dir
                    / source_slug
                    / f"{int(example['source_rank']):02d}_{video_slug}.png"
                )
            frame_reports = render_clip(
                Path(args.root), item, example, predictions, args.frames_per_clip, output_path
            )
            report_examples.append(
                {
                    **{key: value for key, value in example.items() if key != "outside_center_mass"},
                    "output_path": str(output_path),
                    "frames": frame_reports,
                }
            )
            print(output_path, flush=True)

    report = {
        "schema_version": 1,
        "split": args.split,
        "selection": {
            "criterion": "highest mean ground-truth mass outside Center-32 over a consecutive frame window",
            "model_independent": True,
            "distinct_sources": args.clips_per_source is None,
            "source_balanced": args.clips_per_source is not None,
            "clips_per_source": args.clips_per_source,
            "prefer_distinct_videos_within_source": args.clips_per_source is not None,
            "center_budget": args.center_budget,
            "frames_per_clip": args.frames_per_clip,
            "render_geometry": (
                "native-aspect RGB and heatmap with normalized 14x14 cells mapped back from "
                "the model's direct full-field 224x224 input"
            ),
        },
        "dataset": {
            "root": args.root,
            "manifest": args.manifest,
            "cell_mass": args.cell_mass,
        },
        "policies": {
            "k16": {"budget": 16, "checkpoint": str(checkpoints["k16"])},
            "k24": {"budget": 24, "checkpoint": str(checkpoints["k24"])},
            "k32": {
                "budget": 32,
                "checkpoint": str(checkpoints["k32"]),
                "note": "Exact-K32 inference from the K36-trained decoder; not a K32-trained policy.",
            },
        },
        "examples": report_examples,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(args.output_manifest)


if __name__ == "__main__":
    main()
