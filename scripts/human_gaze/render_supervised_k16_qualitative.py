#!/usr/bin/env python3
"""Render fixed-panel paired RL/SL videos and contact sheets from action exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autogaze.datasets.av_gaze_stavis import read_manifest
from autogaze.human_gaze.supervised_analysis import (
    cell_mass_for_records,
    load_action_export,
    sha256_file,
)
from scripts.human_gaze.visualize_fixed_budget_qualitative import (
    add_heatmap,
    add_prediction,
    load_native_frame,
)


METHODS = (
    ("human", "Human gaze", None),
    ("rl", "RL K16", "#7b3294"),
    ("supervised", "Supervised K16", "#00a6d6"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-mass", type=Path, required=True)
    parser.add_argument("--curated-metrics", type=Path, required=True)
    parser.add_argument("--fixed-qualitative-manifest", type=Path, required=True)
    parser.add_argument("--supervised-actions", type=Path, required=True)
    parser.add_argument("--rl-actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def draw_frame(
    axes,
    rgb: np.ndarray,
    heatmap: np.ndarray,
    rl_cells: np.ndarray,
    supervised_cells: np.ndarray,
    title: str,
) -> None:
    predictions = {"rl": rl_cells, "supervised": supervised_cells}
    for axis, (method, label, color) in zip(axes, METHODS):
        axis.imshow(rgb)
        add_heatmap(axis, heatmap, max_alpha=0.88 if method == "human" else 0.35)
        if color is not None:
            import torch

            add_prediction(
                axis,
                torch.as_tensor(predictions[method]),
                color,
                rgb.shape[0],
                rgb.shape[1],
            )
        axis.set_title(label)
        axis.axis("off")
    axes[0].text(
        0.01,
        0.02,
        title,
        transform=axes[0].transAxes,
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )


def figure_rgb(figure) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def render_video(
    *,
    root: Path,
    record: dict,
    rl_actions: np.ndarray,
    supervised_actions: np.ndarray,
    output: Path,
    fps: float,
) -> None:
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
        for position, frame_number in enumerate(record["frame_numbers"]):
            rgb, heatmap = load_native_frame(
                root, record["source"], record["video_id"], int(frame_number)
            )
            aspect = rgb.shape[1] / rgb.shape[0]
            figure, axes = plt.subplots(
                1, 3, figsize=(min(15, 4.3 * aspect * 3), 4.6), constrained_layout=True
            )
            draw_frame(
                axes,
                rgb,
                heatmap,
                rl_actions[position],
                supervised_actions[position],
                f"frame {frame_number} · t={record['timestamps_seconds'][position]:.2f}s",
            )
            figure.suptitle(f"{record['source']} / {record['video_id']}")
            writer.append_data(figure_rgb(figure))
            plt.close(figure)


def render_contact_sheet(
    *,
    root: Path,
    record: dict,
    window_start: int,
    rl_actions: np.ndarray,
    supervised_actions: np.ndarray,
    output: Path,
) -> None:
    positions = range(window_start, window_start + 4)
    figure, axes = plt.subplots(4, 3, figsize=(12.5, 13.5), constrained_layout=True)
    for row, position in enumerate(positions):
        frame_number = int(record["frame_numbers"][position])
        rgb, heatmap = load_native_frame(
            root, record["source"], record["video_id"], frame_number
        )
        draw_frame(
            axes[row],
            rgb,
            heatmap,
            rl_actions[position],
            supervised_actions[position],
            f"frame {frame_number} · t={record['timestamps_seconds'][position]:.2f}s",
        )
    figure.suptitle(f"Fixed human-selected off-center window — {record['source']} / {record['video_id']}")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace qualitative output: {args.output_dir}")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    records = read_manifest(args.manifest, split="val")
    cache = np.load(args.cell_mass, mmap_mode="r")
    cell_mass_for_records(records, cache)
    expected_manifest_sha = sha256_file(args.manifest)
    expected_mass_sha = sha256_file(args.cell_mass)
    _, supervised = load_action_export(
        args.supervised_actions,
        records,
        expected_manifest_sha256=expected_manifest_sha,
        expected_cell_mass_sha256=expected_mass_sha,
        expected_method="supervised",
        expected_base_seed=440826,
        expected_training_seed=540826,
        expected_cumulative_update=20000,
    )
    _, rl = load_action_export(
        args.rl_actions,
        records,
        expected_manifest_sha256=expected_manifest_sha,
        expected_cell_mass_sha256=expected_mass_sha,
        expected_method="rl",
        expected_base_seed=440826,
        expected_training_seed=540826,
        expected_cumulative_update=20000,
    )
    metrics = load_json(args.curated_metrics)
    qualitative = load_json(args.fixed_qualitative_manifest)
    if metrics.get("status") != "complete":
        raise ValueError("Qualitative rendering requires complete six-seed curation")
    selection = metrics.get("qualitative_selection", {})
    if selection.get("fixed_panel_base_seed") != 440826 or selection.get("fixed_panel_clips") != 24:
        raise ValueError("Curated qualitative selection violates the frozen panel")
    if (
        qualitative.get("schema_version") != 1
        or qualitative.get("split") != "val"
        or qualitative.get("selection", {}).get("model_independent") is not True
        or len(qualitative.get("examples", [])) != 24
    ):
        raise ValueError("Fixed qualitative manifest violates the frozen panel contract")
    fixed_examples = {row["clip_id"]: row for row in qualitative["examples"]}
    record_index = {record["clip_id"]: index for index, record in enumerate(records)}
    roles = selection["selected_roles"]
    selected = sorted(
        {
            (row["source"], role, row[f"{role}_clip_id"])
            for row in roles
            for role in ("representative", "failure")
        }
    )
    args.output_dir.mkdir(parents=True)
    outputs = []
    for source, role, clip_id in selected:
        if clip_id not in fixed_examples or clip_id not in record_index:
            raise ValueError(f"Curated qualitative clip is not in the fixed panel: {clip_id}")
        index = record_index[clip_id]
        record = records[index]
        safe_video = record["video_id"].replace("/", "-").replace(" ", "_")
        stem = f"{source}_{role}_{safe_video}"
        video_path = args.output_dir / f"{stem}.mp4"
        panel_path = args.output_dir / f"{stem}.png"
        render_video(
            root=args.dataset_root,
            record=record,
            rl_actions=rl[index],
            supervised_actions=supervised[index],
            output=video_path,
            fps=args.fps,
        )
        render_contact_sheet(
            root=args.dataset_root,
            record=record,
            window_start=int(fixed_examples[clip_id]["window_start"]),
            rl_actions=rl[index],
            supervised_actions=supervised[index],
            output=panel_path,
        )
        outputs.append(
            {
                "source": source,
                "role": role,
                "clip_id": clip_id,
                "video": str(video_path),
                "video_sha256": sha256_file(video_path),
                "contact_sheet": str(panel_path),
                "contact_sheet_sha256": sha256_file(panel_path),
            }
        )
    report = {
        "schema_version": 1,
        "status": "complete",
        "selection_source": str(args.curated_metrics),
        "selection_source_sha256": sha256_file(args.curated_metrics),
        "fixed_qualitative_manifest": str(args.fixed_qualitative_manifest),
        "fixed_qualitative_manifest_sha256": sha256_file(args.fixed_qualitative_manifest),
        "fixed_base_seed": 440826,
        "fps": args.fps,
        "outputs": outputs,
        "note": "Role selection is restricted to the pre-existing 24 human-selected clips.",
    }
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
