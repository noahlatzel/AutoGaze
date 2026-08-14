# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate pretrained AutoGaze with fine-only, no-repeat, exact-16 actions."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset
from autogaze.human_gaze.coverage import (
    CoverageAccumulator,
    global_positions_to_fine_cells,
    selected_coverage,
)
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-clips-per-split", type=int)
    parser.add_argument("--smoke-one-per-source", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluation-splits", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    checkpoint = str(args.checkpoint) if args.checkpoint is not None else cfg["checkpoint"]
    cfg["checkpoint"] = checkpoint
    evaluation_splits = args.evaluation_splits or cfg["evaluation_splits"]
    cfg["evaluation_splits"] = evaluation_splits
    report_budgets = [int(value) for value in cfg.get("report_budgets", [1, 2, 4, 8, 16])]
    processor = AutoGazeImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoGaze.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    allowed = list(range(int(cfg["fine_action_offset"]), int(cfg["actions_per_frame"])))
    accumulator = CoverageAccumulator()
    split_counts = {}

    with torch.inference_mode():
        for split in evaluation_splits:
            dataset = AVGazeStavisDataset(
                root=cfg["dataset_root"],
                manifest_path=cfg["manifest_path"],
                split=split,
                load_rgb=True,
                load_heatmap=False,
                cell_mass_path=cfg["cell_mass_path"],
                image_processor=processor,
            )
            load_dataset = dataset
            if args.smoke_one_per_source:
                seen = set()
                indices = []
                for index, record in enumerate(dataset.records):
                    if record["source"] not in seen:
                        seen.add(record["source"])
                        indices.append(index)
                load_dataset = Subset(dataset, indices)
            loader = DataLoader(
                load_dataset,
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                shuffle=False,
                pin_memory=True,
            )
            processed = 0
            for batch in loader:
                if args.max_clips_per_split is not None and processed >= args.max_clips_per_split:
                    break
                video = batch["video"].cuda(non_blocking=True)
                gaze = model(
                    {"video": video},
                    max_gaze_tokens_each_frame=int(cfg["exact_budget"]),
                    allowed_token_ids=allowed,
                    generate_only=True,
                )
                if not torch.equal(
                    gaze["num_gazing_each_frame"],
                    torch.full_like(gaze["num_gazing_each_frame"], int(cfg["exact_budget"])),
                ):
                    raise AssertionError("AutoGaze did not return the exact per-frame budget")
                if gaze["if_padded_gazing"].any():
                    raise AssertionError("Fine-only exact-budget generation returned padding/EOS")
                fine = global_positions_to_fine_cells(
                    gaze["gazing_pos"],
                    num_frames=int(cfg["clip_len"]),
                    exact_budget=int(cfg["exact_budget"]),
                    actions_per_frame=int(cfg["actions_per_frame"]),
                    fine_action_offset=int(cfg["fine_action_offset"]),
                ).cpu()
                for index in range(fine.shape[0]):
                    for budget in report_budgets:
                        values = selected_coverage(
                            batch["cell_mass"][index], fine[index, :, :budget]
                        )
                        accumulator.add(
                            f"model_{split}",
                            budget,
                            batch["source"][index],
                            batch["video_id"][index],
                            values,
                        )
                processed += fine.shape[0]
            split_counts[split] = min(processed, len(dataset))

    metrics = {}
    for split in evaluation_splits:
        method = f"model_{split}"
        metrics[split] = accumulator.finalize([method], report_budgets)[method]
    output = {
        "schema_version": 1,
        "config": cfg,
        "processed_clips": split_counts,
        "metrics": metrics,
    }
    output_path = args.output or Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        split: metrics[split][str(cfg["exact_budget"])]["macro_source_mean"]
        for split in evaluation_splits
    }, sort_keys=True))


if __name__ == "__main__":
    main()
