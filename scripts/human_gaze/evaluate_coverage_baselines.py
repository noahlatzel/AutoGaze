# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate fixed fine-grid coverage baselines on manifest-defined splits."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset
from autogaze.human_gaze.coverage import compute_source_priors, evaluate_coverage_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    train = AVGazeStavisDataset(
        root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        split="train",
        image_size=int(cfg["image_size"]),
        grid_size=int(cfg["grid_size"]),
        load_rgb=False,
        load_heatmap=False,
        cell_mass_path=cfg["cell_mass_path"],
    )
    loader_kwargs = {
        "batch_size": None,
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": False,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["prefetch_factor"])
    priors = compute_source_priors(DataLoader(train, **loader_kwargs))
    results = {
        "schema_version": 1,
        "config": cfg,
        "prior_cell_order": {
            source: torch.argsort(prior, descending=True, stable=True).tolist()
            for source, prior in priors.items()
        },
        "splits": {},
    }
    for split in cfg["evaluation_splits"]:
        dataset = AVGazeStavisDataset(
            root=cfg["dataset_root"],
            manifest_path=cfg["manifest_path"],
            split=split,
            image_size=int(cfg["image_size"]),
            grid_size=int(cfg["grid_size"]),
            load_rgb=False,
            load_heatmap=False,
            cell_mass_path=cfg["cell_mass_path"],
        )
        results["splits"][split] = evaluate_coverage_baselines(
            DataLoader(dataset, **loader_kwargs),
            source_priors=priors,
            budgets=[int(value) for value in cfg["budgets"]],
            random_seeds=[int(value) for value in cfg["random_seeds"]],
            grid_size=int(cfg["grid_size"]),
        )

    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                split: {
                    method: {
                        str(budget): metrics[str(budget)]["macro_source_mean"]
                        for budget in cfg["budgets"]
                    }
                    for method, metrics in split_results.items()
                }
                for split, split_results in results["splits"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
