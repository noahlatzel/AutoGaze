# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load real manifest clips and verify the aligned dataset contract."""

import argparse
import json

import torch

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset, STAVIS_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-clips-per-source", type=int, default=1)
    parser.add_argument("--skip-rgb", action="store_true")
    args = parser.parse_args()

    report = {}
    for split in ("train", "val", "test"):
        dataset = AVGazeStavisDataset(
            root=args.root,
            manifest_path=args.manifest,
            split=split,
            load_rgb=not args.skip_rgb,
        )
        checked = {source: 0 for source in STAVIS_SOURCES}
        for index, record in enumerate(dataset.records):
            source = record["source"]
            if checked[source] >= args.max_clips_per_source:
                continue
            item = dataset[index]
            if not args.skip_rgb and item["video"].shape != (16, 3, 224, 224):
                raise AssertionError(f"Unexpected RGB shape: {item['video'].shape}")
            if item["heatmap"].shape != (16, 224, 224):
                raise AssertionError(f"Unexpected heatmap shape: {item['heatmap'].shape}")
            if item["cell_mass"].shape != (16, 196):
                raise AssertionError(f"Unexpected cell-mass shape: {item['cell_mass'].shape}")
            if not torch.isfinite(item["cell_mass"]).all() or (item["cell_mass"] < 0).any():
                raise AssertionError(f"Invalid cell mass in {item['clip_id']}")
            torch.testing.assert_close(
                item["cell_mass"].sum(dim=-1), torch.ones(16), atol=1e-5, rtol=0
            )
            checked[source] += 1
        if any(count != args.max_clips_per_source for count in checked.values()):
            raise AssertionError(f"Insufficient source coverage for {split}: {checked}")
        report[split] = {"total_clips": len(dataset), "checked_by_source": checked}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
