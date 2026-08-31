# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate every STAViS heatmap clip and cache compact 14x14 cell masses."""

import argparse
import json
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from autogaze.datasets.av_gaze_stavis import (
    cache_valid_cell_masses,
    read_manifest,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    records = read_manifest(Path(cfg["input_manifest"]))
    summary = json.loads(Path(cfg["input_summary"]).read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])
    valid, invalid = cache_valid_cell_masses(
        root=Path(cfg["dataset_root"]),
        records=records,
        output_dir=output_dir,
        image_size=int(cfg["image_size"]),
        grid_size=int(cfg["grid_size"]),
        num_workers=int(cfg["num_workers"]),
    )

    invalid_ids = {item["clip_id"] for item in invalid}
    invalid_records = [record for record in records if record["clip_id"] in invalid_ids]
    invalid_by_source = Counter(record["source"] for record in invalid_records)
    invalid_by_split = Counter(record["split"] for record in invalid_records)
    for source, count in invalid_by_source.items():
        summary["sources"][source]["invalid_heatmap_clips"] = count
    summary["invalid_heatmap_clips"] = len(invalid)
    summary["total_clips"] = len(valid)
    summary["clips_by_split"] = {
        split: sum(record["split"] == split for record in valid)
        for split in ("train", "val", "test")
    }
    for source in summary["sources"]:
        summary["sources"][source]["clips_by_split"] = {
            split: sum(record["source"] == source and record["split"] == split for record in valid)
            for split in ("train", "val", "test")
        }

    write_manifest(valid, summary, output_dir)
    OmegaConf.save(OmegaConf.create(cfg), output_dir / "resolved_target_config.yaml")
    print(json.dumps({
        "valid_clips": len(valid),
        "invalid_clips": len(invalid),
        "invalid_by_source": dict(invalid_by_source),
        "invalid_by_split": dict(invalid_by_split),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
