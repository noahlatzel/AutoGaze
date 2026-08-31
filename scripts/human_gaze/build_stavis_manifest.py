# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a read-only AV-gaze-STAViS clip manifest from a resolved YAML config."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from autogaze.datasets.av_gaze_stavis import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    records, summary = build_manifest(
        root=Path(cfg["dataset_root"]),
        fold=int(cfg["fold"]),
        target_fps=float(cfg["target_fps"]),
        clip_len=int(cfg["clip_len"]),
        clip_stride=int(cfg["clip_stride"]),
        validation_fraction=float(cfg["validation_fraction"]),
        split_seed=int(cfg["split_seed"]),
    )
    output_dir = Path(cfg["output_dir"])
    write_manifest(records, summary, output_dir)
    OmegaConf.save(OmegaConf.create(cfg), output_dir / "resolved_config.yaml")
    print(json.dumps({"output_dir": str(output_dir), **summary["clips_by_split"]}, sort_keys=True))


if __name__ == "__main__":
    main()
