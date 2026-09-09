# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze, validate and export the approved WP2 DINOv3 bridge panel."""

import argparse
import json
import sys
from pathlib import Path

import torch

# Direct execution must import this immutable checkout, even when another
# AutoGaze checkout is installed in editable mode in the Linux environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autogaze.human_gaze.dinov3_export import export_policies, validate_run_config
from autogaze.human_gaze.dinov3_panel import freeze_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "freeze", "export", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New panel directory for freeze/run; existing frozen panel for export")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--num-workers", type=int, choices=(0,), default=0)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 4 or not 1 <= args.cpu_threads <= 2:
        parser.error("Admitted envelope requires batch 1..4 and CPU threads 1..2")
    torch.set_num_threads(args.cpu_threads)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stage == "validate":
        print(json.dumps(validate_run_config(config), indent=2, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required for freeze/export/run")
    # Freeze may proceed while missing checkpoint paths/hashes are being resolved;
    # run validates all checkpoint files before starting any RGB work.
    validate_run_config(config, check_files=args.stage != "freeze")
    if args.stage in ("freeze", "run"):
        freeze_panel(Path(config["population_manifest"]), Path(config["dataset_root"]), args.output)
    if args.stage in ("export", "run"):
        result = export_policies(config, args.output, device=args.device, batch_size=args.batch_size,
                                 cpu_threads=args.cpu_threads, num_workers=args.num_workers)
        print(json.dumps({"manifest": str(result)}))


if __name__ == "__main__":
    main()
