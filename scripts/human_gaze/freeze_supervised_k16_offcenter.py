#!/usr/bin/env python3
"""Freeze the model-independent quantitative off-center validation subgroup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autogaze.datasets.av_gaze_stavis import read_manifest
from autogaze.human_gaze.supervised_analysis import (
    build_offcenter_manifest,
    cell_mass_for_records,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-mass", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--center-budget", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen manifest: {args.output}")
    records = read_manifest(args.manifest, split="val")
    cache = np.load(args.cell_mass, mmap_mode="r")
    mass = cell_mass_for_records(records, cache)
    report = build_offcenter_manifest(
        records,
        mass,
        manifest_sha256=sha256_file(args.manifest),
        cell_mass_sha256=sha256_file(args.cell_mass),
        fraction=args.fraction,
        center_budget=args.center_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "counts": report["counts"],
                "selection_sha256": report["selection_sha256"],
                "file_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
