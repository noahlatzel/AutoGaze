# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate uniform selected-frame JSONL for testing the selected_indices path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.frame_selectors.common import selected_index_record, write_jsonl
from scripts.frame_selectors.validate_selected_indices import frame_count_for_validation
from scripts.runners.hlvid_common import load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/storage/slurm/latn/data/AutoGaze/HLVid")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    rows = load_rows(dataset_root)
    if args.limit is not None:
        rows = rows[: args.limit]
    records = []
    for row in rows:
        video_path = dataset_root / "videos" / row["video_path"]
        frame_count = frame_count_for_validation(video_path)
        indices = np.round(np.linspace(0, frame_count - 1, args.num_frames)).astype(int).tolist()
        records.append(
            selected_index_record(
                question_id=str(row["question_id"]),
                video_path=str(row["video_path"]),
                method="uniform_indices",
                num_frames=args.num_frames,
                selected_source_indices=indices,
                frame_count=frame_count,
            )
        )
    write_jsonl(records, Path(args.output))
    print(f"Wrote {len(records)} uniform selected-index records to {args.output}")


if __name__ == "__main__":
    main()
