# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate selected-frame-index JSONL files before HLVid/NVILA evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runners.hlvid_common import load_rows

from common import load_selected_index_records


def frame_count_for_validation(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,r_frame_rate,duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        streams = json.loads(result.stdout).get("streams", [])
        if streams:
            stream = streams[0]
            value = stream.get("nb_frames")
            if value and str(value).isdigit():
                return int(value)
            duration = float(stream.get("duration") or 0.0)
            rate = stream.get("r_frame_rate") or "0/1"
            numerator, denominator = [float(part) for part in rate.split("/", maxsplit=1)]
            count = int(round(duration * numerator / max(denominator, 1.0)))
            if count > 0:
                return count

    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        count = 0
        for _ in container.decode(stream):
            count += 1
    if count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/storage/slurm/latn/data/AutoGaze/HLVid")
    parser.add_argument("--indices", required=True)
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    rows = load_rows(dataset_root)
    if args.limit is not None:
        rows = rows[: args.limit]
    records = load_selected_index_records(Path(args.indices))

    missing = []
    for row in rows:
        question_id = str(row["question_id"])
        record = records.get(question_id)
        if record is None:
            missing.append(question_id)
            continue
        indices = record["selected_source_indices"]
        if len(indices) != args.num_frames and not args.allow_count_mismatch:
            raise ValueError(f"{question_id}: expected {args.num_frames} indices, got {len(indices)}")
        if indices != sorted(indices):
            raise ValueError(f"{question_id}: selected_source_indices must be sorted")
        video_path = dataset_root / "videos" / row["video_path"]
        frame_count = frame_count_for_validation(video_path)
        for index in indices:
            if index < 0 or index >= frame_count:
                raise ValueError(f"{question_id}: index {index} outside [0, {frame_count}) for {row['video_path']}")

    if missing:
        raise ValueError(f"Missing selected-index records for {len(missing)} questions. First missing: {missing[:5]}")
    print(f"Validated {len(rows)} records from {args.indices}")


if __name__ == "__main__":
    main()
