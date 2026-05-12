# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyze dataset temporal metadata for delta scheduler Experiment 0A.

This script does not decode full videos or load models. It reads container
metadata, file sizes, and filename-encoded FPS/time spans to decide which delta
actions are feasible and how frame deltas map to wall-clock time.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import av
import numpy as np
from tqdm import tqdm


FPS_RE = re.compile(r"_fps(?P<fps>[0-9]+(?:\.[0-9]+)?)")
TIME_RE = re.compile(r"_t(?P<start>[0-9]+(?:\.[0-9]+)?)-(?P<end>[0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help=(
            "One or more dataset/video roots. Each root can contain videos directly "
            "or contain split subdirectories such as train/ and val/."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Optional split subdirectory to use under each input root. If root/split "
            "does not exist, the root itself is scanned."
        ),
    )
    parser.add_argument("--video-ext", default=".mp4", help="Video filename extension to include.")
    parser.add_argument("--output", required=True, help="Output JSON path for records and aggregate summaries.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to analyze across all roots. Omit for all discovered videos.",
    )
    parser.add_argument("--tile-len", type=int, default=16, help="Number of frames in one AutoGaze temporal tile.")
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=16,
        help="Spacing between current tile starts used when counting feasible tile positions.",
    )
    parser.add_argument(
        "--deltas",
        default="1,2,4,8,16,32,64",
        help="Comma-separated frame deltas to test for feasibility and seconds conversion.",
    )
    parser.add_argument(
        "--fps-buckets",
        default="low:0:4,medium:4:10,high:10:inf",
        help=(
            "Comma-separated FPS buckets as name:min:max. Bounds are [min, max). "
            "Use inf for an open upper bound. Unknown/missing FPS is bucketed as unknown."
        ),
    )
    parser.add_argument(
        "--include-records",
        action="store_true",
        help="Include one metadata record per video in the output JSON. Summaries are always written.",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_fps_buckets(value: str) -> List[Dict[str, float]]:
    buckets = []
    for item in value.split(","):
        if not item.strip():
            continue
        name, lower, upper = item.split(":")
        buckets.append(
            {
                "name": name,
                "lower": float(lower),
                "upper": float("inf") if upper == "inf" else float(upper),
            }
        )
    return buckets


def bucket_for_fps(fps: Optional[float], buckets: Sequence[Dict[str, float]]) -> str:
    if fps is None:
        return "unknown"
    for bucket in buckets:
        if bucket["lower"] <= fps < bucket["upper"]:
            return bucket["name"]
    return "other"


def iter_video_paths(input_roots: Sequence[str], split: Optional[str], video_ext: str) -> Iterable[Path]:
    for raw_root in input_roots:
        root = Path(raw_root)
        search_root = root / split if split and (root / split).is_dir() else root
        if not search_root.exists():
            raise FileNotFoundError(f"Video root does not exist: {search_root}")
        yield from search_root.rglob(f"*{video_ext}")


def get_container_metadata(path: Path) -> Dict[str, Optional[float]]:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            average_rate = float(stream.average_rate) if stream.average_rate is not None else None
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            return {
                "frame_count": int(stream.frames or 0),
                "metadata_fps": average_rate,
                "metadata_duration_seconds": duration,
            }
    except Exception:
        return {
            "frame_count": None,
            "metadata_fps": None,
            "metadata_duration_seconds": None,
        }


def parse_filename_metadata(path: Path) -> Dict[str, Optional[float]]:
    name = path.name
    fps_match = FPS_RE.search(name)
    time_match = TIME_RE.search(name)
    fps = float(fps_match.group("fps")) if fps_match else None
    start = float(time_match.group("start")) if time_match else None
    end = float(time_match.group("end")) if time_match else None
    duration = end - start if start is not None and end is not None else None
    return {
        "filename_fps": fps,
        "filename_start_seconds": start,
        "filename_end_seconds": end,
        "filename_duration_seconds": duration,
    }


def feasible_tile_starts(frame_count: int, tile_len: int, tile_stride: int, deltas: Sequence[int]) -> List[int]:
    if frame_count is None or frame_count <= 0:
        return []
    min_delta = min(deltas)
    max_start = frame_count - min_delta - tile_len
    if max_start < 0:
        return []
    return list(range(0, max_start + 1, tile_stride))


def feasible_deltas_for_start(frame_count: int, start: int, tile_len: int, deltas: Sequence[int]) -> List[int]:
    return [delta for delta in deltas if start + delta + tile_len <= frame_count]


def percentile_summary(values: Sequence[float]) -> Optional[Dict[str, float]]:
    clean = [value for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def summarize_records(records: List[dict], deltas: Sequence[int], fps_buckets: Sequence[Dict[str, float]]) -> Dict[str, dict]:
    summary = {
        "num_videos": len(records),
        "fps_buckets": {},
        "frame_count": percentile_summary([record["frame_count"] for record in records]),
        "file_size_mb": percentile_summary([record["file_size_mb"] for record in records]),
        "filename_fps": percentile_summary([record["filename_fps"] for record in records]),
        "metadata_fps": percentile_summary([record["metadata_fps"] for record in records]),
        "effective_fps": percentile_summary([record["effective_fps"] for record in records]),
        "filename_duration_seconds": percentile_summary([record["filename_duration_seconds"] for record in records]),
        "metadata_duration_seconds": percentile_summary([record["metadata_duration_seconds"] for record in records]),
        "tile_duration_seconds": percentile_summary([record["tile_duration_seconds"] for record in records]),
        "num_tile_starts": percentile_summary([record["num_tile_starts"] for record in records]),
        "max_feasible_delta": percentile_summary([record["max_feasible_delta"] for record in records]),
        "feasible_delta_counts": dict(Counter(delta for record in records for delta in record["feasible_deltas_anywhere"])),
    }

    bucket_names = [bucket["name"] for bucket in fps_buckets] + ["unknown", "other"]
    for bucket_name in bucket_names:
        bucket_records = [record for record in records if record["fps_bucket"] == bucket_name]
        if not bucket_records:
            continue
        summary["fps_buckets"][bucket_name] = {
            "num_videos": len(bucket_records),
            "frame_count": percentile_summary([record["frame_count"] for record in bucket_records]),
            "filename_fps": percentile_summary([record["filename_fps"] for record in bucket_records]),
            "metadata_fps": percentile_summary([record["metadata_fps"] for record in bucket_records]),
            "effective_fps": percentile_summary([record["effective_fps"] for record in bucket_records]),
            "tile_duration_seconds": percentile_summary([record["tile_duration_seconds"] for record in bucket_records]),
            "num_tile_starts": percentile_summary([record["num_tile_starts"] for record in bucket_records]),
            "max_feasible_delta": percentile_summary([record["max_feasible_delta"] for record in bucket_records]),
            "feasible_delta_counts": dict(Counter(delta for record in bucket_records for delta in record["feasible_deltas_anywhere"])),
            "delta_seconds": {
                str(delta): percentile_summary(
                    [
                        record["delta_seconds"].get(str(delta))
                        for record in bucket_records
                        if str(delta) in record["delta_seconds"]
                    ]
                )
                for delta in deltas
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    deltas = parse_csv_ints(args.deltas)
    fps_buckets = parse_fps_buckets(args.fps_buckets)

    records = []
    paths = iter_video_paths(args.input_roots, args.split, args.video_ext)
    for path in tqdm(paths, desc="videos"):
        filename_meta = parse_filename_metadata(path)
        container_meta = get_container_metadata(path)
        frame_count = container_meta["frame_count"]
        if frame_count is None or frame_count <= 0:
            continue

        fps = filename_meta["filename_fps"] or container_meta["metadata_fps"]
        tile_starts = feasible_tile_starts(frame_count, args.tile_len, args.tile_stride, deltas)
        feasible_by_start = {
            str(start): feasible_deltas_for_start(frame_count, start, args.tile_len, deltas)
            for start in tile_starts
        }
        feasible_anywhere = sorted({delta for feasible in feasible_by_start.values() for delta in feasible})
        delta_seconds = {str(delta): delta / fps for delta in feasible_anywhere} if fps else {}
        tile_duration_seconds = args.tile_len / fps if fps else None

        records.append(
            {
                "path": str(path),
                "file_size_mb": path.stat().st_size / (1024 * 1024),
                "frame_count": frame_count,
                "metadata_fps": container_meta["metadata_fps"],
                "metadata_duration_seconds": container_meta["metadata_duration_seconds"],
                **filename_meta,
                "fps_bucket": bucket_for_fps(fps, fps_buckets),
                "effective_fps": fps,
                "tile_duration_seconds": tile_duration_seconds,
                "num_tile_starts": len(tile_starts),
                "feasible_deltas_anywhere": feasible_anywhere,
                "max_feasible_delta": max(feasible_anywhere) if feasible_anywhere else None,
                "delta_seconds": delta_seconds,
            }
        )
        if args.limit is not None and len(records) >= args.limit:
            break

    output = {
        "metadata": {
            "input_roots": args.input_roots,
            "split": args.split,
            "video_ext": args.video_ext,
            "limit": args.limit,
            "tile_len": args.tile_len,
            "tile_stride": args.tile_stride,
            "deltas": deltas,
            "fps_buckets": fps_buckets,
        },
        "summary": summarize_records(records, deltas, fps_buckets),
    }
    if args.include_records:
        output["records"] = records

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
