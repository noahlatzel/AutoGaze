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

"""Combine per-dataset temporal delta label JSON files into one artifact."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Input delta label JSON files.")
    parser.add_argument("--output", required=True, help="Combined output JSON file.")
    return parser.parse_args()


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


def summarize(videos: List[dict], heuristics: Sequence[str]) -> Dict[str, dict]:
    summary = {
        "num_videos": len(videos),
        "num_videos_with_tiles": sum(1 for video in videos if video["tiles"]),
        "num_videos_without_tiles": sum(1 for video in videos if not video["tiles"]),
        "num_tiles": sum(len(video["tiles"]) for video in videos),
        "fps_buckets": dict(sorted(Counter(video["fps_bucket"] for video in videos).items())),
        "heuristics": {},
    }

    for heuristic in heuristics:
        labels = []
        label_seconds = []
        scores = []
        scores_by_delta = {}
        for video in videos:
            for tile in video["tiles"]:
                label = tile["labels"].get(heuristic)
                if label is not None:
                    labels.append(label)
                    if video["effective_fps"]:
                        label_seconds.append(label / video["effective_fps"])
                heuristic_scores = tile["scores"].get(heuristic, {})
                scores.extend(heuristic_scores.values())
                for delta, score in heuristic_scores.items():
                    scores_by_delta.setdefault(delta, []).append(score)

        summary["heuristics"][heuristic] = {
            "num_labels": len(labels),
            "label_counts": dict(sorted(Counter(labels).items())),
            "label_delta_seconds": percentile_summary(label_seconds),
            "num_scores": len(scores),
            "scores": percentile_summary(scores),
            "scores_by_delta": {
                delta: percentile_summary(delta_scores)
                for delta, delta_scores in sorted(scores_by_delta.items(), key=lambda item: int(item[0]))
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    inputs = [Path(path) for path in args.inputs]
    loaded = [json.loads(path.read_text()) for path in inputs]
    heuristics = loaded[0]["metadata"]["heuristics"]

    videos = []
    for path, data in zip(inputs, loaded):
        if data["metadata"]["heuristics"] != heuristics:
            raise ValueError(f"Heuristic mismatch in {path}")
        dataset_name = path.stem
        for video in data["videos"]:
            video = dict(video)
            video["label_source"] = str(path)
            video["label_source_name"] = dataset_name
            videos.append(video)

    output = {
        "metadata": {
            "input_files": [str(path) for path in inputs],
            "heuristics": heuristics,
            "deltas": loaded[0]["metadata"]["deltas"],
            "thresholds": loaded[0]["metadata"]["thresholds"],
            "tile_len": loaded[0]["metadata"]["tile_len"],
            "tile_stride": loaded[0]["metadata"]["tile_stride"],
        },
        "summary": summarize(videos, heuristics),
        "videos": videos,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
