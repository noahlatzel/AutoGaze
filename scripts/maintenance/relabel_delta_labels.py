# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relabel temporal delta score JSONs with new heuristic thresholds."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Existing delta-label JSON with all-action scores.")
    parser.add_argument("--output", required=True, help="Output JSON with updated labels and summary.")
    parser.add_argument(
        "--thresholds",
        required=True,
        help="Comma-separated thresholds, e.g. pixel_mse=0.04,siglip=0.0125.",
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> Dict[str, float]:
    thresholds = {}
    for item in value.split(","):
        if not item.strip():
            continue
        name, threshold = item.split("=", 1)
        thresholds[name.strip()] = float(threshold)
    return thresholds


def choose_label(scores: Dict[str, float], threshold: Optional[float]) -> Optional[int]:
    if threshold is None or not scores:
        return None
    valid = [int(delta) for delta, score in scores.items() if float(score) <= threshold]
    return max(valid) if valid else min(int(delta) for delta in scores)


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


def summarize(videos: list[dict], heuristics: Sequence[str]) -> Dict[str, dict]:
    summary = {
        "num_videos": len(videos),
        "num_videos_with_tiles": sum(1 for video in videos if video["tiles"]),
        "num_videos_without_tiles": sum(1 for video in videos if not video["tiles"]),
        "num_tiles": sum(len(video["tiles"]) for video in videos),
        "fps_buckets": dict(sorted(Counter(video.get("fps_bucket") or "unknown" for video in videos).items())),
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
                    if video.get("effective_fps"):
                        label_seconds.append(label / video["effective_fps"])
                heuristic_scores = tile["scores"].get(heuristic, {})
                scores.extend(float(score) for score in heuristic_scores.values())
                for delta, score in heuristic_scores.items():
                    scores_by_delta.setdefault(delta, []).append(float(score))
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
    data = json.loads(Path(args.input).read_text())
    thresholds = parse_thresholds(args.thresholds)
    heuristics = data["metadata"]["heuristics"]

    unknown = sorted(set(thresholds) - set(heuristics))
    if unknown:
        raise ValueError(f"Thresholds provided for missing heuristics: {unknown}")

    for video in data["videos"]:
        for tile in video["tiles"]:
            for heuristic in heuristics:
                if heuristic in thresholds:
                    tile["labels"][heuristic] = choose_label(tile["scores"].get(heuristic, {}), thresholds[heuristic])

    data["metadata"]["thresholds"] = {
        **data["metadata"].get("thresholds", {}),
        **thresholds,
    }
    data["summary"] = summarize(data["videos"], heuristics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
