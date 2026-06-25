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

"""Generate utility-based pseudo-labels for temporal delta scheduler experiments."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import av
import numpy as np
import torch
from tqdm import tqdm

from autogaze.datasets.video_utils import read_video_pyav
from autogaze.temporal_sampling import (
    PAIRWISE_METRICS,
    SEMANTIC_MODELS,
    SEMANTIC_PAIRWISE_METRICS,
    SEMANTIC_WINDOW_METRICS,
    TEMPORAL_COSTS,
    WINDOW_METRICS,
    Dinov2SemanticEncoder,
    SiglipSemanticEncoder,
    score_semantic_pairwise_metric,
    score_semantic_window_metric,
    score_window_metric,
    select_best_action,
    thumbnail_tensor,
)


DEFAULT_MODEL_ROOT = "/storage/slurm/latn/data/AutoGaze/hf_models"
FPS_RE = re.compile(r"_fps(?P<fps>[0-9]+(?:\.[0-9]+)?)")
QUESTION_CONDITIONED_METRICS = {"siglip_qgain", "question_relevance"}


@dataclass
class VideoRecord:
    path: Path
    relative_path: str
    frame_count: int
    filename_fps: Optional[float]
    metadata_fps: Optional[float]
    effective_fps: Optional[float]
    fps_bucket: str


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
            "One or more dataset/video roots. Each root can either contain videos "
            "directly or contain split subdirectories such as train/ and val/."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Optional split subdirectory to use under each input root, e.g. train or val. "
            "If root/split does not exist, the root itself is used."
        ),
    )
    parser.add_argument("--video-ext", default=".mp4", help="Video filename extension to include.")
    parser.add_argument("--output", required=True, help="Output JSON path for metadata, raw scores, utilities, and labels.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of valid videos to process across all input roots.")
    parser.add_argument(
        "--limit-per-root",
        type=int,
        default=None,
        help="Maximum number of valid videos to process from each input root before moving to the next root.",
    )
    parser.add_argument(
        "--max-tiles-per-video",
        type=int,
        default=None,
        help="Maximum current tile start positions to evaluate per video; starts are taken in temporal order.",
    )
    parser.add_argument("--tile-len", type=int, default=16, help="Number of frames in each current/candidate chunk.")
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=16,
        help="Spacing between evaluated current chunk starts.",
    )
    parser.add_argument(
        "--deltas",
        default="1,2,4,8,16,32",
        help=(
            "Comma-separated candidate gaps after the current chunk. Candidate chunk start is "
            "current_start + tile_len + delta."
        ),
    )
    parser.add_argument(
        "--proxy-metrics",
        default="mean_novelty",
        help=(
            "Comma-separated proxy metrics from autogaze.temporal_sampling. Question-conditioned "
            "metrics are intentionally unsupported in this dataset-level generator because it has no question text."
        ),
    )
    parser.add_argument(
        "--proxy-base-metric",
        choices=list(PAIRWISE_METRICS),
        default="pixel_mse",
        help="Pairwise thumbnail distance used by non-semantic window metrics.",
    )
    parser.add_argument("--lambda-cost", type=float, default=0.01, help="Weight for temporal action cost.")
    parser.add_argument(
        "--temporal-cost",
        choices=list(TEMPORAL_COSTS),
        default="log",
        help="Temporal action cost used when selecting the best candidate.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Spatial resize used before non-semantic thumbnail metrics.",
    )
    parser.add_argument(
        "--full-video-decode-max-frames",
        type=int,
        default=256,
        help=(
            "Decode a whole video once when frame_count is at most this value, then slice chunks from memory. "
            "Set to 0 to always decode individual chunks."
        ),
    )
    parser.add_argument(
        "--semantic-model",
        choices=list(SEMANTIC_MODELS),
        default="siglip",
        help="Embedding model used by semantic_mean_novelty and semantic_diversity.",
    )
    parser.add_argument(
        "--siglip-model",
        default=f"{DEFAULT_MODEL_ROOT}/google--siglip2-base-patch16-224",
        help="Local path or Hugging Face model name for SigLIP semantic metrics.",
    )
    parser.add_argument(
        "--dinov2-model",
        default=f"{DEFAULT_MODEL_ROOT}/facebook--dinov2-with-registers-base",
        help="Local path or Hugging Face model name for DINOv2 semantic metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Frame batch size for semantic feature extraction.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device for semantic metrics.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32", help="Torch dtype for semantic metrics.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic torch/numpy setup.")
    parser.add_argument(
        "--fps-buckets",
        default="low:0:4,medium:4:10,high:10:inf",
        help=(
            "Comma-separated FPS buckets as name:min:max. Bounds are [min, max). "
            "Use inf for an open upper bound. Unknown/missing FPS is bucketed as unknown."
        ),
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def discover_videos(
    input_roots: Sequence[str],
    split: Optional[str],
    video_ext: str,
    limit: Optional[int],
    limit_per_root: Optional[int],
    fps_buckets: Sequence[Dict[str, float]],
) -> List[VideoRecord]:
    records = []
    for raw_root in input_roots:
        root = Path(raw_root)
        search_root = root / split if split and (root / split).is_dir() else root
        if not search_root.exists():
            raise FileNotFoundError(f"Video root does not exist: {search_root}")
        root_records = 0
        for path in search_root.rglob(f"*{video_ext}"):
            video_metadata = get_video_metadata(path)
            frame_count = video_metadata["frame_count"]
            if frame_count <= 0:
                continue
            effective_fps = video_metadata["filename_fps"] or video_metadata["metadata_fps"]
            records.append(
                VideoRecord(
                    path=path,
                    relative_path=str(path.relative_to(search_root)),
                    frame_count=frame_count,
                    filename_fps=video_metadata["filename_fps"],
                    metadata_fps=video_metadata["metadata_fps"],
                    effective_fps=effective_fps,
                    fps_bucket=bucket_for_fps(effective_fps, fps_buckets),
                )
            )
            root_records += 1
            if limit is not None and len(records) >= limit:
                return records
            if limit_per_root is not None and root_records >= limit_per_root:
                break
    return records


def get_video_metadata(path: Path) -> Dict[str, Optional[float]]:
    filename_fps = parse_filename_fps(path)
    metadata_fps = None
    frame_count = 0
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            metadata_fps = float(stream.average_rate) if stream.average_rate is not None else None
            frame_count = int(stream.frames or 0)
            if frame_count <= 0:
                frame_count = sum(1 for _ in container.decode(video=0))
    except Exception:
        frame_count = 0
    return {
        "frame_count": frame_count,
        "filename_fps": filename_fps,
        "metadata_fps": metadata_fps,
    }


def parse_filename_fps(path: Path) -> Optional[float]:
    fps_match = FPS_RE.search(path.name)
    return float(fps_match.group("fps")) if fps_match else None


def valid_starts(frame_count: int, tile_len: int, tile_stride: int, deltas: Sequence[int]) -> List[int]:
    min_delta = min(deltas)
    max_start = frame_count - (2 * tile_len) - min_delta
    if max_start < 0:
        return []
    return list(range(0, max_start + 1, tile_stride))


def candidate_start(current_start: int, tile_len: int, delta: int) -> int:
    return current_start + tile_len + delta


def read_chunk(path: Path, start: int, tile_len: int) -> np.ndarray:
    with av.open(str(path)) as container:
        indices = np.arange(start, start + tile_len)
        return read_video_pyav(container, indices)


def read_video(path: Path, frame_count: int) -> np.ndarray:
    with av.open(str(path)) as container:
        indices = np.arange(frame_count)
        return read_video_pyav(container, indices)


class ProxyMetricScorer:
    def __init__(self, args: argparse.Namespace, proxy_metrics: Sequence[str]):
        self.args = args
        self.dtype = torch_dtype(args.dtype)
        self.siglip_encoder = None
        self.dinov2_encoder = None
        if any(requires_semantic_model(metric, "siglip", args.semantic_model) for metric in proxy_metrics):
            self.siglip_encoder = SiglipSemanticEncoder(args.siglip_model, args.device, self.dtype, args.batch_size)
        if any(requires_semantic_model(metric, "dinov2", args.semantic_model) for metric in proxy_metrics):
            self.dinov2_encoder = Dinov2SemanticEncoder(args.dinov2_model, args.device, self.dtype, args.batch_size)

    def score(
        self,
        metric: str,
        current_start: int,
        next_start: int,
        get_chunk,
        resized_cache: Dict[int, torch.Tensor],
        semantic_cache: Dict[str, Dict[int, torch.Tensor]],
    ) -> float:
        def get_resized(start: int) -> torch.Tensor:
            if start not in resized_cache:
                resized_cache[start] = thumbnail_tensor(get_chunk(start), self.args.resize)
            return resized_cache[start]

        def get_semantic(model_name: str, start: int) -> torch.Tensor:
            if start not in semantic_cache[model_name]:
                encoder = self.semantic_encoder(model_name)
                semantic_cache[model_name][start] = encoder.encode_frames(get_chunk(start))
            return semantic_cache[model_name][start]

        if metric in PAIRWISE_METRICS:
            return float(
                score_window_metric(
                    get_resized(current_start),
                    get_resized(next_start),
                    "mean_pairwise_change",
                    base_metric=metric,
                ).item()
            )

        if metric in WINDOW_METRICS:
            return float(
                score_window_metric(
                    get_resized(current_start),
                    get_resized(next_start),
                    metric,
                    base_metric=self.args.proxy_base_metric,
                ).item()
            )

        if metric in SEMANTIC_PAIRWISE_METRICS:
            model_name = semantic_model_for_metric(metric, self.args.semantic_model)
            return float(
                score_semantic_pairwise_metric(
                    get_semantic(model_name, current_start),
                    get_semantic(model_name, next_start),
                    metric,
                ).item()
            )

        if metric in SEMANTIC_WINDOW_METRICS:
            model_name = semantic_model_for_metric(metric, self.args.semantic_model)
            return float(
                score_semantic_window_metric(
                    get_semantic(model_name, current_start),
                    get_semantic(model_name, next_start),
                    metric,
                ).item()
            )

        raise ValueError(f"Unknown proxy metric: {metric}")

    def semantic_encoder(self, model_name: str):
        if model_name == "siglip":
            if self.siglip_encoder is None:
                raise ValueError("SigLIP encoder was not initialized")
            return self.siglip_encoder
        if model_name == "dinov2":
            if self.dinov2_encoder is None:
                raise ValueError("DINOv2 encoder was not initialized")
            return self.dinov2_encoder
        raise ValueError(f"Unknown semantic model: {model_name}")


def requires_semantic_model(metric: str, model_name: str, default_semantic_model: str) -> bool:
    if metric in ("siglip_cosine",):
        return model_name == "siglip"
    if metric in ("dinov2_cosine",):
        return model_name == "dinov2"
    if metric in ("semantic_mean_novelty", "semantic_diversity"):
        return model_name == default_semantic_model
    return False


def semantic_model_for_metric(metric: str, default_semantic_model: str) -> str:
    if metric == "siglip_cosine":
        return "siglip"
    if metric == "dinov2_cosine":
        return "dinov2"
    return default_semantic_model


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


def summarize_labels(results: List[dict], proxy_metrics: Sequence[str]) -> Dict[str, dict]:
    num_tiles = sum(len(video["tiles"]) for video in results)
    summary = {
        "num_videos": len(results),
        "num_videos_with_tiles": sum(1 for video in results if video["tiles"]),
        "num_videos_without_tiles": sum(1 for video in results if not video["tiles"]),
        "num_tiles": num_tiles,
        "fps_buckets": dict(sorted(Counter(video["fps_bucket"] for video in results).items())),
        "proxy_metrics": {},
    }
    for metric in proxy_metrics:
        labels = []
        label_seconds = []
        scores = []
        utilities = []
        scores_by_delta = {}
        utilities_by_delta = {}
        bucket_summaries = {}
        for video in results:
            for tile in video["tiles"]:
                label = tile["labels"].get(metric)
                if label is not None:
                    labels.append(label)
                    if video["effective_fps"]:
                        label_seconds.append(label / video["effective_fps"])
                metric_scores = tile["scores"].get(metric, {})
                metric_utilities = tile["utilities"].get(metric, {})
                scores.extend(metric_scores.values())
                utilities.extend(metric_utilities.values())
                for delta, score in metric_scores.items():
                    scores_by_delta.setdefault(delta, []).append(score)
                for delta, utility in metric_utilities.items():
                    utilities_by_delta.setdefault(delta, []).append(utility)

        for bucket_name in sorted({video["fps_bucket"] for video in results}):
            bucket_labels = []
            bucket_label_seconds = []
            bucket_scores = []
            bucket_utilities = []
            bucket_score_by_delta = {}
            bucket_utility_by_delta = {}
            for video in results:
                if video["fps_bucket"] != bucket_name:
                    continue
                for tile in video["tiles"]:
                    label = tile["labels"].get(metric)
                    if label is not None:
                        bucket_labels.append(label)
                        if video["effective_fps"]:
                            bucket_label_seconds.append(label / video["effective_fps"])
                    metric_scores = tile["scores"].get(metric, {})
                    metric_utilities = tile["utilities"].get(metric, {})
                    bucket_scores.extend(metric_scores.values())
                    bucket_utilities.extend(metric_utilities.values())
                    for delta, score in metric_scores.items():
                        bucket_score_by_delta.setdefault(delta, []).append(score)
                    for delta, utility in metric_utilities.items():
                        bucket_utility_by_delta.setdefault(delta, []).append(utility)
            bucket_summaries[bucket_name] = {
                "num_labels": len(bucket_labels),
                "label_counts": dict(sorted(Counter(bucket_labels).items())),
                "label_delta_seconds": percentile_summary(bucket_label_seconds),
                "num_scores": len(bucket_scores),
                "scores": percentile_summary(bucket_scores),
                "utilities": percentile_summary(bucket_utilities),
                "scores_by_delta": {
                    delta: percentile_summary(delta_scores)
                    for delta, delta_scores in sorted(bucket_score_by_delta.items(), key=lambda item: int(item[0]))
                },
                "utilities_by_delta": {
                    delta: percentile_summary(delta_utilities)
                    for delta, delta_utilities in sorted(bucket_utility_by_delta.items(), key=lambda item: int(item[0]))
                },
            }

        summary["proxy_metrics"][metric] = {
            "num_labels": len(labels),
            "label_counts": dict(sorted(Counter(labels).items())),
            "label_delta_seconds": percentile_summary(label_seconds),
            "num_scores": len(scores),
            "scores": percentile_summary(scores),
            "utilities": percentile_summary(utilities),
            "scores_by_delta": {
                delta: percentile_summary(delta_scores)
                for delta, delta_scores in sorted(scores_by_delta.items(), key=lambda item: int(item[0]))
            },
            "utilities_by_delta": {
                delta: percentile_summary(delta_utilities)
                for delta, delta_utilities in sorted(utilities_by_delta.items(), key=lambda item: int(item[0]))
            },
            "fps_buckets": bucket_summaries,
        }
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    deltas = parse_csv_ints(args.deltas)
    proxy_metrics = parse_csv_strings(args.proxy_metrics)
    fps_buckets = parse_fps_buckets(args.fps_buckets)

    supported = set(PAIRWISE_METRICS) | set(WINDOW_METRICS) | set(SEMANTIC_PAIRWISE_METRICS) | set(SEMANTIC_WINDOW_METRICS)
    unsupported = sorted(set(proxy_metrics) - supported)
    if unsupported:
        raise ValueError(f"Unknown proxy metrics: {unsupported}")

    # Question-conditioned metrics need per-video question text. This generator
    # operates on raw video folders, so keep them out until the dataset schema
    # includes a question field and text-embedding cache.
    question_conditioned = sorted(set(proxy_metrics) & QUESTION_CONDITIONED_METRICS)
    if question_conditioned:
        raise ValueError(
            "Question-conditioned proxy metrics are not supported by generate_delta_labels.py yet: "
            f"{question_conditioned}"
        )

    videos = discover_videos(args.input_roots, args.split, args.video_ext, args.limit, args.limit_per_root, fps_buckets)
    scorer = ProxyMetricScorer(args, proxy_metrics)

    all_results = []
    for video in tqdm(videos, desc="videos"):
        starts = valid_starts(video.frame_count, args.tile_len, args.tile_stride, deltas)
        if args.max_tiles_per_video is not None:
            starts = starts[: args.max_tiles_per_video]

        chunk_cache: Dict[int, np.ndarray] = {}
        resized_cache: Dict[int, torch.Tensor] = {}
        semantic_cache: Dict[str, Dict[int, torch.Tensor]] = {"siglip": {}, "dinov2": {}}
        full_video = None
        if starts and args.full_video_decode_max_frames > 0 and video.frame_count <= args.full_video_decode_max_frames:
            full_video = read_video(video.path, video.frame_count)

        def get_chunk(start: int) -> np.ndarray:
            if start not in chunk_cache:
                if full_video is not None:
                    chunk_cache[start] = full_video[start : start + args.tile_len]
                else:
                    chunk_cache[start] = read_chunk(video.path, start, args.tile_len)
            return chunk_cache[start]

        needed_starts = set(starts)
        for start in starts:
            for delta in deltas:
                next_start = candidate_start(start, args.tile_len, delta)
                if next_start + args.tile_len <= video.frame_count:
                    needed_starts.add(next_start)
        for needed_start in sorted(needed_starts):
            get_chunk(needed_start)

        video_tiles = []
        for start in starts:
            scores = {metric: {} for metric in proxy_metrics}
            utilities_by_metric = {metric: {} for metric in proxy_metrics}
            labels = {}
            timing_seconds = {metric: 0.0 for metric in proxy_metrics}

            for metric in proxy_metrics:
                metric_started = time.perf_counter()
                for delta in deltas:
                    next_start = candidate_start(start, args.tile_len, delta)
                    if next_start + args.tile_len > video.frame_count:
                        continue
                    scores[metric][str(delta)] = scorer.score(
                        metric,
                        start,
                        next_start,
                        get_chunk,
                        resized_cache,
                        semantic_cache,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_seconds[metric] += time.perf_counter() - metric_started

                if scores[metric]:
                    selected_delta, utilities = select_best_action(
                        {int(delta): score for delta, score in scores[metric].items()},
                        lambda_cost=args.lambda_cost,
                        cost=args.temporal_cost,
                    )
                    labels[metric] = selected_delta
                    utilities_by_metric[metric] = {str(delta): utility for delta, utility in utilities.items()}
                else:
                    labels[metric] = None

            video_tiles.append(
                {
                    "start": start,
                    "labels": labels,
                    "scores": scores,
                    "utilities": utilities_by_metric,
                    "timing_seconds": timing_seconds,
                }
            )

        all_results.append(
            {
                "video_path": str(video.path),
                "relative_path": video.relative_path,
                "frame_count": video.frame_count,
                "filename_fps": video.filename_fps,
                "metadata_fps": video.metadata_fps,
                "effective_fps": video.effective_fps,
                "fps_bucket": video.fps_bucket,
                "tiles": video_tiles,
            }
        )

    output = {
        "metadata": {
            "input_roots": args.input_roots,
            "split": args.split,
            "limit": args.limit,
            "limit_per_root": args.limit_per_root,
            "tile_len": args.tile_len,
            "tile_stride": args.tile_stride,
            "delta_semantics": "candidate_start = current_start + tile_len + delta",
            "deltas": deltas,
            "fps_buckets": fps_buckets,
            "proxy_metrics": proxy_metrics,
            "proxy_base_metric": args.proxy_base_metric,
            "lambda_cost": args.lambda_cost,
            "temporal_cost": args.temporal_cost,
            "resize": args.resize,
            "semantic_model": args.semantic_model,
            "siglip_model": args.siglip_model if any(requires_semantic_model(metric, "siglip", args.semantic_model) for metric in proxy_metrics) else None,
            "dinov2_model": args.dinov2_model if any(requires_semantic_model(metric, "dinov2", args.semantic_model) for metric in proxy_metrics) else None,
        },
        "summary": summarize_labels(all_results, proxy_metrics),
        "videos": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
