# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dry-run hierarchical coherent-tile sampling on HLVid videos.

This runner builds a top-down temporal partition of each video. Each interval is
represented by a uniformly sampled 16-frame tile, scored for incoherence, and
high-incoherence intervals are split until a target leaf budget is reached.
No NVILA generation is performed here; this is a diagnostic step before full
benchmark launches.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from autogaze.temporal_sampling import (
    PAIRWISE_METRICS,
    SEMANTIC_MODELS,
    SEMANTIC_WINDOW_METRICS,
    WINDOW_METRICS,
    Dinov2SemanticEncoder,
    SiglipSemanticEncoder,
    score_semantic_window_metric,
    score_window_metric,
    thumbnail_tensor,
)
from hlvid_common import load_rows, video_frame_count_from_capture


DEFAULT_MODEL_ROOT = "/storage/slurm/latn/data/AutoGaze/hf_models"
SUPPORTED_METRICS = tuple(WINDOW_METRICS) + tuple(SEMANTIC_WINDOW_METRICS)


@dataclass
class IntervalNode:
    start: int
    end: int
    depth: int
    score: float | None = None
    sampled_indices: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def splittable(self, tile_len: int, min_interval_frames: int, max_depth: int) -> bool:
        if self.depth >= max_depth:
            return False
        if self.length < max(min_interval_frames, 2 * tile_len):
            return False
        midpoint = (self.start + self.end) // 2
        return midpoint >= self.start and midpoint + 1 <= self.end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", default="/storage/slurm/latn/data/AutoGaze/HLVid")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--tile-len", type=int, default=16)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--target-leaf-tiles", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-interval-frames", type=int, default=64)
    parser.add_argument(
        "--split-policy",
        choices=["best_first", "level_sync"],
        default="best_first",
        help=(
            "best_first splits one highest-incoherence leaf at a time. "
            "level_sync splits all currently eligible high-scoring leaves up to the remaining budget."
        ),
    )
    parser.add_argument(
        "--coherence-metric",
        choices=SUPPORTED_METRICS,
        default="semantic_diversity",
        help="Higher scores are treated as less coherent and split first.",
    )
    parser.add_argument(
        "--proxy-base-metric",
        choices=PAIRWISE_METRICS,
        default="pixel_mse",
        help="Pairwise thumbnail distance used by non-semantic window metrics.",
    )
    parser.add_argument(
        "--semantic-model",
        choices=SEMANTIC_MODELS,
        default="dinov2",
        help="Embedding model for semantic_mean_novelty and semantic_diversity.",
    )
    parser.add_argument("--siglip-model", default=f"{DEFAULT_MODEL_ROOT}/google--siglip2-base-patch16-224")
    parser.add_argument("--dinov2-model", default=f"{DEFAULT_MODEL_ROOT}/facebook--dinov2-with-registers-base")
    parser.add_argument("--semantic-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--semantic-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max-batch-size-semantic", type=int, default=32)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def unique_videos(rows: Sequence[dict]) -> list[dict]:
    seen = {}
    for row in rows:
        seen.setdefault(row["video_path"], {"video_path": row["video_path"], "category": row["category"]})
    return list(seen.values())


def interval_indices(start: int, end: int, tile_len: int) -> list[int]:
    return np.round(np.linspace(start, end, tile_len)).astype(np.int64).clip(start, end).tolist()


class VideoFrameLoader:
    def __init__(self, video_path: Path, resize: int, preload_indices: Sequence[int] | None = None):
        import cv2

        self.cv2 = cv2
        self.video_path = video_path
        self.resize = resize
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        self.cache = {}
        self.current_position = -1
        if preload_indices:
            self.prefetch(preload_indices)

    def close(self) -> None:
        self.cap.release()

    def prefetch(self, indices: Sequence[int]) -> None:
        for index in sorted({int(index) for index in indices}):
            self._read_index(index)

    def load(self, indices: Sequence[int]) -> np.ndarray:
        frames = []
        for index in indices:
            index = int(index)
            self._read_index(index)
            if index in self.cache:
                frames.append(self.cache[index])
        if not frames:
            raise ValueError(f"Could not load frames from {self.video_path}")
        while len(frames) < len(indices):
            frames.append(frames[-1])
        return np.stack(frames)

    def _read_index(self, index: int) -> None:
        if index in self.cache:
            return
        if self.current_position >= 0 and index > self.current_position and index - self.current_position <= 180:
            while self.current_position + 1 < index:
                if not self.cap.grab():
                    break
                self.current_position += 1
        else:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            self.current_position = index - 1

        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_position = index
        frame = self.cv2.resize(frame, (self.resize, self.resize), interpolation=self.cv2.INTER_AREA)
        self.cache[index] = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)


def load_frames_by_indices(video_path: Path, indices: Sequence[int]) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames_by_index = {}
    for index in sorted({int(index) for index in indices}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok:
            frames_by_index[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    frames = [frames_by_index[int(index)] for index in indices if int(index) in frames_by_index]
    if not frames:
        raise ValueError(f"Could not load frames from {video_path}")
    while len(frames) < len(indices):
        frames.append(frames[-1])
    return np.stack(frames)


def split_halves(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = values.shape[0] // 2
    if midpoint == 0 or values.shape[0] - midpoint != midpoint:
        raise ValueError("Half-split metrics require an even, non-empty tile length.")
    return values[:midpoint], values[midpoint:]


class CoherenceScorer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.semantic_encoder = None
        if args.coherence_metric in SEMANTIC_WINDOW_METRICS:
            dtype = torch_dtype(args.semantic_dtype)
            if args.semantic_model == "siglip":
                self.semantic_encoder = SiglipSemanticEncoder(
                    args.siglip_model,
                    args.semantic_device,
                    dtype,
                    args.max_batch_size_semantic,
                )
            else:
                self.semantic_encoder = Dinov2SemanticEncoder(
                    args.dinov2_model,
                    args.semantic_device,
                    dtype,
                    args.max_batch_size_semantic,
                )

    def score(self, frames: np.ndarray) -> float:
        metric = self.args.coherence_metric
        if metric in WINDOW_METRICS:
            thumb = thumbnail_tensor(frames, self.args.resize)
            if metric == "within_diversity":
                current = candidate = thumb
            else:
                current, candidate = split_halves(thumb)
            return float(
                score_window_metric(
                    current,
                    candidate,
                    metric,
                    base_metric=self.args.proxy_base_metric,
                ).item()
            )

        embeddings = self.semantic_encoder.encode_frames(frames)
        if metric == "semantic_diversity":
            current = candidate = embeddings
        elif metric == "semantic_mean_novelty":
            current, candidate = split_halves(embeddings)
        else:
            raise ValueError(
                f"{metric} is question-conditioned and is not implemented for question-agnostic coherence."
            )
        return float(score_semantic_window_metric(current, candidate, metric).item())


def score_node(node: IntervalNode, frame_loader: VideoFrameLoader, tile_len: int, scorer: CoherenceScorer) -> None:
    node.sampled_indices = interval_indices(node.start, node.end, tile_len)
    frames = frame_loader.load(node.sampled_indices)
    node.score = scorer.score(frames)


def split_node(node: IntervalNode) -> tuple[IntervalNode, IntervalNode]:
    midpoint = (node.start + node.end) // 2
    left = IntervalNode(node.start, midpoint, node.depth + 1)
    right = IntervalNode(midpoint + 1, node.end, node.depth + 1)
    return left, right


def possible_tree_indices(
    start: int,
    end: int,
    *,
    tile_len: int,
    min_interval_frames: int,
    max_depth: int,
    depth: int = 0,
) -> list[int]:
    node = IntervalNode(start, end, depth)
    indices = interval_indices(start, end, tile_len)
    if not node.splittable(tile_len, min_interval_frames, max_depth):
        return indices
    left, right = split_node(node)
    indices.extend(
        possible_tree_indices(
            left.start,
            left.end,
            tile_len=tile_len,
            min_interval_frames=min_interval_frames,
            max_depth=max_depth,
            depth=left.depth,
        )
    )
    indices.extend(
        possible_tree_indices(
            right.start,
            right.end,
            tile_len=tile_len,
            min_interval_frames=min_interval_frames,
            max_depth=max_depth,
            depth=right.depth,
        )
    )
    return indices


def build_hierarchy(
    video_path: Path,
    frame_count: int,
    args: argparse.Namespace,
    scorer: CoherenceScorer,
) -> tuple[list[IntervalNode], list[dict]]:
    leaves = [IntervalNode(0, frame_count - 1, 0)]
    decisions = []
    preload_indices = possible_tree_indices(
        0,
        frame_count - 1,
        tile_len=args.tile_len,
        min_interval_frames=args.min_interval_frames,
        max_depth=args.max_depth,
    )
    frame_loader = VideoFrameLoader(video_path, resize=args.resize, preload_indices=preload_indices)
    try:
        score_node(leaves[0], frame_loader, args.tile_len, scorer)

        while len(leaves) < args.target_leaf_tiles:
            candidates = [
                node
                for node in leaves
                if node.score is not None and node.splittable(args.tile_len, args.min_interval_frames, args.max_depth)
            ]
            if not candidates:
                break

            remaining = args.target_leaf_tiles - len(leaves)
            candidates.sort(key=lambda node: (node.score, node.length), reverse=True)
            selected = candidates[:remaining] if args.split_policy == "level_sync" else candidates[:1]

            selected_ids = {id(node) for node in selected}
            next_leaves = []
            for node in leaves:
                if id(node) not in selected_ids:
                    next_leaves.append(node)
                    continue
                left, right = split_node(node)
                score_node(left, frame_loader, args.tile_len, scorer)
                score_node(right, frame_loader, args.tile_len, scorer)
                decisions.append(
                    {
                        "depth": node.depth,
                        "start": node.start,
                        "end": node.end,
                        "score": node.score,
                        "left_score": left.score,
                        "right_score": right.score,
                        "split_policy": args.split_policy,
                    }
                )
                next_leaves.extend([left, right])
            leaves = next_leaves
    finally:
        frame_loader.close()

    leaves.sort(key=lambda node: node.start)
    return leaves, decisions


def summarize_records(records: list[dict]) -> dict:
    leaf_counts = [record["num_leaf_tiles"] for record in records]
    frame_counts = [record["num_selected_frames"] for record in records]
    depths = [depth for record in records for depth in record["leaf_depths"]]
    scores = [score for record in records for score in record["leaf_scores"] if score is not None]
    categories = sorted({record["category"] for record in records})
    by_category = {}
    for category in categories:
        subset = [record for record in records if record["category"] == category]
        by_category[category] = {
            "num_videos": len(subset),
            "mean_leaf_tiles": float(np.mean([record["num_leaf_tiles"] for record in subset])),
            "mean_selected_frames": float(np.mean([record["num_selected_frames"] for record in subset])),
        }
    return {
        "num_videos": len(records),
        "mean_leaf_tiles": float(np.mean(leaf_counts)) if leaf_counts else None,
        "mean_selected_frames": float(np.mean(frame_counts)) if frame_counts else None,
        "mean_leaf_depth": float(np.mean(depths)) if depths else None,
        "max_leaf_depth": int(max(depths)) if depths else None,
        "mean_leaf_score": float(np.mean(scores)) if scores else None,
        "leaf_count_distribution": dict(Counter(str(value) for value in leaf_counts)),
        "by_category": by_category,
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    videos = unique_videos(load_rows(dataset_root))
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    scorer = CoherenceScorer(args)
    records = []
    with output_path.open("w") as out:
        for video in videos:
            started = time.perf_counter()
            video_path = dataset_root / "videos" / video["video_path"]
            frame_count = video_frame_count_from_capture(video_path)
            leaves, decisions = build_hierarchy(video_path, frame_count, args, scorer)
            selected_indices = [index for leaf in leaves for index in leaf.sampled_indices]
            record = {
                "video_path": video["video_path"],
                "category": video["category"],
                "frame_count": frame_count,
                "coherence_metric": args.coherence_metric,
                "proxy_base_metric": args.proxy_base_metric,
                "semantic_model": args.semantic_model if args.coherence_metric in SEMANTIC_WINDOW_METRICS else None,
                "target_leaf_tiles": args.target_leaf_tiles,
                "split_policy": args.split_policy,
                "num_leaf_tiles": len(leaves),
                "num_selected_frames": len(selected_indices),
                "selected_frame_ratio": len(selected_indices) / max(frame_count, 1),
                "leaf_intervals": [
                    {
                        "start": leaf.start,
                        "end": leaf.end,
                        "depth": leaf.depth,
                        "score": leaf.score,
                        "sampled_indices": leaf.sampled_indices,
                    }
                    for leaf in leaves
                ],
                "leaf_depths": [leaf.depth for leaf in leaves],
                "leaf_scores": [leaf.score for leaf in leaves],
                "split_decisions": decisions,
                "seconds": time.perf_counter() - started,
            }
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()

    summary = summarize_records(records)
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "tile_len": args.tile_len,
            "resize": args.resize,
            "target_leaf_tiles": args.target_leaf_tiles,
            "split_policy": args.split_policy,
            "max_depth": args.max_depth,
            "min_interval_frames": args.min_interval_frames,
            "coherence_metric": args.coherence_metric,
            "proxy_base_metric": args.proxy_base_metric,
            "semantic_model": args.semantic_model,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
