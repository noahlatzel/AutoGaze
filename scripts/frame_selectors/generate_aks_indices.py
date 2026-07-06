# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate HLVid selected-frame JSONL with AKS-style temporal selection.

This adapter keeps the benchmark interface local to AutoGaze while reusing the
selection logic from external/frame_selectors/AKS/frame_select.py. Feature
scoring can be supplied as upstream AKS-format JSON files or computed with CLIP
for small smoke runs.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.frame_selectors.common import load_pil_frames_av, model_path, selected_index_record, video_metadata


DEFAULT_DATASET_ROOT = "/storage/slurm/latn/data/AutoGaze/HLVid"
DEFAULT_OUTPUT = "outputs/frame_indices/aks_128.jsonl"


@dataclass
class Segment:
    scores: np.ndarray
    frames: list[int]
    depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AKS frame-index JSONL for HLVid.")
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--method", default="aks")
    parser.add_argument(
        "--score-backend",
        choices=("precomputed", "clip", "dry_uniform"),
        default="precomputed",
        help=(
            "precomputed reads --scores-json/--frames-json in upstream AKS list-of-lists format; "
            "clip scores one candidate per second with the registry CLIP model; "
            "dry_uniform is only for schema smoke tests."
        ),
    )
    parser.add_argument("--scores-json", type=Path, help="Upstream AKS scores.json list aligned to HLVid rows.")
    parser.add_argument("--frames-json", type=Path, help="Upstream AKS frames.json list aligned to HLVid rows.")
    parser.add_argument("--clip-model", default=model_path("clip_vit_l14"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-stride-seconds", type=float, default=1.0)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--t1", type=float, default=0.8)
    parser.add_argument("--t2", type=float, default=-100.0)
    parser.add_argument("--all-depth", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Optional smoke-test row limit.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-dry-output",
        action="store_true",
        help="Permit writing method=aks rows with dry_uniform scores. Intended only for plumbing tests.",
    )
    return parser.parse_args()


def load_hlvid_rows(dataset_root: Path) -> list[dict[str, Any]]:
    parquet_path = dataset_root / "data" / "test-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing HLVid metadata parquet: {parquet_path}")
    return pq.read_table(parquet_path).to_pylist()


def existing_question_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    question_ids = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                question_ids.add(str(json.loads(line)["question_id"]))
    return question_ids


def normalize_scores(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("AKS requires at least one candidate score")
    span = float(values.max() - values.min())
    if span <= 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def split_segments(segments: Sequence[Segment], budget: int, t1: float, t2: float, all_depth: int) -> list[Segment]:
    split: list[Segment] = []
    keep: list[Segment] = []

    for segment in segments:
        scores = segment.scores
        top_n = heapq.nlargest(min(budget, len(scores)), range(len(scores)), scores.__getitem__)
        mean_diff = float(np.mean([scores[index] for index in top_n]) - np.mean(scores))
        std = float(np.std(scores))
        if mean_diff > t1 and std > t2:
            keep.append(segment)
        elif segment.depth < all_depth and len(scores) > 1:
            mid = len(scores) // 2
            split.append(Segment(scores[:mid], segment.frames[:mid], segment.depth + 1))
            split.append(Segment(scores[mid:], segment.frames[mid:], segment.depth + 1))
        else:
            keep.append(segment)

    if split:
        keep.extend(split_segments(split, budget, t1, t2, all_depth))
    return keep


def repair_to_fixed_count(
    selected: Iterable[int],
    candidate_frames: Sequence[int],
    candidate_scores: Sequence[float],
    frame_count: int,
    num_frames: int,
) -> list[int]:
    selected_set = {int(frame) for frame in selected}
    ranked_candidates = [
        int(frame)
        for _, frame in sorted(
            zip(candidate_scores, candidate_frames),
            key=lambda item: (float(item[0]), int(item[1])),
            reverse=True,
        )
    ]
    for frame in ranked_candidates:
        if len(selected_set) >= num_frames:
            break
        selected_set.add(frame)

    if len(selected_set) < num_frames:
        uniform = np.round(np.linspace(0, max(frame_count - 1, 0), num_frames)).astype(int).tolist()
        for frame in uniform:
            if len(selected_set) >= num_frames:
                break
            selected_set.add(int(frame))

    fixed = sorted(selected_set)[:num_frames]
    while fixed and len(fixed) < num_frames:
        fixed.append(fixed[-1])
    if not fixed:
        fixed = [0] * num_frames
    return fixed


def aks_select(
    scores: Sequence[float],
    frames: Sequence[int],
    frame_count: int,
    num_frames: int,
    ratio: int,
    t1: float,
    t2: float,
    all_depth: int,
) -> list[int]:
    if ratio <= 0:
        raise ValueError("--ratio must be positive")
    paired = [(float(score), int(frame)) for score, frame in zip(scores, frames)][::ratio]
    if not paired:
        return repair_to_fixed_count([], [], [], frame_count, num_frames)

    sampled_scores = [score for score, _ in paired]
    sampled_frames = [max(0, min(int(frame), frame_count - 1)) for _, frame in paired]
    if len(sampled_scores) <= num_frames:
        return repair_to_fixed_count(sampled_frames, sampled_frames, sampled_scores, frame_count, num_frames)

    normalized = normalize_scores(sampled_scores)
    segments = split_segments([Segment(normalized, sampled_frames, 0)], num_frames, t1, t2, all_depth)
    selected: list[int] = []
    for segment in segments:
        segment_budget = max(1, int(num_frames / (2**segment.depth)))
        topk = heapq.nlargest(
            min(segment_budget, len(segment.scores)),
            range(len(segment.scores)),
            segment.scores.__getitem__,
        )
        selected.extend(segment.frames[index] for index in topk)
    return repair_to_fixed_count(selected, sampled_frames, sampled_scores, frame_count, num_frames)


def video_frame_count(video_path: Path) -> int:
    return video_metadata(video_path)[0]


def dry_uniform_scores(video_path: Path, num_candidates: int = 256) -> tuple[list[float], list[int], int]:
    frame_count = video_frame_count(video_path)
    frames = np.round(np.linspace(0, frame_count - 1, min(num_candidates, frame_count))).astype(int).tolist()
    scores = np.linspace(0.0, 1.0, len(frames)).tolist()
    return scores, frames, frame_count


class ClipScorer:
    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def score_video(self, video_path: Path, question: str, stride_seconds: float) -> tuple[list[float], list[int], int]:
        frame_count, fps = video_metadata(video_path)
        stride = max(1, int(round(fps * stride_seconds)))
        candidate_frames = list(range(0, max(frame_count, 1), stride))

        text_inputs = self.processor(text=question, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)

        scores: list[float] = []
        decoded_frames = load_pil_frames_av(video_path, candidate_frames)
        frames = candidate_frames[: len(decoded_frames)]
        for image in decoded_frames:
            image_inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
            score = self.torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
            scores.append(float(score.item()))
        if not scores:
            raise ValueError(f"No CLIP candidate frames decoded from {video_path}")
        return scores, frames, frame_count


def row_question(row: dict[str, Any]) -> str:
    for key in ("question", "prompt"):
        if row.get(key):
            return str(row[key])
    return json.dumps(row, sort_keys=True)


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.score_backend == "dry_uniform" and not args.allow_dry_output:
        raise ValueError("dry_uniform is for plumbing only; pass --allow-dry-output to write rows")
    if args.score_backend == "precomputed" and (args.scores_json is None or args.frames_json is None):
        raise ValueError("precomputed mode requires --scores-json and --frames-json")

    rows = load_hlvid_rows(args.dataset_root)
    if args.limit is not None:
        rows = rows[: args.limit]

    precomputed_scores = precomputed_frames = None
    if args.score_backend == "precomputed":
        precomputed_scores = json.loads(args.scores_json.read_text())
        precomputed_frames = json.loads(args.frames_json.read_text())
        if len(precomputed_scores) < len(rows) or len(precomputed_frames) < len(rows):
            raise ValueError("Precomputed AKS score/frame lists are shorter than selected HLVid rows")

    clip_scorer = None
    if args.score_backend == "clip":
        clip_scorer = ClipScorer(args.clip_model, args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = existing_question_ids(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"
    with args.output.open(mode) as f:
        for row_index, row in enumerate(rows):
            question_id = str(row["question_id"])
            if question_id in done_ids:
                continue
            video_path = args.dataset_root / "videos" / row["video_path"]
            if args.score_backend == "precomputed":
                frame_count = video_frame_count(video_path)
                scores = precomputed_scores[row_index]
                frames = precomputed_frames[row_index]
            elif args.score_backend == "clip":
                assert clip_scorer is not None
                scores, frames, frame_count = clip_scorer.score_video(
                    video_path,
                    row_question(row),
                    args.candidate_stride_seconds,
                )
            else:
                scores, frames, frame_count = dry_uniform_scores(video_path)

            selected = aks_select(
                scores,
                frames,
                frame_count,
                args.num_frames,
                args.ratio,
                args.t1,
                args.t2,
                args.all_depth,
            )
            record = selected_index_record(
                question_id=question_id,
                video_path=str(row["video_path"]),
                method=args.method,
                num_frames=args.num_frames,
                selected_source_indices=selected,
                frame_count=frame_count,
                metadata={
                    "score_backend": args.score_backend,
                    "ratio": args.ratio,
                    "t1": args.t1,
                    "t2": args.t2,
                    "all_depth": args.all_depth,
                },
            )
            f.write(json.dumps(record) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
