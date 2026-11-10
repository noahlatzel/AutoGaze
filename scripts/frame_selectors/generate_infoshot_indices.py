# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate fixed-budget HLVid indices with an InfoShot reimplementation.

The public InfoShot repository did not contain implementation code when this
adapter was written. The selection logic follows Algorithm 1 and the equations
in arXiv:2603.17374v1. Choices not fixed by the paper are exposed as arguments
and recorded in every output row.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS_DIR = REPO_ROOT / "scripts" / "runners"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNERS_DIR))

from hlvid_common import existing_question_ids, load_rows  # noqa: E402
from scripts.frame_selectors.common import selected_index_record, video_metadata  # noqa: E402


DEFAULT_DATASET_ROOT = "/storage/slurm/latn/data/AutoGaze/HLVid"
PAPER_ID = "arXiv:2603.17374v1"
UPSTREAM_COMMIT = "ab20b22fcb48fe865af72b98d3688f24b15052ba"


@dataclass(frozen=True)
class Segment:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--method", default="infoshot_reimpl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned workload without loading a model.")

    parser.add_argument("--candidate-fps", type=float, default=10.0)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--resnet-weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--boundary-window", type=int, default=None)
    parser.add_argument("--boundary-window-min", type=int, default=3)
    parser.add_argument("--boundary-window-max", type=int, default=15)
    parser.add_argument("--min-segment-frames", type=int, default=3)
    parser.add_argument("--max-segment-frames", type=int, default=300)
    parser.add_argument("--neighborhood", type=int, default=1)
    parser.add_argument("--common-weight", type=float, default=0.7)
    parser.add_argument("--unique-weight", type=float, default=0.5)
    return parser.parse_args()


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    span = float(values.max() - values.min())
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span


def candidate_source_indices(frame_count: int, source_fps: float, candidate_fps: float) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if source_fps <= 0 or candidate_fps <= 0:
        raise ValueError("source_fps and candidate_fps must be positive")
    duration = max((frame_count - 1) / source_fps, 0.0)
    times = np.arange(0.0, duration + 0.5 / candidate_fps, 1.0 / candidate_fps)
    indices = np.rint(times * source_fps).astype(np.int64).clip(0, frame_count - 1)
    unique = np.unique(indices).tolist()
    if unique[-1] != frame_count - 1:
        unique.append(frame_count - 1)
    return [int(index) for index in unique]


def cosine_affinity(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [num_candidates, feature_dim]")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def scaled_boundary_window(
    segment_length: int,
    *,
    fixed_window: int | None,
    min_window: int,
    max_window: int,
    min_segment_frames: int,
    max_segment_frames: int,
) -> int:
    if fixed_window is not None:
        return max(1, int(fixed_window))
    if max_segment_frames <= min_segment_frames:
        return max(1, int(min_window))
    fraction = (segment_length - min_segment_frames) / (max_segment_frames - min_segment_frames)
    fraction = min(max(float(fraction), 0.0), 1.0)
    return max(1, int(round(min_window + fraction * (max_window - min_window))))


def boundary_score(affinity: np.ndarray, split: int, window: int) -> float:
    left = affinity[split - window : split, split - window : split]
    right = affinity[split : split + window, split : split + window]
    cross = affinity[split - window : split, split : split + window]
    return 0.5 * (float(left.mean()) + float(right.mean())) - float(cross.mean())


def best_segment_split(
    affinity: np.ndarray,
    segment: Segment,
    *,
    fixed_window: int | None,
    min_window: int,
    max_window: int,
    min_segment_frames: int,
    max_segment_frames: int,
) -> tuple[float, int] | None:
    window = scaled_boundary_window(
        segment.length,
        fixed_window=fixed_window,
        min_window=min_window,
        max_window=max_window,
        min_segment_frames=min_segment_frames,
        max_segment_frames=max_segment_frames,
    )
    child_floor = max(min_segment_frames, window)
    first = segment.start + child_floor
    last = segment.end - child_floor
    if first > last:
        return None
    scores = [(boundary_score(affinity, split, window), split) for split in range(first, last + 1)]
    return max(scores, key=lambda item: (item[0], -item[1]))


def greedy_segments(
    affinity: np.ndarray,
    num_segments: int,
    *,
    fixed_window: int | None = None,
    min_window: int = 3,
    max_window: int = 15,
    min_segment_frames: int = 3,
    max_segment_frames: int = 300,
) -> list[Segment]:
    total = int(affinity.shape[0])
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if total < num_segments * min_segment_frames:
        raise ValueError(
            f"Need at least {num_segments * min_segment_frames} candidates for "
            f"{num_segments} segments, got {total}."
        )

    segments = [Segment(0, total)]
    while len(segments) < num_segments:
        candidates = []
        for index, segment in enumerate(segments):
            best = best_segment_split(
                affinity,
                segment,
                fixed_window=fixed_window,
                min_window=min_window,
                max_window=max_window,
                min_segment_frames=min_segment_frames,
                max_segment_frames=max_segment_frames,
            )
            if best is not None:
                score, split = best
                candidates.append((segment.length > max_segment_frames, score, -segment.start, index, split))
        if not candidates:
            raise ValueError(f"Could only form {len(segments)} valid segments; requested {num_segments}.")
        if any(candidate[0] for candidate in candidates):
            candidates = [candidate for candidate in candidates if candidate[0]]
        _, _, _, segment_index, split = max(candidates)
        segment = segments[segment_index]
        segments[segment_index : segment_index + 1] = [
            Segment(segment.start, split),
            Segment(split, segment.end),
        ]
    return segments


def dual_keyframes(
    affinity: np.ndarray,
    segment: Segment,
    *,
    neighborhood: int,
    common_weight: float,
    unique_weight: float,
) -> list[int]:
    positions = np.arange(segment.start, segment.end)
    length = len(positions)
    if length == 1:
        return [int(positions[0])]

    local_affinity = affinity[segment.start : segment.end, segment.start : segment.end]
    typicality = (local_affinity.sum(axis=1) - np.diag(local_affinity)) / max(length - 1, 1)
    volatility = np.zeros(length, dtype=np.float32)
    for local_index in range(length):
        left = max(0, local_index - neighborhood)
        right = min(length, local_index + neighborhood + 1)
        neighbors = [index for index in range(left, right) if index != local_index]
        if neighbors:
            volatility[local_index] = 1.0 - float(local_affinity[local_index, neighbors].mean())

    typicality_hat = minmax_normalize(typicality)
    volatility_hat = minmax_normalize(volatility)
    common_scores = common_weight * typicality_hat - (1.0 - common_weight) * volatility_hat
    common_local = int(np.argmax(common_scores))

    unique_scores = unique_weight * (1.0 - typicality_hat) + (1.0 - unique_weight) * volatility_hat
    unique_scores[common_local] = -np.inf
    unique_local = int(np.argmax(unique_scores))
    return [int(positions[common_local]), int(positions[unique_local])]


def repair_exact_budget(selected: Iterable[int], affinity: np.ndarray, budget: int) -> tuple[list[int], int]:
    total = int(affinity.shape[0])
    if budget <= 0:
        raise ValueError("budget must be positive")
    if total < budget:
        raise ValueError(f"Cannot select {budget} unique frames from {total} candidates")

    selected_set = {int(index) for index in selected}
    if len(selected_set) > budget:
        raise ValueError("InfoShot produced more frames than the requested budget")
    added = 0
    while len(selected_set) < budget:
        if selected_set:
            covered_similarity = affinity[:, sorted(selected_set)].max(axis=1)
            scores = 1.0 - covered_similarity
        else:
            scores = np.ones(total, dtype=np.float32)
        scores[list(selected_set)] = -np.inf
        selected_set.add(int(np.argmax(scores)))
        added += 1
    return sorted(selected_set), added


def infoshot_select(
    features: np.ndarray,
    budget: int,
    *,
    fixed_window: int | None = None,
    min_window: int = 3,
    max_window: int = 15,
    min_segment_frames: int = 3,
    max_segment_frames: int = 300,
    neighborhood: int = 1,
    common_weight: float = 0.7,
    unique_weight: float = 0.5,
) -> tuple[list[int], list[Segment], int]:
    affinity = cosine_affinity(features)
    segment_count = max(1, budget // 2)
    segments = greedy_segments(
        affinity,
        segment_count,
        fixed_window=fixed_window,
        min_window=min_window,
        max_window=max_window,
        min_segment_frames=min_segment_frames,
        max_segment_frames=max_segment_frames,
    )
    selected = []
    for segment in segments:
        remaining = budget - len(selected)
        if remaining <= 0:
            break
        pair = dual_keyframes(
            affinity,
            segment,
            neighborhood=neighborhood,
            common_weight=common_weight,
            unique_weight=unique_weight,
        )
        selected.extend(pair[:remaining])
    selected, repair_count = repair_exact_budget(selected, affinity, budget)
    return selected, segments, repair_count


def torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


class ResNet50Encoder:
    def __init__(self, *, weights_path: Path | None, device: str, dtype: str) -> None:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        if weights_path is None:
            model = resnet50(weights=weights)
        else:
            model = resnet50(weights=None)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        model.fc = torch.nn.Identity()

        self.torch = torch
        self.device = device
        self.dtype = torch_dtype(dtype)
        self.model = model.eval().to(device=device, dtype=self.dtype)
        self.transform = weights.transforms()

    def _encode_batch(self, frames: Sequence) -> np.ndarray:
        pixels = self.torch.stack([self.transform(frame) for frame in frames])
        pixels = pixels.to(device=self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            features = self.model(pixels)
            features = self.torch.nn.functional.normalize(features.float(), dim=-1)
        return features.cpu().numpy()

    def encode_video(
        self,
        video_path: Path,
        source_indices: Sequence[int],
        batch_size: int,
    ) -> tuple[np.ndarray, list[int]]:
        import av

        wanted = set(int(index) for index in source_indices)
        batch_frames = []
        batch_indices = []
        feature_chunks = []
        decoded_indices: list[int] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index not in wanted:
                    continue
                batch_frames.append(frame.to_image())
                batch_indices.append(frame_index)
                if len(batch_frames) >= batch_size:
                    feature_chunks.append(self._encode_batch(batch_frames))
                    decoded_indices.extend(batch_indices)
                    batch_frames = []
                    batch_indices = []
                if len(decoded_indices) + len(batch_indices) == len(wanted):
                    break
        if batch_frames:
            feature_chunks.append(self._encode_batch(batch_frames))
            decoded_indices.extend(batch_indices)
        if not feature_chunks:
            raise ValueError(f"No candidate frames decoded from {video_path}")
        return np.concatenate(feature_chunks, axis=0), decoded_indices


def feature_cache_path(cache_dir: Path, relative_video_path: str) -> Path:
    return cache_dir / Path(relative_video_path).with_suffix(".npz")


def load_or_extract_features(
    *,
    encoder: ResNet50Encoder,
    video_path: Path,
    relative_video_path: str,
    source_indices: Sequence[int],
    batch_size: int,
    cache_dir: Path | None,
) -> tuple[np.ndarray, list[int], bool]:
    cache_path = feature_cache_path(cache_dir, relative_video_path) if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        cached_indices = cached["source_indices"].astype(int).tolist()
        if cached_indices != [int(index) for index in source_indices]:
            raise ValueError(
                f"Cached candidate timeline does not match current settings: {cache_path}. "
                "Use a separate cache directory for each candidate FPS and encoder."
            )
        features = cached["features"].astype(np.float32)
        if features.shape[0] != len(cached_indices):
            raise ValueError(f"Feature/index count mismatch in cache: {cache_path}")
        return features, cached_indices, True

    features, decoded_indices = encoder.encode_video(video_path, source_indices, batch_size)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=features.astype(np.float16),
            source_indices=np.asarray(decoded_indices, dtype=np.int64),
        )
    return features, decoded_indices, False


def validate_args(args: argparse.Namespace) -> None:
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.candidate_fps <= 0 or args.batch_size <= 0:
        raise ValueError("--candidate-fps and --batch-size must be positive")
    if args.min_segment_frames <= 0 or args.max_segment_frames < args.min_segment_frames:
        raise ValueError("Invalid segment-length constraints")
    if args.boundary_window_min <= 0 or args.boundary_window_max < args.boundary_window_min:
        raise ValueError("Invalid boundary-window range")
    if args.neighborhood <= 0:
        raise ValueError("--neighborhood must be positive")
    for name in ("common_weight", "unique_weight"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    rows = load_rows(args.dataset_root)[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]
    done_ids = {str(question_id) for question_id in existing_question_ids(args.output)} if args.resume else set()
    rows = [row for row in rows if str(row["question_id"]) not in done_ids]

    rows_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_video.setdefault(str(row["video_path"]), []).append(row)
    if args.dry_run:
        print(json.dumps({"rows": len(rows), "unique_videos": len(rows_by_video), "output": str(args.output)}))
        return
    if not rows_by_video:
        print("No HLVid rows remain to process.")
        return

    encoder = ResNet50Encoder(weights_path=args.resnet_weights, device=args.device, dtype=args.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode) as output_file:
        for relative_video_path, video_rows in tqdm(rows_by_video.items(), desc="InfoShot videos"):
            video_path = args.dataset_root / "videos" / relative_video_path
            frame_count, source_fps = video_metadata(video_path)
            candidates = candidate_source_indices(frame_count, source_fps, args.candidate_fps)
            features, decoded_indices, cache_hit = load_or_extract_features(
                encoder=encoder,
                video_path=video_path,
                relative_video_path=relative_video_path,
                source_indices=candidates,
                batch_size=args.batch_size,
                cache_dir=args.feature_cache_dir,
            )
            selected_positions, segments, repair_count = infoshot_select(
                features,
                args.num_frames,
                fixed_window=args.boundary_window,
                min_window=args.boundary_window_min,
                max_window=args.boundary_window_max,
                min_segment_frames=args.min_segment_frames,
                max_segment_frames=args.max_segment_frames,
                neighborhood=args.neighborhood,
                common_weight=args.common_weight,
                unique_weight=args.unique_weight,
            )
            selected_source_indices = [decoded_indices[position] for position in selected_positions]
            metadata = {
                "implementation": "independent paper reimplementation",
                "paper": PAPER_ID,
                "upstream_commit": UPSTREAM_COMMIT,
                "feature_encoder": "torchvision_resnet50_imagenet1k_v2",
                "candidate_fps": args.candidate_fps,
                "candidate_count": len(decoded_indices),
                "segment_count": len(segments),
                "boundary_window": args.boundary_window,
                "boundary_window_min": args.boundary_window_min,
                "boundary_window_max": args.boundary_window_max,
                "boundary_window_scaling": "linear_by_segment_length",
                "min_segment_frames": args.min_segment_frames,
                "max_segment_frames": args.max_segment_frames,
                "neighborhood": args.neighborhood,
                "common_weight": args.common_weight,
                "unique_weight": args.unique_weight,
                "exact_budget_repair_count": repair_count,
                "feature_cache_hit": cache_hit,
            }
            for row in video_rows:
                record = selected_index_record(
                    question_id=str(row["question_id"]),
                    video_path=relative_video_path,
                    method=args.method,
                    num_frames=args.num_frames,
                    selected_source_indices=selected_source_indices,
                    frame_count=frame_count,
                    metadata=metadata,
                )
                output_file.write(json.dumps(record) + "\n")
                output_file.flush()


if __name__ == "__main__":
    main()
