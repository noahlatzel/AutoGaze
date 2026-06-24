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

"""Evaluate HLVid/NVILA under uniform and proxy-guided frame sampling."""

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

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
from generate_delta_labels import (
    SiglipTileEncoder,
    VideoMAEReconstructionOracle,
    cosine_distance,
    torch_dtype,
)


ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
DEFAULT_DELTAS = "1,2,4,8,16,32,64,128,192,256,320"
DEFAULT_MODEL_ROOT = "/storage/slurm/latn/data/AutoGaze/hf_models"
DEFAULT_VIDEOMAE_WEIGHTS = "/storage/slurm/latn/data/AutoGaze/VideoMAE_AutoGaze/videomae.pt"
QUESTION_CONDITIONED_METRICS = {"siglip_qgain", "question_relevance"}


def cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimingProbe:
    def __init__(self):
        self.seconds = Counter()
        self.counts = Counter()

    def add_seconds(self, key: str, seconds: float) -> None:
        self.seconds[key] += float(seconds)

    def add_count(self, key: str, count: int | float) -> None:
        self.counts[key] += count

    def timed(self, key: str, fn, *args, sync_cuda: bool = False, **kwargs):
        if sync_cuda:
            cuda_synchronize()
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        if sync_cuda:
            cuda_synchronize()
        self.add_seconds(key, time.perf_counter() - started)
        return result

    def reset(self) -> None:
        self.seconds.clear()
        self.counts.clear()

    def snapshot_seconds(self) -> Dict[str, float]:
        return dict(self.seconds)

    def snapshot_counts(self) -> Dict[str, int | float]:
        return dict(self.counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", default="/storage/slurm/latn/data/AutoGaze/HLVid")
    parser.add_argument("--model-path", default=f"{DEFAULT_MODEL_ROOT}/nvidia--NVILA-8B-HD-Video")
    parser.add_argument("--autogaze-model-id", default=f"{DEFAULT_MODEL_ROOT}/nvidia--AutoGaze")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-sampling", action="store_true")

    parser.add_argument("--sampling", required=True, help="'dense' or a fixed sampled-frame count such as 128.")
    parser.add_argument("--tile-len", type=int, default=16)
    parser.add_argument("--selection", choices=["uniform", "proxy"], default="uniform")
    parser.add_argument(
        "--proxy-metric",
        choices=(
            list(PAIRWISE_METRICS)
            + list(WINDOW_METRICS)
            + list(SEMANTIC_PAIRWISE_METRICS)
            + list(SEMANTIC_WINDOW_METRICS)
            + ["siglip", "videomae_reconstruction"]
        ),
        default=None,
        help=(
            "Proxy metric used when --selection proxy. Pairwise thumbnail and semantic cosine metrics "
            "are aggregated with mean_pairwise_change over candidate chunks."
        ),
    )
    parser.add_argument(
        "--proxy-base-metric",
        choices=list(PAIRWISE_METRICS),
        default="pixel_mse",
        help="Pairwise thumbnail distance used by window metrics.",
    )
    parser.add_argument("--lambda-cost", type=float, default=0.0, help="Weight for temporal action cost.")
    parser.add_argument(
        "--temporal-cost",
        choices=list(TEMPORAL_COSTS),
        default="none",
        help="Temporal action cost used when selecting the best candidate.",
    )
    parser.add_argument("--deltas", default=DEFAULT_DELTAS)

    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument(
        "--max-tiles-video",
        type=int,
        default=1,
        help=(
            "NVILA max spatial tiles per temporal chunk. This is not the number of temporal tiles; "
            "keep it small for high-frame-count runs."
        ),
    )
    parser.add_argument(
        "--max-dense-sampled-frames",
        type=int,
        default=None,
        help=(
            "Optional safety cap for dense mode. True dense HLVid can require thousands of 4K frames "
            "in one NVILA call, so use this for a feasible dense approximation."
        ),
    )
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])

    parser.add_argument("--proxy-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--proxy-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--feature-frame-stride", type=int, default=4)
    parser.add_argument("--siglip-model", default=f"{DEFAULT_MODEL_ROOT}/google--siglip2-base-patch16-224")
    parser.add_argument("--dinov2-model", default=f"{DEFAULT_MODEL_ROOT}/facebook--dinov2-with-registers-base")
    parser.add_argument(
        "--semantic-model",
        choices=list(SEMANTIC_MODELS),
        default="siglip",
        help="Embedding model used by semantic_mean_novelty and semantic_diversity.",
    )
    parser.add_argument("--max-batch-size-semantic", type=int, default=32)
    parser.add_argument("--videomae-model", default=f"{DEFAULT_MODEL_ROOT}/facebook--vit-mae-large")
    parser.add_argument("--videomae-weights", default=DEFAULT_VIDEOMAE_WEIGHTS)
    parser.add_argument("--videomae-scales", default="224")
    parser.add_argument("--videomae-patch-size", type=int, default=16)
    parser.add_argument("--videomae-gaze-ratio", type=float, default=0.25)
    parser.add_argument("--videomae-attn-mode", default="sdpa")
    return parser.parse_args()


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def extract_letter(text: str) -> str:
    match = ANSWER_RE.search(text.strip())
    return match.group(1).upper() if match else ""


def load_rows(dataset_root: Path) -> List[dict]:
    parquet_path = dataset_root / "data" / "test-00000-of-00001.parquet"
    return pq.read_table(parquet_path).to_pylist()


def existing_question_ids(path: Path) -> set:
    ids = set()
    if not path.exists():
        return ids
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["question_id"])
            except Exception:
                continue
    return ids


def video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if cap.grab():
            break
        frame_count -= 1
    cap.release()
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    return frame_count


def video_frame_count_from_capture(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        return video_frame_count(video_path)
    return frame_count


def sampled_count(
    sampling: str,
    frame_count: int,
    tile_len: int,
    max_dense_sampled_frames: Optional[int] = None,
) -> int:
    if sampling == "dense":
        count = (frame_count // tile_len) * tile_len
        count = max(tile_len, count) if frame_count >= tile_len else tile_len
        if max_dense_sampled_frames is not None:
            max_dense = (max_dense_sampled_frames // tile_len) * tile_len
            if max_dense < tile_len:
                raise ValueError("--max-dense-sampled-frames must be at least tile_len")
            count = min(count, max_dense)
        return count
    count = int(sampling)
    if count <= 0:
        raise ValueError("--sampling must be 'dense' or a positive integer")
    if count % tile_len != 0:
        raise ValueError(f"Fixed --sampling values must be multiples of tile_len={tile_len}: {count}")
    return count


def max_sampled_count_for_rows(
    rows: Sequence[dict],
    dataset_root: Path,
    sampling: str,
    tile_len: int,
    max_dense_sampled_frames: Optional[int],
) -> int:
    if sampling != "dense":
        return sampled_count(sampling, 1, tile_len, max_dense_sampled_frames)
    counts = []
    for row in tqdm(rows, desc="scan frame counts"):
        video_path = dataset_root / "videos" / row["video_path"]
        counts.append(
            sampled_count(
                sampling,
                video_frame_count_from_capture(video_path),
                tile_len,
                max_dense_sampled_frames,
            )
        )
    return max(counts) if counts else tile_len


def uniform_indices(frame_count: int, num_samples: int) -> np.ndarray:
    return np.round(np.linspace(0, frame_count - 1, num_samples)).astype(np.int64)


def load_video_samples(
    video_path: Path,
    sampling: str,
    tile_len: int,
    max_dense_sampled_frames: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    frame_count = video_frame_count(video_path)
    indices = uniform_indices(frame_count, sampled_count(sampling, frame_count, tile_len, max_dense_sampled_frames))

    cap = cv2.VideoCapture(str(video_path))
    frames_by_index = {}
    for index in sorted({int(index) for index in indices}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        frames_by_index[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    frames = [frames_by_index[int(index)] for index in indices if int(index) in frames_by_index]
    if not frames:
        raise ValueError(f"Could not extract frames from video: {video_path}")
    while len(frames) < len(indices):
        frames.append(frames[-1])
    return np.stack(frames), indices, frame_count


def uniform_tile_starts(num_samples: int, tile_len: int) -> List[int]:
    return list(range(0, num_samples - tile_len + 1, tile_len))


class ProxyScorer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.siglip_encoder = None
        self.semantic_encoder = None
        self.videomae_oracle = None
        dtype = torch_dtype(args.proxy_dtype)
        if args.proxy_metric == "siglip":
            self.siglip_encoder = SiglipTileEncoder(
                args.siglip_model,
                args.proxy_device,
                dtype,
                args.max_batch_size_siglip,
            )
        if self._needs_semantic_encoder():
            semantic_model = self._semantic_model_name()
            if semantic_model == "siglip":
                self.semantic_encoder = SiglipSemanticEncoder(
                    args.siglip_model,
                    args.proxy_device,
                    dtype,
                    args.max_batch_size_semantic,
                )
            elif semantic_model == "dinov2":
                self.semantic_encoder = Dinov2SemanticEncoder(
                    args.dinov2_model,
                    args.proxy_device,
                    dtype,
                    args.max_batch_size_semantic,
                )
            else:
                raise ValueError(f"Unknown semantic model: {semantic_model}")
        if args.proxy_metric == "videomae_reconstruction":
            self.videomae_oracle = VideoMAEReconstructionOracle(
                model_name_or_path=args.videomae_model,
                weights_path=args.videomae_weights,
                scales=args.videomae_scales,
                patch_size=args.videomae_patch_size,
                gaze_ratio=args.videomae_gaze_ratio,
                attn_mode=args.videomae_attn_mode,
                device=args.proxy_device,
                dtype=dtype,
            )

    def _needs_semantic_encoder(self) -> bool:
        return self.args.proxy_metric in SEMANTIC_PAIRWISE_METRICS + SEMANTIC_WINDOW_METRICS

    def _semantic_model_name(self) -> str:
        if self.args.proxy_metric in ("siglip_cosine", "siglip_qgain", "question_relevance"):
            return "siglip"
        if self.args.proxy_metric == "dinov2_cosine":
            return "dinov2"
        return self.args.semantic_model

    def score(
        self,
        proxy: str,
        current_start: int,
        candidate_start: int,
        frames: np.ndarray,
        tile_len: int,
        tile_cache: Dict[int, np.ndarray],
        resized_cache: Dict[int, torch.Tensor],
        siglip_cache: Dict[int, torch.Tensor],
        semantic_cache: Dict[int, torch.Tensor],
        text_cache: Dict[str, torch.Tensor],
        videomae_cache: Dict[int, float],
        question: Optional[str] = None,
    ) -> float:
        def get_tile(start: int) -> np.ndarray:
            if start not in tile_cache:
                tile_cache[start] = frames[start : start + tile_len]
            return tile_cache[start]

        def get_resized(start: int) -> torch.Tensor:
            if start not in resized_cache:
                resized_cache[start] = thumbnail_tensor(get_tile(start), self.args.resize)
            return resized_cache[start]

        def get_siglip(start: int) -> torch.Tensor:
            if start not in siglip_cache:
                siglip_cache[start] = self.siglip_encoder.encode(get_tile(start), self.args.feature_frame_stride)
            return siglip_cache[start]

        def get_semantic(start: int) -> torch.Tensor:
            if start not in semantic_cache:
                semantic_cache[start] = self.semantic_encoder.encode_frames(get_tile(start))
            return semantic_cache[start]

        def get_question_embedding() -> torch.Tensor:
            if question is None:
                raise ValueError(f"{proxy} requires a question")
            if not isinstance(self.semantic_encoder, SiglipSemanticEncoder):
                raise ValueError(f"{proxy} requires the SigLIP semantic encoder")
            if question not in text_cache:
                text_cache[question] = self.semantic_encoder.encode_text(question)
            return text_cache[question]

        def get_videomae(start: int) -> float:
            if start not in videomae_cache:
                videomae_cache[start] = self.videomae_oracle.score(get_tile(start))
            return videomae_cache[start]

        if proxy in PAIRWISE_METRICS:
            return float(
                score_window_metric(
                    get_resized(current_start),
                    get_resized(candidate_start),
                    "mean_pairwise_change",
                    base_metric=proxy,
                ).item()
            )
        if proxy in WINDOW_METRICS:
            return float(
                score_window_metric(
                    get_resized(current_start),
                    get_resized(candidate_start),
                    proxy,
                    base_metric=self.args.proxy_base_metric,
                ).item()
            )
        if proxy in SEMANTIC_PAIRWISE_METRICS:
            question_embedding = get_question_embedding() if proxy in QUESTION_CONDITIONED_METRICS else None
            return float(
                score_semantic_pairwise_metric(
                    get_semantic(current_start),
                    get_semantic(candidate_start),
                    proxy,
                    question_embedding=question_embedding,
                ).item()
            )
        if proxy in SEMANTIC_WINDOW_METRICS:
            question_embedding = get_question_embedding() if proxy in QUESTION_CONDITIONED_METRICS else None
            return float(
                score_semantic_window_metric(
                    get_semantic(current_start),
                    get_semantic(candidate_start),
                    proxy,
                    question_embedding=question_embedding,
                ).item()
            )
        if proxy == "siglip":
            return cosine_distance(get_siglip(current_start), get_siglip(candidate_start))
        if proxy == "videomae_reconstruction":
            return get_videomae(candidate_start)
        raise ValueError(f"Unknown proxy: {proxy}")


def install_processor_timing(processor, probe: TimingProbe) -> None:
    if getattr(processor, "_autogaze_timing_installed", False):
        return

    if hasattr(processor, "_preprocess_videos"):
        original_preprocess_videos = processor._preprocess_videos

        def timed_preprocess_videos(self, *args, **kwargs):
            return probe.timed("nvila_preprocess_videos", original_preprocess_videos, *args, **kwargs)

        processor._preprocess_videos = MethodType(timed_preprocess_videos, processor)

    if hasattr(processor, "_get_gazing_info_from_videos"):
        original_get_gazing_info = processor._get_gazing_info_from_videos

        def timed_get_gazing_info(self, *args, **kwargs):
            result = probe.timed(
                "autogaze_processing",
                original_get_gazing_info,
                *args,
                sync_cuda=True,
                **kwargs,
            )
            add_gazing_counts(probe, result)
            return result

        processor._get_gazing_info_from_videos = MethodType(timed_get_gazing_info, processor)

    processor._autogaze_timing_installed = True


def install_model_timing(model, probe: TimingProbe) -> None:
    if getattr(model, "_autogaze_timing_installed", False):
        return

    if hasattr(model, "_run_vision_tower_batched"):
        original_run_vision_tower = model._run_vision_tower_batched

        def timed_run_vision_tower(self, all_pixels, gazing_info_batch, *args, **kwargs):
            label = "vision_encoding_thumbnails" if int(all_pixels.shape[1]) == 1 else "vision_encoding_tiles"
            probe.add_count("num_vision_tower_batches", 1)
            probe.add_count("num_encoded_thumbnails" if label.endswith("thumbnails") else "num_encoded_tiles", int(all_pixels.shape[0]))
            return probe.timed(
                label,
                original_run_vision_tower,
                all_pixels,
                gazing_info_batch,
                *args,
                sync_cuda=True,
                **kwargs,
            )

        model._run_vision_tower_batched = MethodType(timed_run_vision_tower, model)

    if hasattr(model, "_encode_vision"):
        original_encode_vision = model._encode_vision

        def timed_encode_vision(self, *args, **kwargs):
            result = probe.timed(
                "encode_vision_total",
                original_encode_vision,
                *args,
                sync_cuda=True,
                **kwargs,
            )
            probe.add_count("num_visual_tokens_projected", sum(int(features.shape[0]) for features in result))
            return result

        model._encode_vision = MethodType(timed_encode_vision, model)

    if hasattr(model, "_embed"):
        original_embed = model._embed

        def timed_embed(self, *args, **kwargs):
            return probe.timed(
                "embed_total",
                original_embed,
                *args,
                sync_cuda=True,
                **kwargs,
            )

        model._embed = MethodType(timed_embed, model)

    model._autogaze_timing_installed = True


def add_gazing_counts(probe: TimingProbe, gazing_info: Optional[dict]) -> None:
    if not gazing_info:
        return

    for key, count_key in [
        ("num_gazing_each_frame_tiles", "num_gaze_tokens_tiles"),
        ("num_gazing_each_frame_thumbnails", "num_gaze_tokens_thumbnails"),
    ]:
        values = gazing_info.get(key, [])
        for value in values:
            probe.add_count(count_key, int(value.sum().item()))

    for key, count_key in [
        ("if_padded_gazing_tiles", "num_non_padded_gaze_tokens_tiles"),
        ("if_padded_gazing_thumbnails", "num_non_padded_gaze_tokens_thumbnails"),
    ]:
        values = gazing_info.get(key, [])
        for value in values:
            probe.add_count(count_key, int((~value).sum().item()))


def collect_input_counts(inputs: dict, model, tile_starts: Sequence[int], visited_frames: Sequence[Image.Image]) -> Dict[str, int]:
    counts = {
        "num_temporal_tiles": int(len(tile_starts)),
        "num_visited_frames": int(len(visited_frames)),
    }
    if "num_spatial_tiles_each_video" in inputs:
        counts["num_spatial_tiles"] = int(sum(inputs["num_spatial_tiles_each_video"]))
    if "pixel_values_videos_tiles" in inputs:
        counts["num_encoded_tiles"] = int(sum(t.shape[0] for t in inputs["pixel_values_videos_tiles"]))
    if "pixel_values_videos_thumbnails" in inputs:
        counts["num_encoded_thumbnails"] = int(sum(t.shape[0] for t in inputs["pixel_values_videos_thumbnails"]))
        counts["num_thumbnail_frames"] = counts["num_encoded_thumbnails"]
    if "input_ids" in inputs:
        input_ids = inputs["input_ids"]
        counts["num_input_tokens_total"] = int(input_ids.numel())
        video_token_id = getattr(getattr(model, "config", None), "video_token_id", None)
        if video_token_id is not None:
            counts["num_visual_tokens_in_context"] = int((input_ids == video_token_id).sum().item())
            counts["num_text_tokens"] = counts["num_input_tokens_total"] - counts["num_visual_tokens_in_context"]
    return counts


def finalize_v2_timing(timing: Dict[str, float], probe: TimingProbe, processor_seconds: float, generate_seconds: float) -> Dict[str, float]:
    seconds = probe.snapshot_seconds()
    vision_encoding = seconds.get("vision_encoding_tiles", 0.0) + seconds.get("vision_encoding_thumbnails", 0.0)
    visual_token_processing = max(
        0.0,
        seconds.get("embed_total", seconds.get("encode_vision_total", 0.0)) - vision_encoding,
    )
    autogaze_processing = seconds.get("autogaze_processing", 0.0)
    nvila_preprocess = max(0.0, processor_seconds - autogaze_processing)

    timing.update(
        {
            "video_decode_and_sampling": timing.get("frame_load", 0.0),
            "proxy_scoring": timing.get("proxy", 0.0),
            "nvila_preprocess": nvila_preprocess,
            "nvila_preprocess_videos": seconds.get("nvila_preprocess_videos", 0.0),
            "autogaze_processing": autogaze_processing,
            "vision_encoding": vision_encoding,
            "vision_encoding_tiles": seconds.get("vision_encoding_tiles", 0.0),
            "vision_encoding_thumbnails": seconds.get("vision_encoding_thumbnails", 0.0),
            "visual_token_processing": visual_token_processing,
            "encode_vision_total": seconds.get("encode_vision_total", 0.0),
            "embed_total": seconds.get("embed_total", 0.0),
            "model_generate": generate_seconds,
        }
    )
    attributed = (
        timing["video_decode_and_sampling"]
        + timing["proxy_scoring"]
        + timing["nvila_preprocess"]
        + timing["model_generate"]
    )
    timing["system_overhead"] = max(0.0, timing.get("total", 0.0) - attributed)
    return timing


def proxy_decision_counts(decisions: Sequence[dict]) -> Dict[str, object]:
    num_scores = sum(len(decision.get("scores", {})) for decision in decisions)
    selected_delta_counts = Counter(str(decision.get("selected_delta")) for decision in decisions)
    return {
        "proxy_num_candidate_scores": int(num_scores),
        "proxy_seconds_per_score": None,
        "selected_delta_counts": dict(selected_delta_counts),
    }


def proxy_tile_starts(
    frames: np.ndarray,
    tile_len: int,
    deltas: Sequence[int],
    scorer: ProxyScorer,
    proxy: str,
    lambda_cost: float,
    temporal_cost_name: str,
    question: Optional[str] = None,
) -> Tuple[List[int], List[dict], float]:
    starts = [0] if len(frames) >= tile_len else []
    decisions = []
    tile_cache: Dict[int, np.ndarray] = {}
    resized_cache: Dict[int, torch.Tensor] = {}
    siglip_cache: Dict[int, torch.Tensor] = {}
    semantic_cache: Dict[int, torch.Tensor] = {}
    text_cache: Dict[str, torch.Tensor] = {}
    videomae_cache: Dict[int, float] = {}
    proxy_seconds = 0.0

    while starts:
        current_start = starts[-1]
        base_next_start = current_start + tile_len
        valid = [delta for delta in deltas if base_next_start + delta + tile_len <= len(frames)]
        if not valid:
            break

        scores: Dict[int, float] = {}
        started = time.perf_counter()
        for delta in valid:
            candidate_start = base_next_start + delta
            scores[int(delta)] = scorer.score(
                proxy,
                current_start,
                candidate_start,
                frames,
                tile_len,
                tile_cache,
                resized_cache,
                siglip_cache,
                semantic_cache,
                text_cache,
                videomae_cache,
                question=question,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        proxy_seconds += time.perf_counter() - started

        selected_delta, utilities = select_best_action(
            scores,
            lambda_cost=lambda_cost,
            cost=temporal_cost_name,
        )
        next_start = base_next_start + selected_delta
        if next_start + tile_len > len(frames):
            break
        decisions.append(
            {
                "current_start": current_start,
                "selected_delta": selected_delta,
                "next_start": next_start,
                "num_valid_deltas": len(valid),
                "scores": {str(delta): score for delta, score in scores.items()},
                "utilities": {str(delta): utility for delta, utility in utilities.items()},
            }
        )
        starts.append(next_start)

    return starts, decisions, proxy_seconds


def frames_to_pil(frames: np.ndarray, tile_starts: Sequence[int], tile_len: int) -> List[Image.Image]:
    selected = []
    for start in tile_starts:
        selected.extend(frames[start : start + tile_len])
    return [Image.fromarray(frame) for frame in selected]


def summarize(results_path: Path) -> dict:
    rows = []
    with results_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    total = len(rows)
    correct = sum(1 for row in rows if row.get("is_correct"))
    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "num_examples": len(subset),
            "num_correct": sum(1 for row in subset if row.get("is_correct")),
            "accuracy": sum(1 for row in subset if row.get("is_correct")) / max(len(subset), 1),
        }
    tile_counts = [row["num_visited_tiles"] for row in rows]
    visited_frame_counts = [row["num_visited_frames"] for row in rows]
    llm_times = [
        row["timing_seconds"].get("model_generate", row["timing_seconds"].get("llm_inference", 0.0))
        for row in rows
        if "timing_seconds" in row
    ]
    total_times = [
        row["timing_seconds"].get("total")
        for row in rows
        if isinstance(row.get("timing_seconds", {}).get("total"), (int, float))
    ]
    timing_keys = sorted({key for row in rows for key in row.get("timing_seconds", {})})
    timing_summary = {}
    for key in timing_keys:
        values = [row.get("timing_seconds", {}).get(key) for row in rows]
        values = [value for value in values if isinstance(value, (int, float))]
        if values:
            timing_summary[key] = {
                "mean": float(np.mean(values)),
                "total": float(np.sum(values)),
            }
    count_keys = sorted({key for row in rows for key in row.get("timing_counts", {})})
    count_summary = {}
    for key in count_keys:
        values = [row.get("timing_counts", {}).get(key) for row in rows]
        values = [value for value in values if isinstance(value, (int, float))]
        if values:
            count_summary[key] = {
                "mean": float(np.mean(values)),
                "total": float(np.sum(values)),
            }
    selected_delta_counts = Counter()
    for row in rows:
        selected_delta_counts.update(row.get("timing_counts", {}).get("selected_delta_counts", {}))
    visual_tokens_projected = _count_values(rows, "num_visual_tokens_projected")
    visual_tokens_in_context = _count_values(rows, "num_visual_tokens_in_context")

    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": correct / max(total, 1),
        "by_category": by_category,
        "prediction_counts": dict(Counter(row.get("prediction_letter", "") for row in rows)),
        "mean_visited_tiles": float(np.mean(tile_counts)) if tile_counts else 0.0,
        "total_visited_tiles": int(np.sum(tile_counts)) if tile_counts else 0,
        "mean_visited_frames": float(np.mean(visited_frame_counts)) if visited_frame_counts else 0.0,
        "total_visited_frames": int(np.sum(visited_frame_counts)) if visited_frame_counts else 0,
        "mean_visual_tokens_projected": float(np.mean(visual_tokens_projected)) if visual_tokens_projected else None,
        "total_visual_tokens_projected": int(np.sum(visual_tokens_projected)) if visual_tokens_projected else None,
        "mean_visual_tokens_in_context": float(np.mean(visual_tokens_in_context)) if visual_tokens_in_context else None,
        "total_visual_tokens_in_context": int(np.sum(visual_tokens_in_context)) if visual_tokens_in_context else None,
        "mean_end_to_end_seconds": float(np.mean(total_times)) if total_times else 0.0,
        "total_end_to_end_seconds": float(np.sum(total_times)) if total_times else 0.0,
        "examples_per_end_to_end_second": total / float(np.sum(total_times)) if total_times and np.sum(total_times) > 0 else 0.0,
        "selected_delta_counts": dict(selected_delta_counts),
        "total_llm_inference_seconds": float(np.sum(llm_times)) if llm_times else 0.0,
        "mean_llm_inference_seconds": float(np.mean(llm_times)) if llm_times else 0.0,
        "timing_summary_seconds": timing_summary,
        "timing_count_summary": count_summary,
    }


def _count_values(rows: Sequence[dict], key: str) -> List[float]:
    values = [row.get("timing_counts", {}).get(key) for row in rows]
    return [float(value) for value in values if isinstance(value, (int, float))]


def main() -> None:
    args = parse_args()
    if args.selection == "proxy" and args.proxy_metric is None:
        raise ValueError("--selection proxy requires --proxy-metric")
    if args.selection == "uniform" and args.proxy_metric is not None:
        raise ValueError("--proxy-metric is only valid with --selection proxy")
    if args.proxy_metric in QUESTION_CONDITIONED_METRICS and args.semantic_model != "siglip":
        raise ValueError(f"{args.proxy_metric} uses SigLIP text embeddings; set --semantic-model siglip")

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_root)[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]
    done_ids = existing_question_ids(output_path) if args.resume else set()
    deltas = parse_csv_ints(args.deltas)

    max_samples = max_sampled_count_for_rows(
        rows,
        dataset_root,
        args.sampling,
        args.tile_len,
        args.max_dense_sampled_frames,
    )
    max_tiles_video = args.max_tiles_video

    scorer = ProxyScorer(args) if args.selection == "proxy" else None
    processor = None
    model = None
    timing_probe = TimingProbe()
    if not args.dry_run_sampling:
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            autogaze_model_id=args.autogaze_model_id,
            num_video_frames=max_samples,
            num_video_frames_thumbnail=args.num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            gazing_ratio_tile=[0.2] + [0.06] * 15,
            gazing_ratio_thumbnail=1,
            task_loss_requirement_tile=0.6,
            task_loss_requirement_thumbnail=None,
            max_batch_size_autogaze=args.max_batch_size_autogaze,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            device_map=args.device_map,
            torch_dtype=dtype_from_name(args.torch_dtype),
            max_batch_size_siglip=args.max_batch_size_siglip,
        )
        model.eval()
        install_processor_timing(processor, timing_probe)
        install_model_timing(model, timing_probe)

    cached_video_path = None
    cached_samples = None
    with output_path.open("a") as out:
        for row in tqdm(rows, desc="HLVid"):
            if row["question_id"] in done_ids:
                continue
            timing_probe.reset()
            total_started = time.perf_counter()
            video_path = dataset_root / "videos" / row["video_path"]
            load_started = time.perf_counter()
            cache_hit = cached_video_path == video_path
            if cache_hit:
                frames, source_indices, source_frame_count = cached_samples
                frame_load_seconds = 0.0
            else:
                frames, source_indices, source_frame_count = load_video_samples(
                    video_path,
                    args.sampling,
                    args.tile_len,
                    args.max_dense_sampled_frames,
                )
                cached_video_path = video_path
                cached_samples = (frames, source_indices, source_frame_count)
                frame_load_seconds = time.perf_counter() - load_started

            proxy_seconds = 0.0
            decisions = []
            if args.selection == "uniform":
                tile_starts = uniform_tile_starts(len(frames), args.tile_len)
            else:
                tile_starts, decisions, proxy_seconds = proxy_tile_starts(
                    frames=frames,
                    tile_len=args.tile_len,
                    deltas=deltas,
                    scorer=scorer,
                    proxy=args.proxy_metric,
                    lambda_cost=args.lambda_cost,
                    temporal_cost_name=args.temporal_cost,
                    question=row.get("question"),
                )
            visited_frames = frames_to_pil(frames, tile_starts, args.tile_len)
            proxy_counts = proxy_decision_counts(decisions)
            if proxy_counts["proxy_num_candidate_scores"]:
                proxy_counts["proxy_seconds_per_score"] = proxy_seconds / proxy_counts["proxy_num_candidate_scores"]

            timing = {
                "frame_load": frame_load_seconds,
                "proxy": proxy_seconds,
                "processor": 0.0,
                "llm_inference": 0.0,
                "model_generate": 0.0,
                "total": 0.0,
            }
            timing_counts = dict(proxy_counts)
            response = ""
            pred = ""
            is_correct = False
            if not args.dry_run_sampling:
                prompt = f"{processor.tokenizer.video_token}\n\nQuestion: {row['question']}"
                processor_started = time.perf_counter()
                inputs = processor(text=prompt, videos=[visited_frames], return_tensors="pt")
                cuda_synchronize()
                timing["processor"] = time.perf_counter() - processor_started
                target_device = model.device if hasattr(model, "device") else next(model.parameters()).device
                timing_counts.update(collect_input_counts(inputs, model, tile_starts, visited_frames))
                inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                llm_started = time.perf_counter()
                cuda_synchronize()
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                cuda_synchronize()
                timing["llm_inference"] = time.perf_counter() - llm_started
                timing["model_generate"] = timing["llm_inference"]
                timing_counts["generated_tokens"] = int(outputs.shape[1] - inputs["input_ids"].shape[1])
                timing_counts.update(timing_probe.snapshot_counts())
                response = processor.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )[0].strip()
                pred = extract_letter(response)
                is_correct = pred == str(row["answer"]).strip().upper()

            timing["total"] = time.perf_counter() - total_started
            if not args.dry_run_sampling:
                timing = finalize_v2_timing(
                    timing,
                    timing_probe,
                    processor_seconds=timing["processor"],
                    generate_seconds=timing["model_generate"],
                )
            record = {
                "question_id": row["question_id"],
                "category": row["category"],
                "video_path": row["video_path"],
                "answer": str(row["answer"]).strip().upper(),
                "prediction": response,
                "prediction_letter": pred,
                "is_correct": is_correct,
                "sampling": args.sampling,
                "selection": args.selection,
                "proxy_metric": args.proxy_metric,
                "proxy_base_metric": args.proxy_base_metric,
                "semantic_model": args.semantic_model,
                "lambda_cost": args.lambda_cost,
                "temporal_cost": args.temporal_cost,
                "deltas": deltas,
                "tile_len": args.tile_len,
                "source_frame_count": source_frame_count,
                "frame_cache_hit": cache_hit,
                "num_sampled_frames": len(frames),
                "num_unique_source_frames": len(set(int(index) for index in source_indices)),
                "num_visited_frames": len(visited_frames),
                "num_visited_tiles": len(tile_starts),
                "visited_frame_ratio": len(visited_frames) / max(len(frames), 1),
                "tile_starts": tile_starts,
                "proxy_decisions": decisions,
                "timing_seconds": timing,
                "timing_counts": timing_counts,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

    summary = summarize(output_path)
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "model_path": args.model_path,
            "autogaze_model_id": args.autogaze_model_id,
            "sampling": args.sampling,
            "selection": args.selection,
            "proxy_metric": args.proxy_metric,
            "proxy_base_metric": args.proxy_base_metric,
            "semantic_model": args.semantic_model,
            "lambda_cost": args.lambda_cost,
            "temporal_cost": args.temporal_cost,
            "deltas": deltas,
            "tile_len": args.tile_len,
            "max_tiles_video": max_tiles_video,
            "max_dense_sampled_frames": args.max_dense_sampled_frames,
            "dry_run_sampling": args.dry_run_sampling,
        }
    )
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
