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

"""Generate pseudo-labels for temporal delta scheduler experiments.

This is Experiment 0 tooling: it computes candidate skip scores for videos and
optionally converts them into labels by choosing the largest delta whose score
is below a configured threshold.
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import av
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.tasks.video_mae_reconstruction import VideoMAEReconstruction


DEFAULT_MODEL_ROOT = "/storage/slurm/latn/data/AutoGaze/hf_models"
DEFAULT_VIDEOMAE_WEIGHTS = "/storage/slurm/latn/data/AutoGaze/VideoMAE_AutoGaze/videomae.pt"
FPS_RE = re.compile(r"_fps(?P<fps>[0-9]+(?:\.[0-9]+)?)")


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
    parser.add_argument("--output", required=True, help="Output JSON path for metadata, raw scores, labels, and summary.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of valid videos to process across all input roots; useful for calibration runs.",
    )
    parser.add_argument(
        "--max-tiles-per-video",
        type=int,
        default=None,
        help="Maximum tile start positions to evaluate per video; starts are taken in temporal order.",
    )
    parser.add_argument("--tile-len", type=int, default=16, help="Number of frames in each current/candidate temporal tile.")
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=16,
        help="Spacing between evaluated current tile starts. This is separate from candidate deltas.",
    )
    parser.add_argument(
        "--deltas",
        default="1,2,4,8,16,32",
        help="Comma-separated candidate next-start offsets, in frames, evaluated from each current tile start.",
    )
    parser.add_argument(
        "--heuristics",
        default="pixel_mse,siglip",
        help=(
            "Comma-separated subset of heuristics to compute: pixel_mse compares resized pixels; "
            "siglip compares pooled SigLIP frame features; videomae_reconstruction scores the "
            "candidate destination tile with the repo's VideoMAE reconstruction task."
        ),
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help=(
            "Optional comma-separated per-heuristic thresholds, e.g. pixel_mse=0.02,siglip=0.15. "
            "When provided, the label is the largest delta whose score is <= threshold; otherwise labels are null."
        ),
    )
    parser.add_argument("--resize", type=int, default=224, help="Spatial resize used before pixel MSE comparisons.")
    parser.add_argument(
        "--full-video-decode-max-frames",
        type=int,
        default=256,
        help=(
            "Decode a whole video once when frame_count is at most this value, then slice tiles from memory. "
            "Set to 0 to always decode individual tiles."
        ),
    )
    parser.add_argument(
        "--feature-frame-stride",
        type=int,
        default=4,
        help="Use every n-th frame from a tile when computing SigLIP tile embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Frame batch size for SigLIP feature extraction.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device for model heuristics.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32", help="Torch dtype for model heuristics.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic torch/numpy setup.")
    parser.add_argument(
        "--fps-buckets",
        default="low:0:4,medium:4:10,high:10:inf",
        help=(
            "Comma-separated FPS buckets as name:min:max. Bounds are [min, max). "
            "Use inf for an open upper bound. Unknown/missing FPS is bucketed as unknown."
        ),
    )

    parser.add_argument(
        "--siglip-model",
        default=os.path.join(DEFAULT_MODEL_ROOT, "google--siglip2-base-patch16-224"),
        help="Local path or Hugging Face model name for the SigLIP heuristic.",
    )

    parser.add_argument(
        "--videomae-model",
        default=os.path.join(DEFAULT_MODEL_ROOT, "facebook--vit-mae-large"),
        help="Base model path/name used to construct the repo's VideoMAE reconstruction task.",
    )
    parser.add_argument(
        "--videomae-weights",
        default=DEFAULT_VIDEOMAE_WEIGHTS,
        help="Optional task checkpoint path loaded into the VideoMAE reconstruction task if it exists.",
    )
    parser.add_argument(
        "--videomae-scales",
        default="224",
        help="Scale string for the VideoMAE reconstruction oracle, matching task.scales format.",
    )
    parser.add_argument(
        "--videomae-patch-size",
        type=int,
        default=16,
        help="Patch size used to construct the synthetic fixed gaze mask for the VideoMAE oracle.",
    )
    parser.add_argument(
        "--videomae-gaze-ratio",
        type=float,
        default=0.25,
        help="Fraction of per-frame patches included in the synthetic fixed gaze mask for the VideoMAE oracle.",
    )
    parser.add_argument(
        "--videomae-attn-mode",
        default="sdpa",
        help="Attention implementation passed when constructing the VideoMAE reconstruction task.",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_thresholds(value: str) -> Dict[str, float]:
    thresholds = {}
    if not value:
        return thresholds
    for item in value.split(","):
        if not item.strip():
            continue
        name, threshold = item.split("=", 1)
        thresholds[name.strip()] = float(threshold)
    return thresholds


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
    fps_buckets: Sequence[Dict[str, float]],
) -> List[VideoRecord]:
    records = []
    for raw_root in input_roots:
        root = Path(raw_root)
        search_root = root / split if split and (root / split).is_dir() else root
        if not search_root.exists():
            raise FileNotFoundError(f"Video root does not exist: {search_root}")
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
            if limit is not None and len(records) >= limit:
                return records
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
    max_start = frame_count - min_delta - tile_len
    if max_start < 0:
        return []
    return list(range(0, max_start + 1, tile_stride))


def read_tile(path: Path, start: int, tile_len: int) -> np.ndarray:
    with av.open(str(path)) as container:
        indices = np.arange(start, start + tile_len)
        return read_video_pyav(container, indices)


def read_video(path: Path, frame_count: int) -> np.ndarray:
    with av.open(str(path)) as container:
        indices = np.arange(frame_count)
        return read_video_pyav(container, indices)


def resize_tile(tile: np.ndarray, size: int) -> torch.Tensor:
    tensor = torch.from_numpy(tile).permute(0, 3, 1, 2).float() / 255.0
    tensor = F.interpolate(tensor, size=(size, size), mode="bicubic", align_corners=False)
    return tensor.clamp(0, 1)


class SiglipTileEncoder:
    def __init__(self, model_name_or_path: str, device: str, dtype: torch.dtype, batch_size: int):
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device=device, dtype=dtype)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, tile: np.ndarray, frame_stride: int) -> torch.Tensor:
        frames = tile[::frame_stride]
        embeddings = []
        for start in range(0, len(frames), self.batch_size):
            batch = list(frames[start:start + self.batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
            if hasattr(self.model, "vision_model"):
                outputs = self.model.vision_model(pixel_values=pixel_values)
            else:
                outputs = self.model(pixel_values=pixel_values)
            frame_embeddings = get_pooled_model_output(outputs)
            embeddings.append(frame_embeddings.float())
        embedding = torch.cat(embeddings, dim=0).mean(dim=0)
        return F.normalize(embedding, dim=0).cpu()

    @torch.inference_mode()
    def encode_many(self, tiles: Dict[int, np.ndarray], frame_stride: int) -> Dict[int, torch.Tensor]:
        if not tiles:
            return {}

        starts = []
        frames = []
        for tile_start, tile in tiles.items():
            tile_frames = tile[::frame_stride]
            starts.extend([tile_start] * len(tile_frames))
            frames.extend(list(tile_frames))

        frame_embeddings = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start:start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
            if hasattr(self.model, "vision_model"):
                outputs = self.model.vision_model(pixel_values=pixel_values)
            else:
                outputs = self.model(pixel_values=pixel_values)
            frame_embeddings.append(get_pooled_model_output(outputs).float())

        all_frame_embeddings = torch.cat(frame_embeddings, dim=0)
        tile_embeddings = {}
        for tile_start in tiles:
            frame_indices = [idx for idx, start in enumerate(starts) if start == tile_start]
            embedding = all_frame_embeddings[frame_indices].mean(dim=0)
            tile_embeddings[tile_start] = F.normalize(embedding, dim=0).cpu()
        return tile_embeddings


def get_pooled_model_output(outputs) -> torch.Tensor:
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "vision_model_output") and outputs.vision_model_output is not None:
        vision_output = outputs.vision_model_output
        if hasattr(vision_output, "pooler_output") and vision_output.pooler_output is not None:
            return vision_output.pooler_output
        return vision_output.last_hidden_state.mean(dim=1)
    return outputs.last_hidden_state.mean(dim=1)


class VideoMAEReconstructionOracle:
    def __init__(
        self,
        model_name_or_path: str,
        weights_path: Optional[str],
        scales: str,
        patch_size: int,
        gaze_ratio: float,
        attn_mode: str,
        device: str,
        dtype: torch.dtype,
    ):
        recon_model_config = OmegaConf.create(
            {
                "scale_embed": True,
                "max_num_frames": 256,
                "time_embed": True,
                "causal": True,
                "loss_type": "l1",
                "loss_weights": "1",
                "l1_loss_config": None,
                "dinov2_reg_loss_config": {"model": ""},
                "siglip2_loss_config": {"model": ""},
            }
        )
        self.task = VideoMAEReconstruction(
            recon_model=model_name_or_path,
            recon_model_config=recon_model_config,
            scales=scales,
            recon_sample_rate=1,
            attn_mode=attn_mode,
        ).to(device=device, dtype=dtype)
        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            self.task.load_state_dict(state_dict, strict=False)
        self.task.eval()
        self.device = device
        self.dtype = dtype
        self.scales = [int(scale) for scale in str(scales).split("+")]
        self.patch_size = patch_size
        self.gaze_ratio = gaze_ratio

    @torch.inference_mode()
    def score(self, tile: np.ndarray) -> float:
        video = transform_video_for_pytorch(tile, self.task.transform)
        video = video.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        gazing_info = self.make_fixed_gazing_info(
            batch_size=video.shape[0],
            num_frames=video.shape[1],
            device=video.device,
        )
        frame_idx_to_reconstruct = torch.arange(video.shape[1], device=video.device)
        outputs = self.task.mae(
            video,
            gazing_info=gazing_info,
            frame_idx_to_reconstruct=frame_idx_to_reconstruct,
            interpolate_pos_encoding=True,
        )
        return float(outputs.loss_mean.mean().detach().cpu())

    def make_fixed_gazing_info(self, batch_size: int, num_frames: int, device: torch.device) -> Dict[str, torch.Tensor]:
        num_tokens_each_frame = sum((scale // self.patch_size) ** 2 for scale in self.scales)
        num_gazed = max(1, min(num_tokens_each_frame, math.ceil(num_tokens_each_frame * self.gaze_ratio)))
        base_pos = torch.linspace(0, num_tokens_each_frame - 1, steps=num_gazed, device=device).round().long()
        frame_positions = [base_pos + frame_idx * num_tokens_each_frame for frame_idx in range(num_frames)]
        gazing_pos = torch.cat(frame_positions, dim=0).unsqueeze(0).repeat(batch_size, 1)
        return {
            "gazing_pos": gazing_pos,
            "num_gazing_each_frame": torch.full((num_frames,), num_gazed, device=device, dtype=torch.long),
            "if_padded_gazing": torch.zeros_like(gazing_pos, dtype=torch.bool),
        }


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((1 - torch.dot(first, second)).clamp(min=0).item())


def pixel_mse(first: np.ndarray, second: np.ndarray, size: int) -> float:
    first_tensor = resize_tile(first, size)
    second_tensor = resize_tile(second, size)
    return float((first_tensor - second_tensor).pow(2).mean().item())


def pixel_mse_tensors(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).pow(2).mean().item())


def choose_label(scores: Dict[str, float], threshold: Optional[float]) -> Optional[int]:
    if threshold is None or not scores:
        return None
    valid = [int(delta) for delta, score in scores.items() if score <= threshold]
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


def summarize_labels(results: List[dict], heuristics: Sequence[str]) -> Dict[str, dict]:
    num_tiles = sum(len(video["tiles"]) for video in results)
    summary = {
        "num_videos": len(results),
        "num_videos_with_tiles": sum(1 for video in results if video["tiles"]),
        "num_videos_without_tiles": sum(1 for video in results if not video["tiles"]),
        "num_tiles": num_tiles,
        "fps_buckets": dict(sorted(Counter(video["fps_bucket"] for video in results).items())),
        "heuristics": {},
    }
    for heuristic in heuristics:
        labels = []
        label_seconds = []
        scores = []
        scores_by_delta = {}
        bucket_summaries = {}
        for video in results:
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

        for bucket_name in sorted({video["fps_bucket"] for video in results}):
            bucket_labels = []
            bucket_label_seconds = []
            bucket_scores = []
            bucket_score_by_delta = {}
            for video in results:
                if video["fps_bucket"] != bucket_name:
                    continue
                for tile in video["tiles"]:
                    label = tile["labels"].get(heuristic)
                    if label is not None:
                        bucket_labels.append(label)
                        if video["effective_fps"]:
                            bucket_label_seconds.append(label / video["effective_fps"])
                    heuristic_scores = tile["scores"].get(heuristic, {})
                    bucket_scores.extend(heuristic_scores.values())
                    for delta, score in heuristic_scores.items():
                        bucket_score_by_delta.setdefault(delta, []).append(score)
            bucket_summaries[bucket_name] = {
                "num_labels": len(bucket_labels),
                "label_counts": dict(sorted(Counter(bucket_labels).items())),
                "label_delta_seconds": percentile_summary(bucket_label_seconds),
                "num_scores": len(bucket_scores),
                "scores": percentile_summary(bucket_scores),
                "scores_by_delta": {
                    delta: percentile_summary(delta_scores)
                    for delta, delta_scores in sorted(bucket_score_by_delta.items(), key=lambda item: int(item[0]))
                },
            }

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
            "fps_buckets": bucket_summaries,
        }
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    deltas = parse_csv_ints(args.deltas)
    heuristics = parse_csv_strings(args.heuristics)
    thresholds = parse_thresholds(args.thresholds)
    fps_buckets = parse_fps_buckets(args.fps_buckets)
    dtype = torch_dtype(args.dtype)

    unknown = sorted(set(heuristics) - {"pixel_mse", "siglip", "videomae_reconstruction"})
    if unknown:
        raise ValueError(f"Unknown heuristics: {unknown}")

    videos = discover_videos(args.input_roots, args.split, args.video_ext, args.limit, fps_buckets)
    siglip_encoder = None
    videomae_oracle = None
    if "siglip" in heuristics:
        siglip_encoder = SiglipTileEncoder(args.siglip_model, args.device, dtype, args.batch_size)
    if "videomae_reconstruction" in heuristics:
        videomae_oracle = VideoMAEReconstructionOracle(
            model_name_or_path=args.videomae_model,
            weights_path=args.videomae_weights,
            scales=args.videomae_scales,
            patch_size=args.videomae_patch_size,
            gaze_ratio=args.videomae_gaze_ratio,
            attn_mode=args.videomae_attn_mode,
            device=args.device,
            dtype=dtype,
        )

    all_results = []
    for video in tqdm(videos, desc="videos"):
        starts = valid_starts(video.frame_count, args.tile_len, args.tile_stride, deltas)
        if args.max_tiles_per_video is not None:
            starts = starts[:args.max_tiles_per_video]
        tile_cache: Dict[int, np.ndarray] = {}
        resized_cache: Dict[int, torch.Tensor] = {}
        siglip_cache: Dict[int, torch.Tensor] = {}
        videomae_cache: Dict[int, float] = {}
        full_video = None
        resized_full_video = None
        if starts and args.full_video_decode_max_frames > 0 and video.frame_count <= args.full_video_decode_max_frames:
            full_video = read_video(video.path, video.frame_count)
            if "pixel_mse" in heuristics:
                resized_full_video = resize_tile(full_video, args.resize)

        def get_tile(start: int) -> np.ndarray:
            if start not in tile_cache:
                if full_video is not None:
                    tile_cache[start] = full_video[start:start + args.tile_len]
                else:
                    tile_cache[start] = read_tile(video.path, start, args.tile_len)
            return tile_cache[start]

        def get_resized_tile(start: int) -> torch.Tensor:
            if start not in resized_cache:
                if resized_full_video is not None:
                    resized_cache[start] = resized_full_video[start:start + args.tile_len]
                else:
                    resized_cache[start] = resize_tile(get_tile(start), args.resize)
            return resized_cache[start]

        def get_siglip(start: int) -> torch.Tensor:
            if start not in siglip_cache:
                siglip_cache[start] = siglip_encoder.encode(get_tile(start), args.feature_frame_stride)
            return siglip_cache[start]

        def get_videomae_score(start: int) -> float:
            if start not in videomae_cache:
                videomae_cache[start] = videomae_oracle.score(get_tile(start))
            return videomae_cache[start]

        needed_starts = set(starts)
        for start in starts:
            for delta in deltas:
                candidate_start = start + delta
                if candidate_start + args.tile_len <= video.frame_count:
                    needed_starts.add(candidate_start)
        for needed_start in sorted(needed_starts):
            get_tile(needed_start)
        if "siglip" in heuristics and siglip_encoder is not None:
            siglip_cache.update(siglip_encoder.encode_many(tile_cache, args.feature_frame_stride))

        video_tiles = []
        for start in starts:
            scores = {heuristic: {} for heuristic in heuristics}
            labels = {}
            current_siglip = get_siglip(start) if "siglip" in heuristics else None

            for delta in deltas:
                candidate_start = start + delta
                if candidate_start + args.tile_len > video.frame_count:
                    continue
                if "pixel_mse" in heuristics:
                    scores["pixel_mse"][str(delta)] = pixel_mse_tensors(
                        get_resized_tile(start),
                        get_resized_tile(candidate_start),
                    )
                if "siglip" in heuristics:
                    candidate_siglip = get_siglip(candidate_start)
                    scores["siglip"][str(delta)] = cosine_distance(current_siglip, candidate_siglip)
                if "videomae_reconstruction" in heuristics:
                    scores["videomae_reconstruction"][str(delta)] = get_videomae_score(candidate_start)

            for heuristic in heuristics:
                labels[heuristic] = choose_label(scores[heuristic], thresholds.get(heuristic))

            video_tiles.append(
                {
                    "start": start,
                    "labels": labels,
                    "scores": scores,
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
            "tile_len": args.tile_len,
            "tile_stride": args.tile_stride,
            "deltas": deltas,
            "fps_buckets": fps_buckets,
            "heuristics": heuristics,
            "thresholds": thresholds,
            "resize": args.resize,
            "feature_frame_stride": args.feature_frame_stride,
            "siglip_model": args.siglip_model if "siglip" in heuristics else None,
            "videomae_model": args.videomae_model if "videomae_reconstruction" in heuristics else None,
            "videomae_weights": args.videomae_weights if "videomae_reconstruction" in heuristics else None,
            "videomae_scales": args.videomae_scales if "videomae_reconstruction" in heuristics else None,
            "videomae_gaze_ratio": args.videomae_gaze_ratio if "videomae_reconstruction" in heuristics else None,
        },
        "summary": summarize_labels(all_results, heuristics),
        "videos": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
