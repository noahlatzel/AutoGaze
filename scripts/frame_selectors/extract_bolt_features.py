# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract BOLT-compatible per-video frame features for HLVid.

The upstream BOLT script depends on decord and writes one .npy file per video.
This local extractor keeps the same feature-file interface but uses PyAV, so it
works in the current AutoGaze/NVILA environments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.frame_selectors.common import load_pil_frames_av, model_path, video_metadata
from scripts.runners.hlvid_common import load_rows


DEFAULT_DATASET_ROOT = Path("/storage/slurm/latn/data/AutoGaze/HLVid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BOLT-compatible HLVid visual features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--clip-model", default=model_path("clip_vit_l14"))
    parser.add_argument("--siglip-model", default=model_path("siglip2_base_patch16_224"))
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def batches(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def feature_path(output_dir: Path, video_path: str) -> Path:
    return output_dir / Path(video_path).with_suffix(".npy")


def sampled_indices(video_path: Path, fps: float) -> list[int]:
    if fps <= 0:
        raise ValueError("--fps must be positive")
    frame_count, video_fps = video_metadata(video_path)
    total = max(1, int(frame_count / video_fps * fps))
    indices = np.round(np.arange(total, dtype=np.float64) / fps * video_fps)
    return np.clip(indices, 0, frame_count - 1).astype(int).tolist()


def load_model(args: argparse.Namespace):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPProcessor

    dtype = torch_dtype(args.dtype)
    if args.model == "clip":
        model = AutoModel.from_pretrained(args.clip_model).to(device=args.device, dtype=dtype)
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        tokenizer = AutoTokenizer.from_pretrained(args.clip_model)
    else:
        model = AutoModel.from_pretrained(args.siglip_model).to(device=args.device, dtype=dtype)
        processor = AutoProcessor.from_pretrained(args.siglip_model)
        tokenizer = AutoTokenizer.from_pretrained(args.siglip_model)
    model.eval()
    return model, processor, tokenizer


@torch.inference_mode()
def extract_features(model, processor, frames: list, args: argparse.Namespace) -> np.ndarray:
    chunks = []
    dtype = torch_dtype(args.dtype)
    device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    for frame_batch in batches(frames, args.batch_size):
        inputs = processor(images=frame_batch, return_tensors="pt").to(device=args.device, dtype=dtype)
        with torch.autocast(device_type, enabled=device_type == "cuda"):
            features = model.get_image_features(**inputs)
        chunks.append(features.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset_root)
    if args.limit is not None:
        rows = rows[: args.limit]

    model, processor, _ = load_model(args)
    videos_root = args.dataset_root / "videos"
    for row in tqdm(rows, desc=f"bolt_features_{args.model}"):
        output_path = feature_path(args.output_dir, row["video_path"])
        if args.resume and output_path.exists():
            continue
        video_path = videos_root / row["video_path"]
        indices = sampled_indices(video_path, args.fps)
        frames = load_pil_frames_av(video_path, indices)
        features = extract_features(model, processor, frames, args)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, features)


if __name__ == "__main__":
    main()
