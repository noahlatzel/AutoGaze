# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate HLVid selected-frame JSONL using BOLT query-aware sampling.

This wrapper consumes BOLT-style pre-extracted visual features: one ``.npy`` file
per video containing CLIP/SigLIP frame features sampled at ``--feature-fps``.
It scores each feature row against the HLVid question text, applies BOLT inverse
transform sampling, maps the sampled feature rows back to original video frame
indices, and writes the fixed JSONL interface expected by the parent evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runners.hlvid_common import load_rows  # noqa: E402
from scripts.frame_selectors.common import model_path, selected_index_record, video_metadata  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/storage/slurm/latn/data/AutoGaze/HLVid")
DEFAULT_BOLT_ROOT = REPO_ROOT / "external" / "frame_selectors" / "BOLT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BOLT selected source-frame indices for HLVid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bolt-root", type=Path, default=DEFAULT_BOLT_ROOT)
    parser.add_argument("--model", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--clip-model", default=model_path("clip_vit_l14"))
    parser.add_argument("--siglip-model", default=model_path("siglip2_base_patch16_224"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--feature-fps", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--power", type=float, default=-1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading BOLT or writing JSONL.")
    parser.add_argument(
        "--skip-missing-features",
        action="store_true",
        help="Skip rows whose feature file is missing instead of failing.",
    )
    return parser.parse_args()


def video_feature_path(features_dir: Path, video_path: str) -> Path:
    rel = Path(video_path)
    candidates = [
        features_dir / rel.with_suffix(".npy"),
        features_dir / f"{rel.stem}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def inverse_transform_sampling(score: np.ndarray, n: int, power: float = -1.0) -> np.ndarray:
    """BOLT inverse transform sampling with a deterministic uniform fallback.

    Upstream BOLT normalizes query-frame scores to [0, 1], builds a CDF, then
    searches evenly spaced CDF quantiles. Degenerate scores can produce an empty
    distribution, so this wrapper falls back to uniform feature-row positions.
    """

    if n <= 0:
        raise ValueError("--num-frames must be positive")
    if score.ndim != 1:
        raise ValueError(f"Expected 1D score array, got shape {score.shape}")
    if len(score) == 0:
        raise ValueError("Cannot sample from an empty score array")

    score = np.nan_to_num(score.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    score = score - np.min(score)
    max_score = np.max(score)
    if max_score <= 0:
        return np.round(np.linspace(0, len(score) - 1, n)).astype(np.int64)

    score = score / max_score
    if power != -1:
        score = score**power

    total = np.sum(score)
    if total <= 0 or not np.isfinite(total):
        return np.round(np.linspace(0, len(score) - 1, n)).astype(np.int64)

    cdf = np.cumsum(score / total)
    uniform_sampling = np.linspace(1 / n, 1 - 1 / n, n)
    return np.searchsorted(cdf, uniform_sampling).clip(0, len(score) - 1).astype(np.int64)


def feature_rows_to_source_indices(
    feature_rows: np.ndarray,
    video_path: Path,
    feature_fps: float,
    frame_count: Optional[int] = None,
) -> list[int]:
    if feature_fps <= 0:
        raise ValueError("--feature-fps must be positive")

    if frame_count is None:
        frame_count, video_fps = video_metadata(video_path)
    else:
        _, video_fps = video_metadata(video_path)

    source = np.round(feature_rows.astype(np.float64) / feature_fps * video_fps)
    source = np.clip(source, 0, frame_count - 1).astype(np.int64)
    return [int(index) for index in source.tolist()]


def iter_rows(rows: list[dict], limit: Optional[int]) -> Iterable[dict]:
    yield from rows if limit is None else rows[:limit]


def torch_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


class TextFeatureScorer:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPProcessor

        self.torch = torch
        self.device = args.device
        self.model_name = args.clip_model if args.model == "clip" else args.siglip_model
        dtype = torch_dtype(args.dtype)
        self.model = AutoModel.from_pretrained(self.model_name).to(device=args.device, dtype=dtype)
        self.model.eval()
        if args.model == "clip":
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def extract_text_features(self, text: str):
        inputs = self.tokenizer([text], truncation=True, padding="max_length", return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            return self.model.get_text_features(**inputs)

    def compute_similarity(self, video_features, text_features):
        video_features = video_features.to(device=text_features.device, dtype=text_features.dtype)
        video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        logits = self.torch.matmul(text_features, video_features.t())
        if hasattr(self.model, "logit_scale"):
            logits = logits * self.model.logit_scale.exp().to(text_features.device)
        if hasattr(self.model, "logit_bias"):
            logits = logits + self.model.logit_bias.to(text_features.device)
        return logits.detach().cpu().squeeze(0).float()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset_root)
    rows = list(iter_rows(rows, args.limit))

    missing = []
    for row in rows:
        if not video_feature_path(args.features_dir, row["video_path"]).exists():
            missing.append(row["video_path"])

    if args.dry_run:
        print(f"HLVid rows: {len(rows)}")
        print(f"Feature directory: {args.features_dir}")
        print(f"Missing feature files: {len(missing)}")
        if missing[:5]:
            print("First missing features:")
            for item in missing[:5]:
                print(f"  {item}")
        return

    if missing and not args.skip_missing_features:
        examples = "\n".join(f"  {item}" for item in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} feature files under {args.features_dir}. "
            f"Run BOLT feature extraction first, or pass --skip-missing-features.\n{examples}"
        )

    model = TextFeatureScorer(args)
    videos_dir = args.dataset_root / "videos"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w") as f:
        for row in rows:
            feature_path = video_feature_path(args.features_dir, row["video_path"])
            if not feature_path.exists():
                continue

            video_path = videos_dir / row["video_path"]
            video_features = np.load(feature_path)
            if video_features.ndim != 2:
                raise ValueError(f"Expected [T, C] features in {feature_path}, got {video_features.shape}")

            import torch

            video_features_tensor = torch.from_numpy(video_features).to(args.device)
            text_features = model.extract_text_features(row["question"])
            score = model.compute_similarity(video_features_tensor, text_features).detach().cpu().numpy()
            feature_rows = inverse_transform_sampling(score, args.num_frames, args.power)
            frame_count, _ = video_metadata(video_path)
            selected_source_indices = feature_rows_to_source_indices(
                feature_rows,
                video_path,
                args.feature_fps,
                frame_count=frame_count,
            )

            record = selected_index_record(
                question_id=str(row["question_id"]),
                video_path=str(row["video_path"]),
                method="bolt",
                num_frames=args.num_frames,
                selected_source_indices=selected_source_indices,
                frame_count=frame_count,
                metadata={
                    "feature_path": str(feature_path),
                    "feature_fps": args.feature_fps,
                    "model": args.model,
                    "model_name": model.model_name,
                    "power": args.power,
                },
            )
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} BOLT selections to {args.output}")


if __name__ == "__main__":
    main()
