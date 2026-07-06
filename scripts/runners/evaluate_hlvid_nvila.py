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

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from autogaze.datasets.video_utils import transform_video_for_pytorch
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from extract_delta_probe_features import representation_features
from hlvid_common import dtype_from_name, existing_question_ids, extract_letter, load_rows
from run_hierarchical_coherence_dry_sweep import (
    AUTOGAZE_METRICS,
    SUPPORTED_METRICS as HIERARCHICAL_COHERENCE_METRICS,
    CoherenceScorer,
    build_hierarchy,
)
from train_delta_probe import build_delta_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NVILA-HD-Video on HLVid.")
    parser.add_argument("--dataset-root", default="/storage/slurm/latn/data/AutoGaze/HLVid")
    parser.add_argument("--model-path", default="/storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video")
    parser.add_argument("--autogaze-model-id", default="/storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze")
    parser.add_argument("--output", default="outputs/hlvid/nvila_default_autogaze/results.jsonl")
    parser.add_argument("--summary-output", default="outputs/hlvid/nvila_default_autogaze/summary.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument(
        "--frame-source",
        choices=["uniform", "dense", "hierarchical"],
        default="uniform",
        help=(
            "uniform: sample --num-video-frames over the full video, then apply tile skipping. "
            "dense: apply tile skipping in original video-frame coordinates and load only visited tiles. "
            "hierarchical: build best-first coherent temporal leaves and load selected source frames."
        ),
    )
    parser.add_argument(
        "--dense-frame-stride",
        type=int,
        default=1,
        help="Frame stride for dense mode before tile/delta traversal. 1 means original video frame units.",
    )
    parser.add_argument(
        "--max-source-tiles",
        type=int,
        default=None,
        help="Maximum visited dense tiles to load. Defaults to --max-tiles-video in dense mode.",
    )
    parser.add_argument("--tile-len", type=int, default=16)
    parser.add_argument("--delta-gap", type=int, default=0)
    parser.add_argument(
        "--delta-policy-checkpoint",
        default=None,
        help="Optional trained delta-probe checkpoint. When set, uniform mode uses learned inter-tile deltas.",
    )
    parser.add_argument("--delta-policy-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--delta-policy-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--delta-policy-gazing-ratio", type=float, default=0.75)
    parser.add_argument("--delta-policy-task-loss-requirement", type=float, default=0.7)
    parser.add_argument(
        "--hierarchical-coherence-metric",
        choices=list(HIERARCHICAL_COHERENCE_METRICS),
        default="within_diversity",
    )
    parser.add_argument("--hierarchical-proxy-base-metric", default="pixel_mse")
    parser.add_argument("--hierarchical-semantic-model", choices=["siglip", "dinov2"], default="dinov2")
    parser.add_argument("--hierarchical-target-leaf-tiles", type=int, default=16)
    parser.add_argument("--hierarchical-split-policy", choices=["best_first", "level_sync"], default="best_first")
    parser.add_argument("--hierarchical-max-depth", type=int, default=5)
    parser.add_argument("--hierarchical-min-interval-frames", type=int, default=64)
    parser.add_argument("--hierarchical-resize", type=int, default=224)
    parser.add_argument("--hierarchical-semantic-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hierarchical-semantic-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--hierarchical-max-batch-size-semantic", type=int, default=32)
    parser.add_argument("--hierarchical-autogaze-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hierarchical-autogaze-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--hierarchical-autogaze-gazing-ratio", type=float, default=0.75)
    parser.add_argument("--hierarchical-autogaze-task-loss-requirement", type=float, default=0.7)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def torch_dtype_from_name(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def read_video_metadata(video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    return {"frame_count": frame_count, "fps": fps, "duration": duration}


class LearnedDeltaPolicy:
    """Runtime wrapper for a supervised delta probe checkpoint.

    Question-conditioned policies are intentionally not supported here yet; this rollout uses
    tile-only AutoGaze features plus runtime metadata.
    """

    def __init__(
        self,
        checkpoint_path,
        autogaze_model_id,
        device,
        dtype_name,
        gazing_ratio,
        task_loss_requirement,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.checkpoint_path = str(checkpoint_path)
        self.deltas = [int(delta) for delta in checkpoint["deltas"]]
        self.feature_key = checkpoint.get("feature_key") or "mean_vision"
        self.input_mode = checkpoint["input_mode"]
        self.metadata_fields = list(checkpoint["metadata_fields"])
        self.feature_mean = checkpoint["feature_mean"].float()
        self.feature_std = checkpoint["feature_std"].float().clamp_min(1e-6)
        self.metadata_mean = checkpoint["metadata_mean"].float()
        self.metadata_std = checkpoint["metadata_std"].float().clamp_min(1e-6)
        self.device = device
        self.dtype = torch_dtype_from_name(dtype_name)
        self.gazing_ratio = gazing_ratio
        self.task_loss_requirement = task_loss_requirement

        self.probe = build_delta_model(
            checkpoint["architecture"],
            checkpoint["probe"],
            int(checkpoint["input_dim"]),
            self.deltas,
        )
        self.probe.load_state_dict(checkpoint["model_state_dict"])
        self.probe.to(device).eval()

        self.processor = AutoGazeImageProcessor.from_pretrained(autogaze_model_id)
        self.autogaze = AutoGaze.from_pretrained(autogaze_model_id).to(device=device, dtype=self.dtype)
        self.autogaze.eval()

    def _metadata_tensor(self, start, source_frame_count, effective_fps):
        row = []
        for field in self.metadata_fields:
            if field == "effective_fps":
                row.append(float(effective_fps or 0.0))
            elif field == "frame_count":
                row.append(float(source_frame_count or 0.0))
            elif field == "start":
                row.append(float(start))
            else:
                raise ValueError(
                    f"Delta policy metadata field {field!r} is not available at HLVid inference time."
                )
        return torch.tensor([row], dtype=torch.float32)

    @torch.inference_mode()
    def _extract_feature(self, tile_frames):
        np_video = np.stack([np.asarray(frame.convert("RGB")) for frame in tile_frames])
        video = transform_video_for_pytorch(np_video, self.processor)[None].to(device=self.device, dtype=self.dtype)
        gazing_info = self.autogaze(
            {"video": video},
            gazing_ratio=self.gazing_ratio,
            task_loss_requirement=self.task_loss_requirement,
            generate_only=True,
        )
        outputs = self.autogaze.gazing_model(video, gazing_info, output_hidden_states=True)
        features = representation_features(outputs.hidden_states[-1][0], gazing_info, [self.feature_key], batch_idx=0)
        return features[self.feature_key][None]

    @torch.inference_mode()
    def predict(self, tile_frames, start, source_frame_count, effective_fps):
        feature = (self._extract_feature(tile_frames) - self.feature_mean) / self.feature_std
        metadata = self._metadata_tensor(start, source_frame_count, effective_fps)
        metadata = (metadata - self.metadata_mean) / self.metadata_std
        if self.input_mode == "features":
            x = feature
        elif self.input_mode == "metadata":
            x = metadata
        elif self.input_mode == "concat":
            x = torch.cat([feature, metadata], dim=-1)
        else:
            raise ValueError(f"Unknown delta policy input_mode: {self.input_mode}")
        logits = self.probe(x.to(self.device)).float().cpu()[0]
        index = int(logits.argmax().item())
        return {
            "delta": int(self.deltas[index]),
            "delta_index": index,
            "logits": [float(value) for value in logits.tolist()],
        }


def load_uniform_frames(video_path, num_frames):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if cap.grab():
            break
        frame_count -= 1
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames_by_index = {}
    for index in indices:
        if int(index) in frames_by_index:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_by_index[int(index)] = Image.fromarray(frame)
    cap.release()

    frames = [frames_by_index[int(index)] for index in indices if int(index) in frames_by_index]
    if len(frames) < num_frames:
        if not frames:
            raise ValueError(f"Could not extract frames from video: {video_path}")
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


def load_frames_by_indices(video_path, indices):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames_by_index = {}
    for index in sorted({int(index) for index in indices}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_by_index[index] = Image.fromarray(frame)
    cap.release()

    frames = [frames_by_index[int(index)] for index in indices if int(index) in frames_by_index]
    if len(frames) < len(indices):
        if not frames:
            raise ValueError(f"Could not extract requested frames from video: {video_path}")
        frames.extend([frames[-1]] * (len(indices) - len(frames)))
    return frames


def select_inter_tile_delta_frames(frames, tile_len, delta_gap):
    if delta_gap < 0:
        raise ValueError("delta_gap must be non-negative")
    if tile_len <= 0:
        raise ValueError("tile_len must be positive")

    step = tile_len + delta_gap
    selected = []
    starts = []
    for start in range(0, len(frames) - tile_len + 1, step):
        starts.append(start)
        selected.extend(frames[start : start + tile_len])

    if not selected:
        selected = frames[:tile_len]
        starts = [0]
    return selected, starts


def select_learned_delta_frames(frames, tile_len, policy, video_meta, num_video_frames):
    if tile_len <= 0:
        raise ValueError("tile_len must be positive")

    duration = video_meta.get("duration")
    effective_fps = (num_video_frames / duration) if duration and duration > 0 else video_meta.get("fps")
    selected = []
    starts = []
    policy_records = []
    start = 0
    while start + tile_len <= len(frames):
        tile_frames = frames[start : start + tile_len]
        prediction = policy.predict(
            tile_frames,
            start=start,
            source_frame_count=len(frames),
            effective_fps=effective_fps,
        )
        starts.append(start)
        selected.extend(tile_frames)
        policy_records.append(
            {
                "start": int(start),
                "delta": int(prediction["delta"]),
                "delta_index": int(prediction["delta_index"]),
                "logits": prediction["logits"],
            }
        )
        start = start + tile_len + int(prediction["delta"])

    if not selected:
        selected = frames[:tile_len]
        starts = [0]
    return selected, starts, policy_records, effective_fps


def select_dense_inter_tile_delta_frames(video_path, tile_len, delta_gap, frame_stride, max_source_tiles):
    import cv2

    if frame_stride <= 0:
        raise ValueError("dense_frame_stride must be positive")
    if max_source_tiles is not None and max_source_tiles <= 0:
        raise ValueError("max_source_tiles must be positive when provided")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    step = tile_len + delta_gap
    source_positions = list(range(0, frame_count, frame_stride))
    tile_starts = []
    source_indices = []
    for start in range(0, len(source_positions) - tile_len + 1, step):
        tile_starts.append(start)
        source_indices.extend(source_positions[start : start + tile_len])
        if max_source_tiles is not None and len(tile_starts) >= max_source_tiles:
            break

    if not source_indices:
        tile_starts = [0]
        source_indices = source_positions[:tile_len]
    return load_frames_by_indices(video_path, source_indices), tile_starts, frame_count


def hierarchical_namespace(args):
    return argparse.Namespace(
        dataset_root=args.dataset_root,
        output=args.output,
        summary_output=args.summary_output,
        limit_videos=None,
        tile_len=args.tile_len,
        resize=args.hierarchical_resize,
        target_leaf_tiles=args.hierarchical_target_leaf_tiles,
        max_depth=args.hierarchical_max_depth,
        min_interval_frames=args.hierarchical_min_interval_frames,
        split_policy=args.hierarchical_split_policy,
        coherence_metric=args.hierarchical_coherence_metric,
        proxy_base_metric=args.hierarchical_proxy_base_metric,
        semantic_model=args.hierarchical_semantic_model,
        siglip_model=f"{Path(args.model_path).parent}/google--siglip2-base-patch16-224",
        dinov2_model=f"{Path(args.model_path).parent}/facebook--dinov2-with-registers-base",
        autogaze_model=args.autogaze_model_id,
        autogaze_device=args.hierarchical_autogaze_device,
        autogaze_dtype=args.hierarchical_autogaze_dtype,
        autogaze_gazing_ratio=args.hierarchical_autogaze_gazing_ratio,
        autogaze_task_loss_requirement=args.hierarchical_autogaze_task_loss_requirement,
        semantic_device=args.hierarchical_semantic_device,
        semantic_dtype=args.hierarchical_semantic_dtype,
        max_batch_size_semantic=args.hierarchical_max_batch_size_semantic,
    )


def select_hierarchical_frames(video_path, frame_count, scorer, hargs):
    leaves, decisions = build_hierarchy(
        video_path=video_path,
        frame_count=frame_count,
        args=hargs,
        scorer=scorer,
    )
    source_indices = [index for leaf in leaves for index in leaf.sampled_indices]
    return {
        "tile_starts": [int(leaf.start) for leaf in leaves],
        "source_indices": [int(index) for index in source_indices],
        "leaf_intervals": [
            {
                "start": int(leaf.start),
                "end": int(leaf.end),
                "depth": int(leaf.depth),
                "score": None if leaf.score is None else float(leaf.score),
                "sampled_indices": [int(index) for index in leaf.sampled_indices],
            }
            for leaf in leaves
        ],
        "split_decisions": decisions,
    }


def summarize(results_path):
    rows = []
    with Path(results_path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "num_examples": len(subset),
            "num_correct": sum(1 for row in subset if row["is_correct"]),
            "accuracy": sum(1 for row in subset if row["is_correct"]) / max(len(subset), 1),
        }

    visited_frames = [row.get("num_visited_frames") for row in rows if row.get("num_visited_frames") is not None]
    visited_tiles = [row.get("num_visited_tiles") for row in rows if row.get("num_visited_tiles") is not None]
    delta_counts = Counter()
    for row in rows:
        for policy_record in row.get("delta_policy_records") or []:
            delta_counts[str(policy_record["delta"])] += 1

    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": correct / max(total, 1),
        "by_category": by_category,
        "prediction_counts": dict(Counter(row["prediction_letter"] for row in rows)),
        "mean_visited_frames": float(np.mean(visited_frames)) if visited_frames else None,
        "mean_visited_tiles": float(np.mean(visited_tiles)) if visited_tiles else None,
        "delta_policy_counts": dict(delta_counts),
    }


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_root)
    rows = rows[args.start_index:]
    if args.limit is not None:
        rows = rows[: args.limit]

    done_ids = existing_question_ids(output_path) if args.resume else set()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        autogaze_model_id=args.autogaze_model_id,
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
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
    delta_policy = None
    if args.delta_policy_checkpoint is not None:
        if args.frame_source != "uniform":
            raise ValueError("--delta-policy-checkpoint currently supports --frame-source uniform only.")
        delta_policy = LearnedDeltaPolicy(
            checkpoint_path=args.delta_policy_checkpoint,
            autogaze_model_id=args.autogaze_model_id,
            device=args.delta_policy_device,
            dtype_name=args.delta_policy_dtype,
            gazing_ratio=args.delta_policy_gazing_ratio,
            task_loss_requirement=args.delta_policy_task_loss_requirement,
        )
    hierarchical_scorer = None
    hierarchical_args = None
    hierarchical_cache = {}
    if args.frame_source == "hierarchical":
        hierarchical_args = hierarchical_namespace(args)
        hierarchical_scorer = CoherenceScorer(hierarchical_args)

    video_token = processor.tokenizer.video_token
    with output_path.open("a") as out:
        for row in tqdm(rows, desc="HLVid"):
            if row["question_id"] in done_ids:
                continue

            video_path = dataset_root / "videos" / row["video_path"]
            if not video_path.exists():
                raise FileNotFoundError(video_path)

            prompt = f"{video_token}\n\nQuestion: {row['question']}"
            source_frame_count = None
            delta_policy_records = None
            delta_policy_effective_fps = None
            hierarchical_record = None
            if args.frame_source == "uniform":
                sampled_frames = load_uniform_frames(video_path, args.num_video_frames)
                if delta_policy is None:
                    visited_frames, tile_starts = select_inter_tile_delta_frames(
                        sampled_frames,
                        tile_len=args.tile_len,
                        delta_gap=args.delta_gap,
                    )
                else:
                    video_meta = read_video_metadata(video_path)
                    source_frame_count = video_meta["frame_count"]
                    visited_frames, tile_starts, delta_policy_records, delta_policy_effective_fps = select_learned_delta_frames(
                        sampled_frames,
                        tile_len=args.tile_len,
                        policy=delta_policy,
                        video_meta=video_meta,
                        num_video_frames=args.num_video_frames,
                    )
                num_source_frames = len(sampled_frames)
            elif args.frame_source == "dense":
                max_source_tiles = args.max_source_tiles
                if max_source_tiles is None:
                    max_source_tiles = args.max_tiles_video
                visited_frames, tile_starts, source_frame_count = select_dense_inter_tile_delta_frames(
                    video_path,
                    tile_len=args.tile_len,
                    delta_gap=args.delta_gap,
                    frame_stride=args.dense_frame_stride,
                    max_source_tiles=max_source_tiles,
                )
                num_source_frames = source_frame_count
            else:
                video_meta = read_video_metadata(video_path)
                source_frame_count = int(video_meta["frame_count"])
                cache_key = str(video_path)
                if cache_key not in hierarchical_cache:
                    hierarchical_cache[cache_key] = select_hierarchical_frames(
                        video_path,
                        source_frame_count,
                        hierarchical_scorer,
                        hierarchical_args,
                    )
                cached = hierarchical_cache[cache_key]
                visited_frames = load_frames_by_indices(video_path, cached["source_indices"])
                tile_starts = cached["tile_starts"]
                hierarchical_record = dict(cached)
                num_source_frames = source_frame_count
            inputs = processor(text=prompt, videos=[visited_frames], return_tensors="pt")
            target_device = model.device if hasattr(model, "device") else next(model.parameters()).device
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            response = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0].strip()
            pred = extract_letter(response)
            answer = str(row["answer"]).strip().upper()
            record = {
                "question_id": row["question_id"],
                "category": row["category"],
                "video_path": row["video_path"],
                "answer": answer,
                "prediction": response,
                "prediction_letter": pred,
                "is_correct": pred == answer,
                "delta_gap": args.delta_gap,
                "tile_len": args.tile_len,
                "frame_source": args.frame_source,
                "dense_frame_stride": args.dense_frame_stride,
                "max_source_tiles": args.max_source_tiles,
                "source_frame_count": source_frame_count,
                "num_sampled_frames": num_source_frames,
                "num_visited_frames": len(visited_frames),
                "visited_frame_ratio": len(visited_frames) / max(num_source_frames, 1),
                "tile_starts": tile_starts,
                "num_visited_tiles": len(tile_starts),
                "delta_policy_checkpoint": args.delta_policy_checkpoint,
                "delta_policy_records": delta_policy_records,
                "delta_policy_effective_fps": delta_policy_effective_fps,
                "hierarchical_coherence_metric": args.hierarchical_coherence_metric if args.frame_source == "hierarchical" else None,
                "hierarchical_semantic_model": args.hierarchical_semantic_model if args.frame_source == "hierarchical" else None,
                "hierarchical_target_leaf_tiles": args.hierarchical_target_leaf_tiles if args.frame_source == "hierarchical" else None,
                "hierarchical_split_policy": args.hierarchical_split_policy if args.frame_source == "hierarchical" else None,
                "hierarchical_record": hierarchical_record,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

    summary = summarize(output_path)
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "model_path": args.model_path,
            "autogaze_model_id": args.autogaze_model_id,
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "frame_source": args.frame_source,
            "dense_frame_stride": args.dense_frame_stride,
            "max_source_tiles": args.max_source_tiles,
            "tile_len": args.tile_len,
            "delta_gap": args.delta_gap,
            "delta_policy_checkpoint": args.delta_policy_checkpoint,
            "delta_policy_gazing_ratio": args.delta_policy_gazing_ratio,
            "delta_policy_task_loss_requirement": args.delta_policy_task_loss_requirement,
            "hierarchical_coherence_metric": args.hierarchical_coherence_metric if args.frame_source == "hierarchical" else None,
            "hierarchical_proxy_base_metric": args.hierarchical_proxy_base_metric if args.frame_source == "hierarchical" else None,
            "hierarchical_semantic_model": args.hierarchical_semantic_model if args.frame_source == "hierarchical" else None,
            "hierarchical_target_leaf_tiles": args.hierarchical_target_leaf_tiles if args.frame_source == "hierarchical" else None,
            "hierarchical_split_policy": args.hierarchical_split_policy if args.frame_source == "hierarchical" else None,
            "hierarchical_max_depth": args.hierarchical_max_depth if args.frame_source == "hierarchical" else None,
            "hierarchical_min_interval_frames": args.hierarchical_min_interval_frames if args.frame_source == "hierarchical" else None,
            "hierarchical_resize": args.hierarchical_resize if args.frame_source == "hierarchical" else None,
            "max_tiles_video": args.max_tiles_video,
            "max_batch_size_autogaze": args.max_batch_size_autogaze,
            "max_batch_size_siglip": args.max_batch_size_siglip,
            "torch_dtype": args.torch_dtype,
        }
    )
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
