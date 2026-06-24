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

from hlvid_common import dtype_from_name, existing_question_ids, extract_letter, load_rows


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
        choices=["uniform", "dense"],
        default="uniform",
        help=(
            "uniform: sample --num-video-frames over the full video, then apply tile skipping. "
            "dense: apply tile skipping in original video-frame coordinates and load only visited tiles."
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
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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

    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": correct / max(total, 1),
        "by_category": by_category,
        "prediction_counts": dict(Counter(row["prediction_letter"] for row in rows)),
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
            if args.frame_source == "uniform":
                sampled_frames = load_uniform_frames(video_path, args.num_video_frames)
                visited_frames, tile_starts = select_inter_tile_delta_frames(
                    sampled_frames,
                    tile_len=args.tile_len,
                    delta_gap=args.delta_gap,
                )
                num_source_frames = len(sampled_frames)
            else:
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
