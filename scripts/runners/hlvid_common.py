# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared HLVid evaluator utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch


ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


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


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def video_frame_count(video_path: Path) -> int:
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
    cap.release()
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    return frame_count


def video_frame_count_from_capture(video_path: Path) -> int:
    import cv2

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


def uniform_indices(frame_count: int, num_samples: int) -> np.ndarray:
    return np.round(np.linspace(0, frame_count - 1, num_samples)).astype(np.int64)


def load_video_samples(
    video_path: Path,
    sampling: str,
    tile_len: int,
    max_dense_sampled_frames: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    import cv2

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
