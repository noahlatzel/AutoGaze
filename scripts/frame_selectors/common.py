# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for external fixed-budget frame selectors."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY = REPO_ROOT / "experiments" / "frame_selectors" / "models.yaml"


def load_model_registry(path: Path | None = None) -> dict:
    registry_path = Path(os.environ.get("AUTOGAZE_FRAME_SELECTOR_MODEL_REGISTRY", path or DEFAULT_MODEL_REGISTRY))
    with registry_path.open() as f:
        return yaml.safe_load(f)


def model_path(model_key: str, path: Path | None = None) -> str:
    registry = load_model_registry(path)
    try:
        return str(registry["models"][model_key]["local_path"])
    except KeyError as exc:
        raise KeyError(f"Model key {model_key!r} is not defined in {path or DEFAULT_MODEL_REGISTRY}") from exc


def video_metadata(video_path: Path) -> tuple[int, float]:
    """Return (frame_count, fps) using ffprobe first, then PyAV."""

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,r_frame_rate,duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if streams:
            stream = streams[0]
            rate = stream.get("r_frame_rate") or "0/1"
            numerator, denominator = [float(part) for part in rate.split("/", maxsplit=1)]
            fps = numerator / max(denominator, 1.0)
            value = stream.get("nb_frames")
            if value and str(value).isdigit() and fps > 0:
                return int(value), fps
            duration = float(stream.get("duration") or 0.0)
            count = int(round(duration * fps))
            if count > 0 and fps > 0:
                return count, fps
    except Exception:
        pass

    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0.0)
        frame_count = int(stream.frames or 0)
        if frame_count <= 0:
            frame_count = sum(1 for _ in container.decode(stream))
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    if fps <= 0:
        raise ValueError(f"Video reports invalid FPS: {video_path}")
    return frame_count, fps


def video_frame_count(video_path: Path) -> int:
    return video_metadata(video_path)[0]


def load_pil_frames_av(video_path: Path, indices: Iterable[int]) -> list[Image.Image]:
    import av

    ordered_indices = [int(index) for index in indices]
    wanted = set(ordered_indices)
    frames_by_index = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in wanted:
                frames_by_index[frame_index] = frame.to_image()
                if len(frames_by_index) == len(wanted):
                    break

    frames = [frames_by_index[index] for index in ordered_indices if index in frames_by_index]
    if not frames:
        raise ValueError(f"Could not read requested frames from video: {video_path}")
    while len(frames) < len(ordered_indices):
        frames.append(frames[-1])
    return frames


def normalize_indices(indices: Iterable[int], frame_count: int | None = None) -> list[int]:
    normalized = sorted(int(index) for index in indices)
    if frame_count is not None:
        for index in normalized:
            if index < 0 or index >= frame_count:
                raise ValueError(f"Selected source index {index} outside valid range [0, {frame_count}).")
    return normalized


def selected_index_record(
    *,
    question_id: str,
    video_path: str,
    method: str,
    num_frames: int,
    selected_source_indices: Iterable[int],
    frame_count: int | None = None,
    metadata: dict | None = None,
) -> dict:
    indices = normalize_indices(selected_source_indices, frame_count=frame_count)
    return {
        "question_id": str(question_id),
        "video_path": str(video_path),
        "method": method,
        "num_frames": int(num_frames),
        "selected_source_indices": indices,
        "frame_count": frame_count,
        "metadata": metadata or {},
    }


def write_jsonl(records: Iterable[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_selected_index_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = str(record["question_id"])
            if question_id in records:
                raise ValueError(f"Duplicate question_id at line {line_no}: {question_id}")
            record["question_id"] = question_id
            record["selected_source_indices"] = [int(index) for index in record["selected_source_indices"]]
            records[question_id] = record
    return records
