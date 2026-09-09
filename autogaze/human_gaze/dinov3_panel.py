# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze an outcome-independent validation panel before RGB or policy inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES, load_aligned_clip, read_manifest


POPULATION_MANIFEST_SHA256 = "0ef9fa17881517ea69d9edbaf57f8571c5c58d1da517370746fbe264fe33ff10"
PANEL_SALT = "wp2-dinov3-bridge-v1"
FRAMES_PER_CLIP = 16
VIDEOS_PER_SOURCE = 2
SPATIAL_TRANSFORM = {"kind": "direct_full_field_resize", "height": 224, "width": 224, "resample": "bilinear"}
SELECTION_RULE = (
    "Within each source, rank validation clips by SHA256(UTF-8(salt + NUL + clip_id)), "
    "then clip_id as tie-breaker; select the first two distinct video IDs, one clip per video. "
    "Return records in ascending (source, video_id, clip_id, frame_numbers) order."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identity_record(record: Mapping) -> dict:
    if record.get("split") != "val":
        raise ValueError("Panel selection accepts validation records only.")
    source, video_id, clip_id = (record.get(field) for field in ("source", "video_id", "clip_id"))
    if source not in STAVIS_SOURCES:
        raise ValueError(f"Unknown STAViS source: {source!r}")
    if (
        not isinstance(video_id, str) or not video_id or video_id in (".", "..")
        or any(character in video_id for character in ("/", "\\", "\0", ":"))
    ):
        raise ValueError("Original video_id must be one unambiguous directory component.")
    if not isinstance(clip_id, str) or not clip_id or "\0" in clip_id:
        raise ValueError("clip_id must be a nonempty string without NUL characters.")
    frames = record.get("frame_numbers")
    if (
        not isinstance(frames, (list, tuple)) or len(frames) != FRAMES_PER_CLIP
        or any(type(value) is not int or value < 1 for value in frames)
        or any(right <= left for left, right in zip(frames, frames[1:]))
    ):
        raise ValueError(f"{clip_id}: expected exactly 16 strictly increasing one-based frame numbers.")
    timestamps = record.get("timestamps_seconds")
    if (
        not isinstance(timestamps, (list, tuple)) or len(timestamps) != FRAMES_PER_CLIP
        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in timestamps)
        or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
    ):
        raise ValueError(f"{clip_id}: expected 16 finite, strictly increasing timestamps_seconds.")
    if record.get("spatial_transform") != SPATIAL_TRANSFORM:
        raise ValueError(f"{clip_id}: population must declare the original full-field 224x224 bilinear transform.")
    # Deliberately do not copy coverage, heatmap summaries, selector output, or
    # other optional fields: none can influence ranking or the frozen payload.
    return {"schema_version": 1, "split": "val", "source": source, "video_id": video_id,
            "clip_id": clip_id, "frame_numbers": list(frames), "timestamps_seconds": list(timestamps),
            "spatial_transform": dict(SPATIAL_TRANSFORM)}


def select_panel(
    records: Sequence[Mapping], *, salt: str = PANEL_SALT, videos_per_source: int = VIDEOS_PER_SOURCE,
) -> list[dict]:
    """Select unique videos by the predeclared clip-hash ranking, never outcomes."""
    if not isinstance(salt, str) or not salt or "\0" in salt:
        raise ValueError("Selection salt must be a nonempty string without NUL characters.")
    if type(videos_per_source) is not int or videos_per_source < 1:
        raise ValueError("videos_per_source must be a positive integer.")
    by_source = defaultdict(list)
    seen_clip_ids, seen_frame_sequences = set(), set()
    for raw_record in records:
        record = _identity_record(raw_record)
        identity = (record["source"], record["video_id"], tuple(record["frame_numbers"]))
        if record["clip_id"] in seen_clip_ids:
            raise ValueError(f"Duplicate or ambiguous clip_id: {record['clip_id']}")
        if identity in seen_frame_sequences:
            raise ValueError(f"Ambiguous duplicate video/frame sequence: {record['clip_id']}")
        seen_clip_ids.add(record["clip_id"])
        seen_frame_sequences.add(identity)
        by_source[record["source"]].append(record)
    missing = set(STAVIS_SOURCES) - set(by_source)
    if missing:
        raise ValueError(f"Panel requires all six sources; missing {sorted(missing)}")

    selected = []
    for source in STAVIS_SOURCES:
        source_records = by_source[source]
        available = {record["video_id"] for record in source_records}
        if len(available) < videos_per_source:
            raise ValueError(f"{source}: need {videos_per_source} distinct validation videos; found {len(available)}")
        ranked = sorted(source_records, key=lambda record: (
            hashlib.sha256((salt + "\0" + record["clip_id"]).encode("utf-8")).hexdigest(), record["clip_id"]
        ))
        chosen_videos = set()
        for record in ranked:
            if record["video_id"] in chosen_videos:
                continue
            selected.append(record)
            chosen_videos.add(record["video_id"])
            if len(chosen_videos) == videos_per_source:
                break
    return sorted(selected, key=lambda record: (
        record["source"], record["video_id"], record["clip_id"], tuple(record["frame_numbers"])
    ))


def _write_new_json(path: Path, payload: Mapping) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def freeze_panel(
    manifest_path: Path, dataset_root: Path, output_dir: Path, *,
    expected_sha: str = POPULATION_MANIFEST_SHA256,
) -> dict:
    """Write a new 12-video panel and RGB cache; return the frozen RGB manifest.

    ``panel.json`` is closed before any selected RGB is opened. Missing or invalid
    RGB aborts without replacement selection; partial directories are preserved
    and are never overwritten. Only a complete export gets ``rgb_manifest.json``.
    ``expected_sha`` exists for CPU fixtures; production callers must retain the
    pinned default population hash.
    """
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    dataset_root = Path(dataset_root).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing panel directory: {output_dir}")
    if not dataset_root.is_dir():
        raise ValueError("dataset_root must be a directory.")
    population_sha = sha256_file(manifest_path)
    if population_sha != expected_sha:
        raise ValueError(f"Population SHA-256 mismatch: expected {expected_sha}, observed {population_sha}")
    records = read_manifest(manifest_path, split="val")
    selected = select_panel(records)
    if sha256_file(manifest_path) != population_sha:
        raise RuntimeError("Population manifest changed during selection.")

    selected_utc = datetime.now(timezone.utc).isoformat()
    counts = {source: {"available_validation_clips": sum(record["source"] == source for record in records),
                       "available_validation_videos": len({record["video_id"] for record in records if record["source"] == source}),
                       "selected_clips": VIDEOS_PER_SOURCE, "selected_videos": VIDEOS_PER_SOURCE}
              for source in STAVIS_SOURCES}
    provenance = {
        "panel_frozen_utc": selected_utc, "population_manifest_path": str(manifest_path),
        "population_manifest_sha256": population_sha, "dataset_root": str(dataset_root),
        "split": "val", "selection_salt": PANEL_SALT, "selection_rule": SELECTION_RULE,
        "videos_per_source": VIDEOS_PER_SOURCE, "frames_per_clip": FRAMES_PER_CLIP,
        "source_counts": counts, "frame_number_space": "One-based source JPEG frame numbers, preserved from population manifest.",
        "timestamps_seconds_space": "Population target-rate timestamps, copied without recomputation.",
        "rgb_transform": dict(SPATIAL_TRANSFORM), "heatmaps_loaded": False,
    }
    panel = {"schema_version": 1, "status": "selected_before_rgb", "provenance": provenance, "clips": selected}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "rgb").mkdir()
    _write_new_json(output_dir / "panel.json", panel)

    rgb_clips = []
    for index, record in enumerate(selected):
        rgb, heatmap = load_aligned_clip(dataset_root, record, image_size=224, load_rgb=True, load_heatmap=False)
        if heatmap is not None:
            raise RuntimeError("RGB-only export unexpectedly loaded a heatmap.")
        if not isinstance(rgb, torch.Tensor) or rgb.dtype != torch.uint8 or tuple(rgb.shape) != (16, 3, 224, 224):
            raise ValueError(f"{record['clip_id']}: expected uint8 RGB [16,3,224,224].")
        array = np.ascontiguousarray(rgb.cpu().numpy())
        relative_path = f"rgb/clip_{index:03d}.npy"
        rgb_path = output_dir / relative_path
        with rgb_path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        rgb_clips.append({
            "source": record["source"], "video_id": f"{record['source']}/{record['video_id']}",
            "original_video_id": record["video_id"], "clip_id": record["clip_id"], "split": "val",
            "frame_numbers": record["frame_numbers"], "timestamps_seconds": record["timestamps_seconds"],
            "rgb_path": relative_path, "rgb_sha256": sha256_file(rgb_path),
            "rgb_dtype": "uint8", "rgb_shape": list(array.shape), "rgb_c_contiguous": True,
            "rgb_array_nbytes": array.nbytes, "rgb_file_nbytes": rgb_path.stat().st_size,
        })
    payload_sha = canonical_payload_sha256({"clips": rgb_clips})
    frozen = {
        "schema_version": 1, "status": "frozen", "provenance": {
            **provenance, "rgb_export_completed_utc": datetime.now(timezone.utc).isoformat(),
            "panel_json_sha256": sha256_file(output_dir / "panel.json"),
            "panel_payload_sha256_rule": "SHA256 of UTF-8 json.dumps({'clips': clips}, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).",
        },
        "panel_payload_sha256": payload_sha, "clips": rgb_clips,
    }
    _write_new_json(output_dir / "rgb_manifest.json", frozen)
    return frozen
