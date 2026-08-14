# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest and aligned clip utilities for the AV-gaze-STAViS collection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


STAVIS_SOURCES: Tuple[str, ...] = (
    "AVAD",
    "Coutrot_db1",
    "Coutrot_db2",
    "DIEM",
    "ETMD_av",
    "SumMe",
)


@dataclass(frozen=True)
class FoldEntry:
    source: str
    video_id: str
    native_frame_count: int
    native_fps: float
    official_split: str


def _fold_list_path(root: Path, source: str, split: str, fold: int) -> Path:
    if source == "DIEM":
        name = f"{source}_list_{split}_fps.txt"
    else:
        name = f"{source}_list_{split}_{fold}_fps.txt"
    return root / "fold_lists" / name


def parse_fold_list(path: Path, source: str, official_split: str) -> List[FoldEntry]:
    entries: List[FoldEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected three fields, got {fields}")
            video_id, frame_count, fps = fields
            entry = FoldEntry(
                source=source,
                video_id=video_id,
                native_frame_count=int(frame_count),
                native_fps=float(fps),
                official_split=official_split,
            )
            if entry.native_frame_count <= 0 or entry.native_fps <= 0:
                raise ValueError(f"{path}:{line_number}: frame count and FPS must be positive")
            entries.append(entry)
    if not entries:
        raise ValueError(f"Fold list is empty: {path}")
    return entries


def target_frame_numbers(
    native_frame_count: int,
    native_fps: float,
    target_fps: float,
) -> List[int]:
    """Return one-based source-frame numbers nearest to a target-rate time grid."""
    if native_frame_count <= 0 or native_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame counts and frame rates must be positive")
    sampled: List[int] = []
    target_index = 0
    while True:
        zero_based = int(math.floor(target_index * native_fps / target_fps + 0.5))
        if zero_based >= native_frame_count:
            break
        frame_number = zero_based + 1
        if not sampled or sampled[-1] != frame_number:
            sampled.append(frame_number)
        target_index += 1
    return sampled


def _stable_validation_ids(
    entries: Sequence[FoldEntry],
    fraction: float,
    seed: int,
) -> set[str]:
    if not 0 <= fraction < 1:
        raise ValueError("Validation fraction must lie in [0, 1)")
    if not entries or fraction == 0:
        return set()
    ranked = sorted(
        entries,
        key=lambda entry: hashlib.sha256(
            f"{seed}:{entry.source}:{entry.video_id}".encode("utf-8")
        ).hexdigest(),
    )
    count = max(1, int(round(len(ranked) * fraction)))
    count = min(count, len(ranked) - 1) if len(ranked) > 1 else 0
    return {entry.video_id for entry in ranked[:count]}


def _count_files(directory: Path, prefix: str, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for entry in os.scandir(directory)
        if entry.is_file() and entry.name.startswith(prefix) and entry.name.endswith(suffix)
    )


def _relative_frame_path(source: str, video_id: str, frame_number: int) -> str:
    return f"video_frames/{source}/{video_id}/img_{frame_number:05d}.jpg"


def _relative_map_path(source: str, video_id: str, frame_number: int) -> str:
    return f"annotations/{source}/{video_id}/maps/eyeMap_{frame_number:05d}.jpg"


def build_manifest(
    root: Path,
    fold: int = 1,
    target_fps: float = 3.0,
    clip_len: int = 16,
    clip_stride: int = 16,
    validation_fraction: float = 0.2,
    split_seed: int = 140826,
) -> Tuple[List[dict], dict]:
    """Build clip records and an audit summary without modifying source data."""
    root = Path(root)
    if clip_len <= 0 or clip_stride <= 0:
        raise ValueError("Clip length and stride must be positive")

    records: List[dict] = []
    source_summaries: Dict[str, dict] = {}
    split_videos: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for source in STAVIS_SOURCES:
        train_entries = parse_fold_list(
            _fold_list_path(root, source, "train", fold), source, "train"
        )
        test_entries = parse_fold_list(
            _fold_list_path(root, source, "test", fold), source, "test"
        )
        train_ids = {entry.video_id for entry in train_entries}
        test_ids = {entry.video_id for entry in test_entries}
        overlap = sorted(train_ids & test_ids)
        if overlap:
            raise ValueError(f"{source} fold {fold} train/test overlap: {overlap}")

        val_ids = _stable_validation_ids(train_entries, validation_fraction, split_seed)
        all_entries = train_entries + test_entries
        listed_ids = train_ids | test_ids
        frame_root = root / "video_frames" / source
        annotation_root = root / "annotations" / source
        local_ids = {entry.name for entry in os.scandir(frame_root) if entry.is_dir()}

        source_summary = {
            "listed_train_videos": len(train_entries),
            "listed_test_videos": len(test_entries),
            "validation_videos": sorted(val_ids),
            "unlisted_local_videos": sorted(local_ids - listed_ids),
            "listed_missing_local_videos": sorted(listed_ids - local_ids),
            "videos": {},
            "clips_by_split": {"train": 0, "val": 0, "test": 0},
            "excluded_clips": 0,
        }

        for entry in all_entries:
            if entry.official_split == "test":
                split = "test"
            elif entry.video_id in val_ids:
                split = "val"
            else:
                split = "train"
            split_videos[source][split].add(entry.video_id)

            rgb_dir = frame_root / entry.video_id
            map_dir = annotation_root / entry.video_id / "maps"
            observed_rgb = _count_files(rgb_dir, "img_", ".jpg")
            observed_maps = _count_files(map_dir, "eyeMap_", ".jpg")
            sampled = target_frame_numbers(
                entry.native_frame_count, entry.native_fps, target_fps
            )
            video_summary = {
                "official_split": entry.official_split,
                "split": split,
                "listed_frames": entry.native_frame_count,
                "observed_rgb_frames": observed_rgb,
                "observed_heatmaps": observed_maps,
                "native_fps": entry.native_fps,
                "target_grid_frames": len(sampled),
                "valid_clips": 0,
                "excluded_clips": 0,
                "missing_sampled_frames": [],
            }

            for target_start in range(0, len(sampled) - clip_len + 1, clip_stride):
                frame_numbers = sampled[target_start : target_start + clip_len]
                missing: List[dict] = []
                for frame_number in frame_numbers:
                    rgb_path = root / _relative_frame_path(source, entry.video_id, frame_number)
                    map_path = root / _relative_map_path(source, entry.video_id, frame_number)
                    if not rgb_path.is_file() or not map_path.is_file():
                        missing.append(
                            {
                                "frame_number": frame_number,
                                "rgb": rgb_path.is_file(),
                                "heatmap": map_path.is_file(),
                            }
                        )
                if missing:
                    video_summary["excluded_clips"] += 1
                    source_summary["excluded_clips"] += 1
                    video_summary["missing_sampled_frames"].extend(missing)
                    continue

                record = {
                    "schema_version": 1,
                    "clip_id": f"{source}/{entry.video_id}/target_{target_start:06d}",
                    "source": source,
                    "video_id": entry.video_id,
                    "official_split": entry.official_split,
                    "split": split,
                    "fold": None if source == "DIEM" else fold,
                    "native_fps": entry.native_fps,
                    "native_frame_count": entry.native_frame_count,
                    "target_fps": target_fps,
                    "target_start_index": target_start,
                    "frame_numbers": frame_numbers,
                    "timestamps_seconds": [index / target_fps for index in range(target_start, target_start + clip_len)],
                    "spatial_transform": {
                        "kind": "direct_full_field_resize",
                        "height": 224,
                        "width": 224,
                        "resample": "bilinear",
                    },
                }
                records.append(record)
                video_summary["valid_clips"] += 1
                source_summary["clips_by_split"][split] += 1

            source_summary["videos"][entry.video_id] = video_summary

        source_summaries[source] = source_summary

    for source, by_split in split_videos.items():
        splits = list(by_split)
        for index, left in enumerate(splits):
            for right in splits[index + 1 :]:
                overlap = by_split[left] & by_split[right]
                if overlap:
                    raise AssertionError(f"{source}: {left}/{right} video leakage: {overlap}")

    summary = {
        "schema_version": 1,
        "source_root": str(root),
        "fold": fold,
        "target_fps": target_fps,
        "clip_len": clip_len,
        "clip_stride": clip_stride,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "spatial_transform": {
            "kind": "direct_full_field_resize",
            "height": 224,
            "width": 224,
            "resample": "bilinear",
        },
        "sources": source_summaries,
        "total_clips": len(records),
        "clips_by_split": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "val", "test")
        },
    }
    return records, summary


def write_manifest(records: Sequence[Mapping], summary: Mapping, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "clips.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_manifest(path: Path, split: Optional[str] = None) -> List[dict]:
    records: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if record.get("schema_version") != 1:
                raise ValueError(f"{path}:{line_number}: unsupported schema version")
            if split is None or record["split"] == split:
                records.append(record)
    return records


def heatmap_to_cell_mass(heatmap: torch.Tensor, grid_size: int = 14) -> torch.Tensor:
    """Normalize heatmaps and sum their mass into a square grid."""
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0)
    if heatmap.ndim != 3:
        raise ValueError(f"Expected [T,H,W] or [H,W] heatmap, got {tuple(heatmap.shape)}")
    time, height, width = heatmap.shape
    if height % grid_size or width % grid_size:
        raise ValueError(f"Heatmap shape {(height, width)} is not divisible by {grid_size}")
    if not torch.isfinite(heatmap).all() or (heatmap < 0).any():
        raise ValueError("Heatmaps must be finite and nonnegative")
    mass = heatmap.sum(dim=(-2, -1), keepdim=True)
    if (mass <= 0).any():
        raise ValueError("Heatmaps must contain positive mass")
    normalized = heatmap / mass
    cell_height = height // grid_size
    cell_width = width // grid_size
    cells = normalized.reshape(time, grid_size, cell_height, grid_size, cell_width)
    cells = cells.sum(dim=(2, 4)).reshape(time, grid_size * grid_size)
    return cells


def load_aligned_clip(
    root: Path,
    record: Mapping,
    image_size: int = 224,
    load_rgb: bool = True,
    load_heatmap: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load a clip with identical full-field geometry for RGB and heatmaps."""
    rgb_frames: List[np.ndarray] = []
    heatmaps: List[np.ndarray] = []
    for frame_number in record["frame_numbers"]:
        if load_rgb:
            rgb_path = Path(root) / _relative_frame_path(
                record["source"], record["video_id"], frame_number
            )
            with Image.open(rgb_path) as image:
                resized = image.convert("RGB").resize(
                    (image_size, image_size), resample=Image.Resampling.BILINEAR
                )
                rgb_frames.append(np.asarray(resized, dtype=np.uint8).copy())
        if load_heatmap:
            map_path = Path(root) / _relative_map_path(
                record["source"], record["video_id"], frame_number
            )
            with Image.open(map_path) as image:
                resized = image.convert("L").resize(
                    (image_size, image_size), resample=Image.Resampling.BILINEAR
                )
                heatmaps.append(np.asarray(resized, dtype=np.float32).copy())

    rgb_tensor: Optional[torch.Tensor] = None
    if load_rgb:
        rgb_tensor = torch.from_numpy(np.stack(rgb_frames)).permute(0, 3, 1, 2)
    heatmap_tensor = torch.from_numpy(np.stack(heatmaps)) if load_heatmap else None
    return rgb_tensor, heatmap_tensor


def cache_valid_cell_masses(
    root: Path,
    records: Sequence[Mapping],
    output_dir: Path,
    image_size: int = 224,
    grid_size: int = 14,
    num_workers: int = 8,
) -> Tuple[List[dict], List[dict]]:
    """Exclude invalid heatmap clips and cache their fine-grid masses once."""
    root = Path(root)
    output_dir = Path(output_dir)

    def load_one(index_record: Tuple[int, Mapping]):
        index, record = index_record
        try:
            _, heatmap = load_aligned_clip(
                root, record, image_size=image_size, load_rgb=False, load_heatmap=True
            )
            frame_mass = heatmap.sum(dim=(-2, -1))
            zero_frames = [
                int(record["frame_numbers"][position])
                for position in torch.nonzero(frame_mass <= 0, as_tuple=True)[0]
            ]
            if zero_frames:
                return index, None, {"reason": "zero_heatmap_mass", "frame_numbers": zero_frames}
            cells = heatmap_to_cell_mass(heatmap, grid_size=grid_size).numpy().astype(np.float32)
            return index, cells, None
        except Exception as error:
            return index, None, {"reason": "heatmap_decode_or_validation_error", "error": repr(error)}

    iterator = enumerate(records)
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            loaded = list(executor.map(load_one, iterator))
    else:
        loaded = [load_one(value) for value in iterator]

    valid_records: List[dict] = []
    invalid_records: List[dict] = []
    cached: List[np.ndarray] = []
    for index, cells, error in loaded:
        record = dict(records[index])
        if error is not None:
            invalid_records.append({"clip_id": record["clip_id"], **error})
            continue
        record["cell_mass_index"] = len(cached)
        valid_records.append(record)
        cached.append(cells)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cell_mass.npy", np.stack(cached))
    with (output_dir / "invalid_clips.json").open("w", encoding="utf-8") as handle:
        json.dump(invalid_records, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return valid_records, invalid_records


class AVGazeStavisDataset(Dataset):
    """Manifest-driven aligned AV-gaze-STAViS clips."""

    def __init__(
        self,
        root: str,
        manifest_path: str,
        split: str,
        clip_len: int = 16,
        image_size: int = 224,
        grid_size: int = 14,
        load_rgb: bool = True,
        load_heatmap: bool = True,
        cell_mass_path: Optional[str] = None,
        image_processor=None,
        gaze_transform=None,
        task_transform=None,
    ) -> None:
        self.root = Path(root)
        self.records = read_manifest(Path(manifest_path), split=split)
        self.split = split
        self.clip_len = clip_len
        self.image_size = image_size
        self.grid_size = grid_size
        self.load_rgb = load_rgb
        self.load_heatmap = load_heatmap
        self.image_processor = image_processor if image_processor is not None else gaze_transform
        self.cell_mass = (
            np.load(cell_mass_path, mmap_mode="r") if cell_mass_path is not None else None
        )
        self.sources = [record["source"] for record in self.records]
        if not self.records:
            raise ValueError(f"No records for split '{split}' in {manifest_path}")
        invalid_lengths = [
            record["clip_id"]
            for record in self.records
            if len(record["frame_numbers"]) != self.clip_len
        ]
        if invalid_lengths:
            raise ValueError(
                f"Expected {self.clip_len}-frame clips; found {len(invalid_lengths)} mismatches"
            )
        if self.cell_mass is None and not self.load_heatmap:
            raise ValueError("load_heatmap=False requires cell_mass_path")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rgb, heatmap = load_aligned_clip(
            self.root,
            record,
            image_size=self.image_size,
            load_rgb=self.load_rgb,
            load_heatmap=self.load_heatmap,
        )
        if self.cell_mass is None:
            cell_mass = heatmap_to_cell_mass(heatmap, grid_size=self.grid_size)
        else:
            cache_index = record.get("cell_mass_index")
            if cache_index is None:
                raise ValueError(f"Manifest record lacks cell_mass_index: {record['clip_id']}")
            cell_mass = torch.from_numpy(np.array(self.cell_mass[cache_index], copy=True))
        item = {
            "cell_mass": cell_mass,
            "clip_id": record["clip_id"],
            "source": record["source"],
            "video_id": record["video_id"],
            "frame_numbers": torch.tensor(record["frame_numbers"], dtype=torch.long),
            "timestamps_seconds": torch.tensor(record["timestamps_seconds"], dtype=torch.float32),
        }
        if heatmap is not None:
            item["heatmap"] = heatmap / heatmap.sum(dim=(-2, -1), keepdim=True)
        if rgb is not None:
            if self.image_processor is None:
                item["video"] = rgb.to(torch.float32) / 255.0
            else:
                frames = [frame.permute(1, 2, 0).numpy() for frame in rgb]
                values = self.image_processor(frames, return_tensors="pt").pixel_values
                item["video"] = values[0] if values.ndim == 5 else values
        return item


class BalancedSourceSampler(Sampler[int]):
    """Sample sources uniformly, then clips uniformly within each source."""

    def __init__(
        self,
        dataset: AVGazeStavisDataset,
        num_samples: Optional[int] = None,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.num_replicas = num_replicas
        self.rank = rank
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must lie within num_replicas")
        default_samples = int(math.ceil(len(dataset) / num_replicas))
        self.num_samples = default_samples if num_samples is None else num_samples
        self.seed = seed
        self.epoch = 0
        by_source_video: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, record in enumerate(dataset.records):
            by_source_video[record["source"]][record["video_id"]].append(index)
        missing = set(STAVIS_SOURCES) - set(by_source_video)
        if missing:
            raise ValueError(f"Balanced sampling requires all six sources; missing {sorted(missing)}")
        self.by_source_video = {
            source: {
                video_id: indices
                for video_id, indices in sorted(by_source_video[source].items())
            }
            for source in STAVIS_SOURCES
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_indices = []
        for _ in range(self.num_samples * self.num_replicas):
            source_index = int(torch.randint(len(STAVIS_SOURCES), (1,), generator=generator))
            videos = self.by_source_video[STAVIS_SOURCES[source_index]]
            video_ids = tuple(videos)
            video_index = int(torch.randint(len(video_ids), (1,), generator=generator))
            indices = videos[video_ids[video_index]]
            clip_index = int(torch.randint(len(indices), (1,), generator=generator))
            global_indices.append(indices[clip_index])
        yield from global_indices[self.rank :: self.num_replicas]
