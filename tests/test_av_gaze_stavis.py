# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from autogaze.datasets.av_gaze_stavis import (
    AVGazeStavisDataset,
    BalancedSourceSampler,
    STAVIS_SOURCES,
    build_manifest,
    cache_valid_cell_masses,
    heatmap_to_cell_mass,
    target_frame_numbers,
    write_manifest,
)


def _write_image(path: Path, value: int, rgb: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 8, 3) if rgb else (8, 8)
    Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)


def _synthetic_stavis(root: Path) -> None:
    (root / "fold_lists").mkdir(parents=True)
    for source in STAVIS_SOURCES:
        train_ids = [f"train_{index}" for index in range(4)]
        test_ids = [f"test_{index}" for index in range(2)]
        suffix = "fps" if source == "DIEM" else "1_fps"
        (root / "fold_lists" / f"{source}_list_train_{suffix}.txt").write_text(
            "".join(f"{video_id} 4 3.0\n" for video_id in train_ids), encoding="utf-8"
        )
        (root / "fold_lists" / f"{source}_list_test_{suffix}.txt").write_text(
            "".join(f"{video_id} 4 3.0\n" for video_id in test_ids), encoding="utf-8"
        )
        for video_id in train_ids + test_ids:
            for frame_number in range(1, 5):
                _write_image(
                    root / "video_frames" / source / video_id / f"img_{frame_number:05d}.jpg",
                    frame_number,
                    rgb=True,
                )
                _write_image(
                    root
                    / "annotations"
                    / source
                    / video_id
                    / "maps"
                    / f"eyeMap_{frame_number:05d}.jpg",
                    frame_number,
                    rgb=False,
                )


def test_target_frame_numbers_uses_nearest_native_frame() -> None:
    assert target_frame_numbers(31, native_fps=30.0, target_fps=3.0) == [1, 11, 21, 31]
    assert target_frame_numbers(26, native_fps=25.0, target_fps=3.0) == [1, 9, 18, 26]


def test_manifest_is_video_disjoint_and_excludes_missing_clip(tmp_path: Path) -> None:
    _synthetic_stavis(tmp_path)
    missing = tmp_path / "annotations" / "AVAD" / "test_0" / "maps" / "eyeMap_00004.jpg"
    missing.unlink()
    records, summary = build_manifest(
        tmp_path,
        fold=1,
        target_fps=3.0,
        clip_len=4,
        clip_stride=4,
        validation_fraction=0.25,
        split_seed=7,
    )
    assert summary["clips_by_split"] == {"train": 18, "val": 6, "test": 11}
    assert summary["sources"]["AVAD"]["excluded_clips"] == 1
    for source in STAVIS_SOURCES:
        split_ids = {
            split: {record["video_id"] for record in records if record["source"] == source and record["split"] == split}
            for split in ("train", "val", "test")
        }
        assert split_ids["train"].isdisjoint(split_ids["val"])
        assert split_ids["train"].isdisjoint(split_ids["test"])
        assert split_ids["val"].isdisjoint(split_ids["test"])


def test_dataset_loads_aligned_normalized_clip(tmp_path: Path) -> None:
    _synthetic_stavis(tmp_path)
    records, summary = build_manifest(
        tmp_path, clip_len=4, clip_stride=4, validation_fraction=0.25, split_seed=7
    )
    output = tmp_path / "manifest"
    write_manifest(records, summary, output)
    dataset = AVGazeStavisDataset(
        root=str(tmp_path),
        manifest_path=str(output / "clips.jsonl"),
        split="val",
        clip_len=4,
        load_rgb=True,
    )
    item = dataset[0]
    assert item["video"].shape == (4, 3, 224, 224)
    assert item["heatmap"].shape == (4, 224, 224)
    assert item["cell_mass"].shape == (4, 196)
    torch.testing.assert_close(item["heatmap"].sum(dim=(-2, -1)), torch.ones(4))
    torch.testing.assert_close(item["cell_mass"].sum(dim=-1), torch.ones(4))


def test_cell_mass_cache_excludes_zero_heatmap_clip(tmp_path: Path) -> None:
    _synthetic_stavis(tmp_path)
    zero_map = tmp_path / "annotations" / "AVAD" / "test_0" / "maps" / "eyeMap_00004.jpg"
    _write_image(zero_map, 0, rgb=False)
    records, _ = build_manifest(
        tmp_path, clip_len=4, clip_stride=4, validation_fraction=0.25, split_seed=7
    )
    valid, invalid = cache_valid_cell_masses(
        tmp_path, records, tmp_path / "targets", image_size=224, grid_size=14, num_workers=2
    )
    assert len(valid) == len(records) - 1
    assert invalid == [
        {
            "clip_id": "AVAD/test_0/target_000000",
            "reason": "zero_heatmap_mass",
            "frame_numbers": [4],
        }
    ]
    cached = np.load(tmp_path / "targets" / "cell_mass.npy")
    assert cached.shape == (len(valid), 4, 196)
    np.testing.assert_allclose(cached.sum(axis=-1), 1, atol=1e-5)


def test_heatmap_rejects_zero_mass() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        heatmap_to_cell_mass(torch.zeros(1, 224, 224))


class _SamplerDataset:
    def __init__(self) -> None:
        self.records = []
        for source in STAVIS_SOURCES:
            self.records.append({"source": source, "video_id": "short"})
            self.records.extend(
                {"source": source, "video_id": "long"} for _ in range(9)
            )
        self.sources = [record["source"] for record in self.records]

    def __len__(self) -> int:
        return len(self.sources)


def test_balanced_sampler_draws_sources_uniformly() -> None:
    dataset = _SamplerDataset()
    sampler = BalancedSourceSampler(dataset, num_samples=6000, seed=11)
    counts = Counter(dataset.sources[index] for index in sampler)
    assert set(counts) == set(STAVIS_SOURCES)
    assert all(abs(count - 1000) < 100 for count in counts.values())


def test_balanced_sampler_draws_videos_before_clips() -> None:
    dataset = _SamplerDataset()
    sampler = BalancedSourceSampler(dataset, num_samples=12000, seed=12)
    counts = Counter(
        (dataset.records[index]["source"], dataset.records[index]["video_id"])
        for index in sampler
    )
    for source in STAVIS_SOURCES:
        source_total = counts[(source, "short")] + counts[(source, "long")]
        short_fraction = counts[(source, "short")] / source_total
        assert 0.4 < short_fraction < 0.6
