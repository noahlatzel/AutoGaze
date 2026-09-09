"""Frozen population, identity ranking and RGB-only cache invariants."""

import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pytest
import torch

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
from autogaze.human_gaze import dinov3_panel as panel


def population(videos=3, clips=2):
    records = []
    for source in STAVIS_SOURCES:
        for video in range(videos):
            for clip in range(clips):
                start = clip * 16
                records.append({
                    "schema_version": 1, "source": source, "video_id": f"video_{video}", "split": "val",
                    "clip_id": f"{source}/video_{video}/target_{start:06d}",
                    "frame_numbers": list(range(start + 1, start + 17)),
                    "timestamps_seconds": [value / 3 for value in range(start, start + 16)],
                    "spatial_transform": dict(panel.SPATIAL_TRANSFORM),
                })
    return records


def write_population(tmp_path, records):
    path = tmp_path / "clips.jsonl"
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    return path


def test_selection_is_independent_of_population_order_and_all_outcome_fields():
    records = population()
    expected = panel.select_panel(records)
    changed = copy.deepcopy(records[::-1])
    for index, record in enumerate(changed):
        record.update(coverage=1e9 - index, fidelity=index, selector_outputs={"best": index}, unknown=object())
    assert panel.select_panel(changed) == expected
    assert len(expected) == 12
    assert Counter(record["source"] for record in expected) == {source: 2 for source in STAVIS_SOURCES}
    assert len({(record["source"], record["video_id"]) for record in expected}) == 12
    assert all("coverage" not in record and "unknown" not in record for record in expected)
    identities = [(record["source"], record["video_id"], record["clip_id"]) for record in expected]
    assert identities == sorted(identities)


def test_frozen_hash_rule_selects_first_distinct_videos_not_first_two_clips():
    records = population(videos=3, clips=8)
    selected = panel.select_panel(records)
    for source in STAVIS_SOURCES:
        ranked = sorted(
            (record for record in records if record["source"] == source),
            key=lambda record: (hashlib.sha256(b"wp2-dinov3-bridge-v1\x00" + record["clip_id"].encode()).hexdigest(),
                                record["clip_id"]),
        )
        first = ranked[0]
        second = next(record for record in ranked if record["video_id"] != first["video_id"])
        assert {record["clip_id"] for record in selected if record["source"] == source} == {
            first["clip_id"], second["clip_id"]}


@pytest.mark.parametrize("defect", ["duplicate_clip", "ambiguous_frames", "non_val", "short_clip", "duplicate_frame",
                                         "zero_frame", "missing_source", "too_few_videos", "unknown_source",
                                         "path_video", "bad_timestamps", "wrong_transform"])
def test_rejects_population_identity_or_geometry_defects(defect):
    records = population()
    if defect == "duplicate_clip":
        records.append(copy.deepcopy(records[0]))
    elif defect == "ambiguous_frames":
        duplicate = copy.deepcopy(records[0])
        duplicate["clip_id"] += "_alias"
        records.append(duplicate)
    elif defect == "non_val":
        records[0]["split"] = "test"
    elif defect == "short_clip":
        records[0]["frame_numbers"].pop()
    elif defect == "duplicate_frame":
        records[0]["frame_numbers"][1] = records[0]["frame_numbers"][0]
    elif defect == "zero_frame":
        records[0]["frame_numbers"][0] = 0
    elif defect == "missing_source":
        records = [record for record in records if record["source"] != STAVIS_SOURCES[0]]
    elif defect == "too_few_videos":
        records = [record for record in records if record["source"] != STAVIS_SOURCES[0] or record["video_id"] == "video_0"]
    elif defect == "unknown_source":
        records[0]["source"] = "unknown"
    elif defect == "path_video":
        records[0]["video_id"] = "../outside"
    elif defect == "bad_timestamps":
        records[0]["timestamps_seconds"][5] = float("nan")
    elif defect == "wrong_transform":
        records[0]["spatial_transform"]["kind"] = "center_crop"
    with pytest.raises(ValueError):
        panel.select_panel(records)


def test_freeze_records_selection_before_rgb_and_exports_exact_uint8_bytes(tmp_path, monkeypatch):
    records = population()
    # Other splits are present in the pinned population but cannot enter the panel.
    excluded = copy.deepcopy(records[0])
    excluded.update(split="test", clip_id="test_only", video_id="test_only")
    path = write_population(tmp_path, [excluded, *records])
    output = tmp_path / "frozen"
    seen = []

    def load_rgb_only(root, record, *, image_size, load_rgb, load_heatmap):
        assert root == tmp_path.resolve()
        assert image_size == 224 and load_rgb is True and load_heatmap is False
        assert (output / "panel.json").is_file()
        assert not (output / "rgb_manifest.json").exists()
        frozen_panel = json.loads((output / "panel.json").read_text(encoding="utf-8"))
        assert frozen_panel["status"] == "selected_before_rgb"
        assert len(frozen_panel["clips"]) == 12
        assert frozen_panel["provenance"]["selection_salt"] == panel.PANEL_SALT
        seen.append(record["clip_id"])
        # Intentionally noncontiguous channels-first view, as in the real loader.
        pixels = torch.empty((16, 224, 224, 3), dtype=torch.uint8)
        pixels[..., 0] = 17
        pixels[..., 1] = 83
        pixels[..., 2] = 201
        return pixels.permute(0, 3, 1, 2), None

    monkeypatch.setattr(panel, "load_aligned_clip", load_rgb_only)
    result = panel.freeze_panel(path, tmp_path, output, expected_sha=panel.sha256_file(path))
    assert result["status"] == "frozen"
    assert seen == [record["clip_id"] for record in panel.select_panel(records)]
    assert json.loads((output / "rgb_manifest.json").read_text(encoding="utf-8")) == result
    assert result["panel_payload_sha256"] == panel.canonical_payload_sha256({"clips": result["clips"]})
    for clip in result["clips"]:
        assert clip["video_id"] == f"{clip['source']}/{clip['original_video_id']}"
        rgb_path = output / clip["rgb_path"]
        assert panel.sha256_file(rgb_path) == clip["rgb_sha256"]
        pixels = np.load(rgb_path, allow_pickle=False)
        assert pixels.dtype == np.uint8 and pixels.shape == (16, 3, 224, 224)
        assert pixels.flags.c_contiguous
        assert pixels[:, 0].min() == pixels[:, 0].max() == 17
        assert pixels[:, 1].min() == pixels[:, 1].max() == 83
        assert pixels[:, 2].min() == pixels[:, 2].max() == 201
        assert clip["rgb_array_nbytes"] == pixels.nbytes
    for counts in result["provenance"]["source_counts"].values():
        assert counts == {"available_validation_clips": 6, "available_validation_videos": 3,
                          "selected_clips": 2, "selected_videos": 2}
    assert result["provenance"]["panel_json_sha256"] == panel.sha256_file(output / "panel.json")


def test_wrong_population_hash_creates_no_output_or_rgb_reads(tmp_path, monkeypatch):
    path = write_population(tmp_path, population())
    output = tmp_path / "wrong_hash"
    monkeypatch.setattr(panel, "load_aligned_clip", lambda *args, **kwargs: pytest.fail("RGB must not be opened"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        panel.freeze_panel(path, tmp_path, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    path = write_population(tmp_path, population())
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "panel.json"
    sentinel.write_text("preserve this", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        panel.freeze_panel(path, tmp_path, output, expected_sha=panel.sha256_file(path))
    assert sentinel.read_text(encoding="utf-8") == "preserve this"


def test_missing_rgb_preserves_frozen_identities_without_replacement_or_complete_manifest(tmp_path):
    path = write_population(tmp_path, population())
    output = tmp_path / "missing_rgb"
    with pytest.raises(FileNotFoundError):
        panel.freeze_panel(path, tmp_path, output, expected_sha=panel.sha256_file(path))
    selected = json.loads((output / "panel.json").read_text(encoding="utf-8"))
    assert selected["clips"] == panel.select_panel(population())
    assert not (output / "rgb_manifest.json").exists()
    assert list((output / "rgb").iterdir()) == []


def test_bad_rgb_shape_does_not_create_complete_manifest(tmp_path, monkeypatch):
    path = write_population(tmp_path, population())
    output = tmp_path / "bad_rgb"
    monkeypatch.setattr(panel, "load_aligned_clip", lambda *args, **kwargs: (torch.zeros(16, 3, 10, 10), None))
    with pytest.raises(ValueError, match="uint8 RGB"):
        panel.freeze_panel(path, tmp_path, output, expected_sha=panel.sha256_file(path))
    assert (output / "panel.json").exists()
    assert not (output / "rgb_manifest.json").exists()
