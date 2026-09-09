"""CPU-only policy bridge contracts; no pretrained model or GPU evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autogaze.human_gaze import dinov3_export as bridge
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor


def run_config():
    policies = {"pretrained_k16": {"checkpoint_dir": None,
                                   "files": dict(bridge.PRETRAINED_FILES)}}
    for seed, digest in bridge.HUMAN_MODEL_SHA256.items():
        policies[f"human_k16_seed{seed}"] = {"checkpoint_dir": None, "files": dict(zip(
            bridge.REQUIRED_FILES, (digest, bridge.HUMAN_CONFIG_SHA256, bridge.HUMAN_PROCESSOR_SHA256),
        ))}
    return {"schema_version": 1, "population_manifest": "/explicit/clips.jsonl",
            "population_sha256": bridge.POPULATION_MANIFEST_SHA256,
            "dataset_root": "/explicit/data", "policies": policies}


def test_frozen_policy_names_and_human_hashes_are_required_but_freeze_can_precede_path_resolution():
    config = run_config()
    assert bridge.validate_run_config(config, check_files=False) == {}
    with pytest.raises(ValueError, match="checkpoint_dir"):
        bridge.validate_run_config(config)
    changed = copy.deepcopy(config)
    changed["policies"].pop("human_k16_seed440831")
    with pytest.raises(ValueError, match="all six"):
        bridge.validate_run_config(changed, check_files=False)
    changed = copy.deepcopy(config)
    changed["policies"]["human_k16_seed440826"]["files"]["model.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="frozen 20k"):
        bridge.validate_run_config(changed, check_files=False)


def test_checkpoint_hashing_blocks_missing_changed_and_implicitly_loaded_files(tmp_path):
    expected = {}
    for name in bridge.REQUIRED_FILES:
        (tmp_path / name).write_bytes(name.encode())
        expected[name] = bridge.sha256_file(tmp_path / name)
    actual = bridge.verify_checkpoint_files(tmp_path, expected)
    assert actual["files"] == expected
    assert len(actual["file_set_sha256"]) == 64
    with pytest.raises(ValueError, match="Missing or invalid"):
        bridge.verify_checkpoint_files(tmp_path, {**expected, "model.safetensors": None})
    (tmp_path / "generation_config.json").write_text("{}")
    with pytest.raises(ValueError, match="must also"):
        bridge.verify_checkpoint_files(tmp_path, expected)
    expected["generation_config.json"] = bridge.sha256_file(tmp_path / "generation_config.json")
    bridge.verify_checkpoint_files(tmp_path, expected)
    (tmp_path / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="mismatch"):
        bridge.verify_checkpoint_files(tmp_path, expected)


def test_actual_processor_uses_byte_values_without_crop_resize_or_batch_coupling():
    # Deliberately configure an incompatible resize/crop. The cached bridge must
    # disable both while retaining the checkpoint's own rescaling/normalization.
    processor = AutoGazeImageProcessor(do_resize=True, size={"shortest_edge": 256},
        do_center_crop=True, crop_size={"height": 100, "width": 100},
        do_rescale=True, rescale_factor=1 / 255, offset=False,
        do_normalize=True, image_mean=[0.1, 0.2, 0.3], image_std=[0.5, 0.25, 0.125])
    image = np.zeros((16, 3, 224, 224), dtype=np.uint8)
    image[:, 0] = np.arange(224, dtype=np.uint8)[None, :]
    image[:, 1] = np.arange(224, dtype=np.uint8)[:, None]
    image[:, 2, 0, 0] = 255
    other = np.full_like(image, 177)
    together = bridge.normalize_cached_rgb(processor, [image, other])
    alone = bridge.normalize_cached_rgb(processor, [image])
    expected = (torch.from_numpy(image).float() / 255
                - torch.tensor([0.1, 0.2, 0.3])[None, :, None, None]) / torch.tensor([0.5, 0.25, 0.125])[None, :, None, None]
    torch.testing.assert_close(together[0], expected)
    torch.testing.assert_close(together[:1], alone, rtol=0, atol=0)
    assert together.dtype == torch.float32 and together.is_contiguous()
    assert processor.do_resize and processor.do_center_crop  # override is per call
    assert bridge.processor_contract(processor)["offset"] is False
    with pytest.raises(ValueError, match="uint8"):
        bridge.normalize_cached_rgb(processor, [image.astype(np.float32) / 255])


def native_gaze(batch_size=2):
    cells = torch.stack([(torch.arange(16) * 11 + batch) % 196 for batch in range(batch_size)])
    cells = cells[:, None, :].expand(-1, 16, -1).clone()
    positions = (cells + 69 + torch.arange(16)[None, :, None] * 265).flatten(1)
    padded = torch.zeros_like(positions, dtype=torch.bool)
    counts = torch.full((16,), 16, dtype=torch.long)
    # Exercise the repository's real mask constructor without constructing or
    # loading the neural model. It returns float32 native-scale masks.
    geometry = SimpleNamespace(frame_sampling_rate=1, num_vision_tokens_each_frame=265,
        num_vision_tokens_each_scale_each_frame=[4, 16, 49, 196], scales=[32, 64, 112, 224])
    masks = AutoGaze.get_mask_from_gazing_pos(geometry, torch.empty(batch_size, 16, 3, 1, 1),
                                            positions, padded, counts)
    return {"scales": geometry.scales, "frame_sampling_rate": 1, "num_vision_tokens_each_frame": 265,
            "gazing_pos": positions, "if_padded_gazing": padded, "num_gazing_each_frame": counts,
            "gazing_mask": masks}, cells


def test_real_native_mask_constructor_agrees_with_global_cells_for_every_batch_and_frame():
    gaze, cells = native_gaze()
    assert gaze["gazing_mask"][-1].dtype == torch.float32
    torch.testing.assert_close(bridge.gaze_outputs_to_cells(gaze, 2), cells, rtol=0, atol=0)
    assert not torch.equal(cells[0], cells[1])


@pytest.mark.parametrize("defect", ["stride", "scales", "float_positions", "eos", "wrong_count",
                                         "duplicate", "wrong_frame", "mask_disagreement", "coarse", "nonbinary"])
def test_rejects_wrong_geometry_eos_repetition_or_misaligned_masks(defect):
    gaze, _ = native_gaze()
    if defect == "stride":
        gaze["frame_sampling_rate"] = 2
    elif defect == "scales":
        gaze["scales"][-1] = 256
    elif defect == "float_positions":
        gaze["gazing_pos"] = gaze["gazing_pos"].float()
    elif defect == "eos":
        gaze["if_padded_gazing"][0, 0] = True
    elif defect == "wrong_count":
        gaze["num_gazing_each_frame"][0] = 15
    elif defect == "duplicate":
        gaze["gazing_pos"][0, 1] = gaze["gazing_pos"][0, 0]
    elif defect == "wrong_frame":
        gaze["gazing_pos"][0, 16] -= 265
    elif defect == "mask_disagreement":
        gaze["gazing_mask"][-1][0, 0, 0] = 0
    elif defect == "coarse":
        gaze["gazing_mask"][0][0, 0, 0] = 1
    elif defect == "nonbinary":
        gaze["gazing_mask"][-1][0, 0, 0] = 0.5
    with pytest.raises(ValueError):
        bridge.gaze_outputs_to_cells(gaze, 2)


def test_rgb_cache_hash_geometry_and_path_are_checked(tmp_path):
    array = np.zeros((16, 3, 224, 224), dtype=np.uint8)
    path = tmp_path / "rgb.npy"
    np.save(path, array, allow_pickle=False)
    clip = {"rgb_path": "rgb.npy", "rgb_sha256": bridge.sha256_file(path)}
    np.testing.assert_array_equal(bridge.load_cached_rgb(tmp_path, clip), array)
    with pytest.raises(ValueError, match="inside"):
        bridge.load_cached_rgb(tmp_path, {**clip, "rgb_path": "../rgb.npy"})
    array[0, 0, 0, 0] = 255
    np.save(path, array, allow_pickle=False)
    with pytest.raises(ValueError, match="SHA-256"):
        bridge.load_cached_rgb(tmp_path, clip)


def test_final_schema_preserves_frame_mask_pairing_and_requires_all_policies():
    _, fine = native_gaze()
    clips = [{"clip_id": f"c{index}", "video_id": f"AVAD/v{index}", "source": "AVAD",
              "frame_numbers": list(range(1, 17)), "rgb_path": f"rgb/{index}.npy"} for index in range(2)]
    frozen = {"provenance": {"before_inference": True}, "panel_payload_sha256": "f" * 64, "clips": clips}
    outputs = {name: [{"clip_id": clip["clip_id"], "video_id": clip["video_id"],
                       "fine_cells": fine[index].tolist()} for index, clip in enumerate(clips)] for name in bridge.POLICY_NAMES}
    result = bridge.assemble_manifest(frozen, outputs, {"completed": True})
    assert result["schema_version"] == 1 and result["provenance"]["before_inference"]
    assert list(result["clips"][0]["policies"]) == list(bridge.POLICY_NAMES)
    assert result["clips"][1]["policies"]["pretrained_k16"] == fine[1].tolist()
    json.dumps(result, allow_nan=False)
    changed = copy.deepcopy(outputs)
    changed["pretrained_k16"].reverse()
    with pytest.raises(ValueError, match="identity"):
        bridge.assemble_manifest(frozen, changed, {})
    changed = copy.deepcopy(outputs)
    changed["pretrained_k16"][0]["fine_cells"][0][1] = changed["pretrained_k16"][0]["fine_cells"][0][0]
    with pytest.raises(ValueError, match="unique"):
        bridge.assemble_manifest(frozen, changed, {})
    with pytest.raises(ValueError, match="all seven"):
        bridge.assemble_manifest(frozen, {"pretrained_k16": outputs["pretrained_k16"]}, {})


def test_code_provenance_uses_the_exporter_repository_root(monkeypatch):
    commands = []
    def fake_git(command, text):
        commands.append(command)
        return "abc123" if "rev-parse" in command else ""
    monkeypatch.setattr(bridge.subprocess, "check_output", fake_git)
    metadata = bridge._code_provenance()
    assert commands[0][2] == str(bridge.Path(bridge.__file__).resolve().parents[2])
    assert metadata["autogaze_commit"] == "abc123"
