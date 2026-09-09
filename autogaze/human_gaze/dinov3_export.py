# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the frozen seven-policy WP2 fine-cell panel without loading a VLM."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from autogaze.human_gaze.coverage import global_positions_to_fine_cells
from autogaze.human_gaze.dinov3_panel import (
    PANEL_SALT, POPULATION_MANIFEST_SHA256, canonical_payload_sha256, sha256_file,
)


HUMAN_MODEL_SHA256 = {
    440826: "d008a458e62a15c69a236e4585a248b185343e82c2b6ece205d6638d4578b9ff",
    440827: "669777ee6191c2a01ba1069f3492cb8e75aeaae307bdc5bf895ff13145c4f22a",
    440828: "83cdc62cda40d75d9dedf34134484715f38a984542d1e27e1dd78c9bcfa69596",
    440829: "fe287cc2aaaf84153f277da87103b37c3c7949ce3f66c9ebbedfeeeebc37658d",
    440830: "ec287894d2f77b8fcd6e6fa102460c7b4533a671942dd713506ca8abc6788180",
    440831: "1721a2175adc3fbf95e2ad293736e80345ac4c89f1d2e3c514f4acd5261a7781",
}
HUMAN_CONFIG_SHA256 = "2fbb670402ba0ac378b97a7ef82f4f69740fa94b4523cc34ca6b6bcd2a2f9887"
HUMAN_PROCESSOR_SHA256 = "a28d1b6ea575bc9877782f5c467428ab986dfb2bba78c9e62f3a7652130cfca3"
PRETRAINED_FILES = {
    "model.safetensors": "a48e6a83a198368e3798420ff5d5df42af7c0003c9230f85581c39a2ea64e9eb",
    "config.json": "2bfa48a1f8ac23e67e3477edfd5c439305bdf5fb22dd0130965863bc3e29236b",
    "preprocessor_config.json": HUMAN_PROCESSOR_SHA256,
}
POLICY_NAMES = ("pretrained_k16", *(f"human_k16_seed{seed}" for seed in HUMAN_MODEL_SHA256))
REQUIRED_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
HISTORICAL_ZERO_BIAS = "gazing_model.gaze_decoder.output_token_logit_bias"
GENERATION_CONTRACT = {
    "max_gaze_tokens_each_frame": 16, "allowed_token_ids": list(range(69, 265)),
    "allow_eos": False, "min_gaze_tokens_each_frame": 16, "gazing_ratio": None,
    "task_loss_requirement": None, "generate_only": True, "use_cache": False,
}


def write_new_json(path: Path, payload: Mapping) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def verify_checkpoint_files(directory: Path, expected: Mapping) -> dict:
    """Hash all declared local files; missing SHA values are execution blockers."""
    if not isinstance(expected, Mapping) or not set(REQUIRED_FILES).issubset(expected):
        raise ValueError(f"Checkpoint must declare SHA-256 for {REQUIRED_FILES}.")
    for name, digest in expected.items():
        if name not in (*REQUIRED_FILES, "generation_config.json"):
            raise ValueError(f"Unexpected checkpoint filename: {name!r}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Missing or invalid SHA-256 for {name}; resolve it on Linux before execution.")
    directory = Path(directory).expanduser().resolve(strict=True)
    # HF can read this optional file automatically. It must also be pinned if present.
    if (directory / "generation_config.json").exists() and "generation_config.json" not in expected:
        raise ValueError("Present generation_config.json must also have an explicit SHA-256.")
    observed = {}
    for name, digest in expected.items():
        observed[name] = sha256_file(directory / name)
        if observed[name] != digest:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {directory / name}")
    return {"checkpoint_dir": str(directory), "files": observed,
            "file_set_sha256": canonical_payload_sha256(observed)}


def verify_checkpoint_loading(model: torch.nn.Module, loading: Mapping) -> dict:
    """Allow only the historical absent logit-bias buffer, verified exactly zero.

    The current decoder registers this persistent buffer, while the frozen
    historical weights predate it. This check never initializes or changes it.
    """
    missing = loading.get("missing_keys", [])
    if (missing not in ([], [HISTORICAL_ZERO_BIAS])
            or any(loading.get(key) for key in ("unexpected_keys", "mismatched_keys", "error_msgs"))):
        raise ValueError(f"Checkpoint did not load exactly apart from the known zero buffer: {loading}")
    try:
        bias = model.get_buffer(HISTORICAL_ZERO_BIAS)
        vocabulary_size = model.gazing_model.gaze_decoder.vocab_size
    except AttributeError as error:
        raise ValueError("Historical-checkpoint compatibility buffer is missing or invalid.") from error
    if (not isinstance(bias, torch.Tensor) or bias.device.type == "meta"
            or not bias.is_floating_point() or type(vocabulary_size) is not int or vocabulary_size < 1
            or tuple(bias.shape) != (vocabulary_size,)
            or HISTORICAL_ZERO_BIAS not in model.state_dict()):
        raise ValueError("Historical-checkpoint compatibility buffer is missing, nonpersistent or invalid.")
    if not torch.isfinite(bias).all() or torch.count_nonzero(bias).item() != 0:
        raise ValueError("Historical-checkpoint compatibility buffer must be finite and exactly zero.")
    return {"accepted_runtime_only_keys": list(missing), "buffer_name": HISTORICAL_ZERO_BIAS,
            "buffer_shape": list(bias.shape), "buffer_dtype": str(bias.dtype),
            "runtime_only_key_initialization": "registered exact-zero buffer; verified without mutation",
            "max_absolute_value": 0.0}


def validate_run_config(config: Mapping, *, check_files: bool = True) -> dict:
    """Require the population and all six frozen human seeds, with explicit paths."""
    if config.get("schema_version") != 1:
        raise ValueError("Expected run config schema_version=1.")
    if config.get("population_sha256") != POPULATION_MANIFEST_SHA256:
        raise ValueError("Run config must retain the frozen population SHA-256.")
    for key in ("population_manifest", "dataset_root"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"Supply an explicit {key} on Linux.")
    policies = config.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != set(POLICY_NAMES):
        raise ValueError("Exactly pretrained_k16 and all six frozen human_k16_seed policies are required.")
    checked = {}
    for name in POLICY_NAMES:
        spec = policies[name]
        if not isinstance(spec, Mapping) or not isinstance(spec.get("files"), Mapping):
            raise ValueError(f"{name}: missing checkpoint file hashes.")
        if name != "pretrained_k16":
            seed = int(name.removeprefix("human_k16_seed"))
            frozen = dict(zip(REQUIRED_FILES, (
                HUMAN_MODEL_SHA256[seed], HUMAN_CONFIG_SHA256, HUMAN_PROCESSOR_SHA256,
            )))
            if any(spec["files"].get(key) != value for key, value in frozen.items()):
                raise ValueError(f"{name}: supplied hashes differ from the frozen 20k policy.")
        elif any(spec["files"].get(key) != value for key, value in PRETRAINED_FILES.items()):
            raise ValueError("pretrained_k16: supplied hashes differ from the verified pretrained checkpoint.")
        if check_files:
            directory = spec.get("checkpoint_dir")
            if not isinstance(directory, str) or not directory:
                raise ValueError(f"{name}: supply its exact checkpoint_dir on Linux.")
            checked[name] = verify_checkpoint_files(Path(directory), spec["files"])
    return checked


def load_cached_rgb(panel_dir: Path, clip: Mapping) -> np.ndarray:
    relative = Path(clip["rgb_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RGB cache paths must stay inside the frozen panel directory.")
    path = (Path(panel_dir) / relative).resolve(strict=True)
    if not path.is_relative_to(Path(panel_dir).resolve()):
        raise ValueError("RGB cache path escapes the frozen panel directory.")
    if sha256_file(path) != clip["rgb_sha256"]:
        raise ValueError(f"Cached RGB SHA-256 mismatch: {path}")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.uint8 or array.shape != (16, 3, 224, 224) or not array.flags.c_contiguous:
        raise ValueError("Expected C-contiguous uint8 RGB [16,3,224,224].")
    return array


def read_frozen_panel(panel_dir: Path) -> dict:
    panel_dir = Path(panel_dir)
    frozen = json.loads((panel_dir / "rgb_manifest.json").read_text(encoding="utf-8"))
    provenance = frozen.get("provenance", {})
    if frozen.get("schema_version") != 1 or frozen.get("status") != "frozen":
        raise ValueError("Expected a completed frozen RGB panel.")
    if (provenance.get("population_manifest_sha256") != POPULATION_MANIFEST_SHA256
            or provenance.get("selection_salt") != PANEL_SALT):
        raise ValueError("Frozen RGB panel does not match the approved population and salt.")
    if sha256_file(panel_dir / "panel.json") != provenance.get("panel_json_sha256"):
        raise ValueError("Frozen selection manifest hash changed.")
    if canonical_payload_sha256({"clips": frozen["clips"]}) != frozen.get("panel_payload_sha256"):
        raise ValueError("Frozen panel identity/RGB payload hash changed.")
    from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
    clips = frozen["clips"]
    if len(clips) != 12 or len({clip["video_id"] for clip in clips}) != 12:
        raise ValueError("Expected exactly 12 distinct source-qualified videos.")
    if any(sum(clip["source"] == source for clip in clips) != 2 for source in STAVIS_SOURCES):
        raise ValueError("Expected two videos per each STAViS source.")
    for clip in clips:
        load_cached_rgb(panel_dir, clip)
    return frozen


def normalize_cached_rgb(processor, arrays: Sequence[np.ndarray]) -> torch.Tensor:
    """Use the checkpoint's normalization, with geometry already fixed by cache."""
    if not arrays:
        raise ValueError("Expected at least one RGB clip.")
    for array in arrays:
        if array.dtype != np.uint8 or array.shape != (16, 3, 224, 224):
            raise ValueError("Policy input must originate in cached uint8 [16,3,224,224] RGB.")
    videos = [[np.ascontiguousarray(frame.transpose(1, 2, 0)) for frame in array] for array in arrays]
    values = processor(
        videos, return_tensors="pt", do_resize=False, do_center_crop=False,
        input_data_format="channels_last", data_format="channels_first",
    ).pixel_values
    if tuple(values.shape) != (len(arrays), 16, 3, 224, 224) or not values.is_floating_point():
        raise ValueError("Processor changed the frozen frame/spatial geometry or returned non-float values.")
    if not torch.isfinite(values).all():
        raise ValueError("Processor returned non-finite normalized RGB.")
    return values.to(dtype=torch.float32).contiguous()


def processor_contract(processor) -> dict:
    return {"geometry": "cached_full_FOV_224x224; processor resize/crop explicitly disabled",
            **{key: getattr(processor, key) for key in (
                "do_rescale", "rescale_factor", "offset", "do_normalize", "image_mean", "image_std",
            )}}


def gaze_outputs_to_cells(gaze: Mapping, batch_size: int) -> torch.Tensor:
    """Check native fine geometry, counts, order and mask agreement before export."""
    if (gaze.get("scales") != [32, 64, 112, 224] or gaze.get("frame_sampling_rate") != 1
            or gaze.get("num_vision_tokens_each_frame") != 265):
        raise ValueError("Expected native [32,64,112,224] scales, 265 actions/frame, temporal stride 1.")
    positions, padded, counts = (gaze[key] for key in ("gazing_pos", "if_padded_gazing", "num_gazing_each_frame"))
    if positions.dtype not in (torch.int32, torch.int64) or tuple(positions.shape) != (batch_size, 256):
        raise ValueError("Expected integer global positions [B,16*16].")
    if padded.dtype != torch.bool or padded.shape != positions.shape or padded.any():
        raise ValueError("Fine-only generation returned padding/EOS or invalid padding metadata.")
    if counts.dtype not in (torch.int32, torch.int64) or counts.shape != (16,) or not torch.all(counts == 16):
        raise ValueError("Expected 16 unpadded actions for each of 16 frames.")
    cells = global_positions_to_fine_cells(positions, num_frames=16, exact_budget=16,
                                          actions_per_frame=265, fine_action_offset=69)
    masks = gaze["gazing_mask"]
    if not isinstance(masks, (list, tuple)) or len(masks) != 4:
        raise ValueError("Expected four native scale masks.")
    for mask, size in zip(masks, (4, 16, 49, 196)):
        # Native AutoGaze.get_mask_from_gazing_pos returns floating 0/1 masks.
        if tuple(mask.shape) != (batch_size, 16, size) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Unexpected native mask geometry/dtype.")
    if any(mask.any() for mask in masks[:3]):
        raise ValueError("Fine-only selection unexpectedly retained a coarse action.")
    expected = torch.zeros_like(masks[-1], dtype=torch.bool).scatter_(2, cells.long(), True)
    if not torch.equal(masks[-1].bool(), expected):
        raise ValueError("Ordered global actions and native fine mask disagree.")
    return cells.cpu()


def assemble_manifest(frozen: Mapping, outputs: Mapping, provenance: Mapping) -> dict:
    if set(outputs) != set(POLICY_NAMES):
        raise ValueError("Final export requires all seven policies.")
    clips = []
    for index, clip in enumerate(frozen["clips"]):
        policies = {}
        for name in POLICY_NAMES:
            if len(outputs[name]) != len(frozen["clips"]):
                raise ValueError(f"{name}: incomplete clip output.")
            record = outputs[name][index]
            if record["clip_id"] != clip["clip_id"] or record["video_id"] != clip["video_id"]:
                raise ValueError(f"{name}: policy output order/identity differs from frozen RGB.")
            cells = record["fine_cells"]
            if (not isinstance(cells, list) or len(cells) != 16
                    or any(not isinstance(row, list) or len(row) != 16
                           or any(type(value) is not int or not 0 <= value < 196 for value in row)
                           or len(set(row)) != 16 for row in cells)):
                raise ValueError(f"{name}: expected 16 unique fine cells for each of 16 frames.")
            policies[name] = cells
        clips.append({**clip, "policies": policies})
    return {"schema_version": 1,
            "provenance": {**frozen["provenance"], "panel_payload_sha256": frozen["panel_payload_sha256"], **provenance},
            "clips": clips}


def _code_provenance() -> dict:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError("Exporter checkout must be clean and committed before real inference.")
    packages = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "transformers", "timm", "numpy", "pillow",
    )}
    return {"autogaze_commit": commit, "autogaze_checkout": str(repo), "packages": packages}


def export_policies(config: Mapping, panel_dir: Path, *, device: str = "cuda", batch_size: int = 1,
                    cpu_threads: int = 2, num_workers: int = 0) -> Path:
    """Run one checkpoint at a time; publish manifest.json only after all succeed."""
    if not 1 <= batch_size <= 4 or not 1 <= cpu_threads <= 2 or num_workers != 0:
        raise ValueError("Admitted envelope requires batch 1..4, CPU threads 1..2, num_workers=0.")
    checked = validate_run_config(config)
    panel_dir = Path(panel_dir).resolve(strict=True)
    frozen = read_frozen_panel(panel_dir)
    if (panel_dir / "policies").exists() or (panel_dir / "manifest.json").exists():
        raise FileExistsError("Policy export already started; preserve this run and use a new panel directory.")
    code = _code_provenance()
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Real selector export requires the admitted Linux CUDA lane; CPU is for unit tests only.")
    if target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(target)
    # Limit allocator reservation to 9 GiB, leaving headroom below the 10 GiB
    # admission cap for the CUDA context and libraries. Record actual peaks.
    total_bytes = torch.cuda.get_device_properties(target).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 9 * 1024**3 / total_bytes), target)
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
    processors, normalization = {}, {}
    for name in POLICY_NAMES:
        processors[name] = AutoGazeImageProcessor.from_pretrained(checked[name]["checkpoint_dir"], local_files_only=True)
        normalization[name] = processor_contract(processors[name])
    if any(value != normalization["pretrained_k16"] for value in normalization.values()):
        raise ValueError("Policy normalization differs between checkpoints; resolve the input contract before export.")
    started = datetime.now(timezone.utc).isoformat()
    settings = {"batch_size": batch_size, "cpu_threads": cpu_threads, "num_workers": 0,
                "generation": GENERATION_CONTRACT, "eval_mode": True, "inference_mode": True,
                "parameter_dtype": "float32", "input_dtype": "float32", "normalization": normalization,
                "device": str(target), "gpu_name": torch.cuda.get_device_name(target),
                "gpu_total_memory_bytes": total_bytes, "cuda_allocator_cap_bytes": min(total_bytes, 9 * 1024**3),
                "cuda_version": torch.version.cuda, "timing_scope":
                "Single execution wall time per batch: normalized RGB already on device, model.generate plus mask construction, CUDA synchronized. No warmup/repetition; first batch is cold. Excludes load/normalization/copy. Not a latency benchmark."}
    (panel_dir / "policies").mkdir()
    write_new_json(panel_dir / "export_run.json", {"schema_version": 1, "started_utc": started,
                   "panel_payload_sha256": frozen["panel_payload_sha256"], "checkpoints": checked,
                   "code": code, "settings": settings, "config": config})
    all_outputs, policy_metadata = {}, {}
    for name in POLICY_NAMES:
        # Recheck immediately before loading to reject checkpoint replacement.
        verify_checkpoint_files(Path(checked[name]["checkpoint_dir"]), checked[name]["files"])
        model, loading = AutoGaze.from_pretrained(checked[name]["checkpoint_dir"], local_files_only=True,
            use_safetensors=True, torch_dtype=torch.float32, output_loading_info=True)
        compatibility = verify_checkpoint_loading(model, loading)
        model = model.to(target).eval()
        execution = {"attn_mode": model.attn_mode,
                     "generation_autocast": "bfloat16" if model.attn_mode == "flash_attention_2" else None,
                     "loading_info": loading, "checkpoint_compatibility": compatibility}
        records, batches = [], []
        torch.cuda.reset_peak_memory_stats(target)
        with (panel_dir / "policies" / f"{name}.progress.jsonl").open("x", encoding="utf-8") as progress:
            with torch.inference_mode():
                for start in range(0, len(frozen["clips"]), batch_size):
                    clips = frozen["clips"][start:start + batch_size]
                    arrays = [load_cached_rgb(panel_dir, clip) for clip in clips]
                    video = normalize_cached_rgb(processors[name], arrays).to(target)
                    torch.cuda.synchronize(target)
                    begin = time.perf_counter()
                    gaze = model({"video": video}, **GENERATION_CONTRACT)
                    torch.cuda.synchronize(target)
                    seconds = time.perf_counter() - begin
                    fine = gaze_outputs_to_cells(gaze, len(clips)).tolist()
                    for clip, cells in zip(clips, fine):
                        record = {"clip_id": clip["clip_id"], "video_id": clip["video_id"], "fine_cells": cells}
                        records.append(record)
                        progress.write(json.dumps(record, sort_keys=True) + "\n")
                    progress.flush()
                    batches.append({"start_clip_index": start, "clip_count": len(clips), "seconds": seconds,
                                    "cold_first_batch": start == 0})
                    del video, gaze, arrays
        policy_metadata[name] = {"checkpoint": checked[name], "batches": batches,
            "execution": execution,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(target),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(target)}
        write_new_json(panel_dir / "policies" / f"{name}.json", {
            "schema_version": 1, "panel_payload_sha256": frozen["panel_payload_sha256"],
            "policy": name, "metadata": policy_metadata[name], "clips": records,
        })
        all_outputs[name] = records
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"policy_completed": name, "clips": len(records)}), flush=True)
    manifest = assemble_manifest(frozen, all_outputs, {
        "export_started_utc": started, "export_completed_utc": datetime.now(timezone.utc).isoformat(),
        "code": code, "settings": settings, "checkpoints": checked, "policy_measurements": policy_metadata,
        "rgb_manifest_sha256": sha256_file(panel_dir / "rgb_manifest.json"),
        "export_run_sha256": sha256_file(panel_dir / "export_run.json"),
    })
    path = panel_dir / "manifest.json"
    write_new_json(path, manifest)
    write_new_json(panel_dir / "complete.json", {"schema_version": 1, "manifest_sha256": sha256_file(path),
        "panel_payload_sha256": frozen["panel_payload_sha256"], "clips": len(frozen["clips"]),
        "frame_policy_inputs": 12 * 16 * len(POLICY_NAMES)})
    return path
