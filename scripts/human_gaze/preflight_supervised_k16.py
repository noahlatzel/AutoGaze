# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded real-model smoke test for supervised fine-only K16 training.

This is a preflight, not a training or evaluation run. It checks one stable
training clip per STAViS source, records full input identities, and refuses to
continue when the declared GPU-allocation or process-RSS ceilings are crossed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset, STAVIS_SOURCES
from autogaze.datasets.collate import collate_fn
from autogaze.human_gaze.coverage import global_positions_to_fine_cells
from autogaze.human_gaze.dinov3_export import (
    PRETRAINED_FILES,
    canonical_payload_sha256,
    sha256_file,
    verify_checkpoint_files,
    verify_checkpoint_loading,
)
from autogaze.models.autogaze import AutoGaze, AutoGazeConfig, AutoGazeImageProcessor


GIB = 1024**3
EXPECTED_SOURCE_SHA = "d85c6558bf9e8f02ece1f3516b4709a715ebec63"
EXPECTED_MODEL_SHA256 = "a48e6a83a198368e3798420ff5d5df42af7c0003c9230f85581c39a2ea64e9eb"
EXPECTED_MANIFEST_SHA256 = "0ef9fa17881517ea69d9edbaf57f8571c5c58d1da517370746fbe264fe33ff10"
EXPECTED_CELL_MASS_SHA256 = "fed5a6552ab94eeaa02a807959d4c3cda6de9dbadeadc35ee248e46ebc16dfb8"
FINE_OFFSET = 69
ACTIONS_PER_FRAME = 265
CLIP_LEN = 16
EXACT_K = 16


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def source_identity() -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("The immutable preflight source checkout is dirty")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_SOURCE_SHA, head],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"Preflight source {head} is not based on {EXPECTED_SOURCE_SHA}")
    changes = git_output("diff", "--name-only", f"{EXPECTED_SOURCE_SHA}..{head}").splitlines()
    allowed = {
        "scripts/human_gaze/preflight_supervised_k16.py",
        "tests/test_supervised_k16_real_preflight.py",
    }
    if not set(changes).issubset(allowed):
        raise ValueError(f"Child branch changes more than the preflight harness: {changes}")
    return {
        "commit": head,
        "implementation_base_commit": EXPECTED_SOURCE_SHA,
        "changes_from_implementation_base": changes,
        "dirty": False,
        "repository_root": str(REPOSITORY_ROOT),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "hydra-core", "omegaconf", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def max_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def gpu_snapshot() -> dict[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    return {"gpus": query, "compute_processes": processes}


def validate_comparison_config(path: Path) -> tuple[dict[str, Any], str]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Comparison config must resolve to a mapping")
    expected = {
        "budget": EXACT_K,
        "initialization": "shared_original_pretrained_autogaze",
        "initialization_model_sha256": EXPECTED_MODEL_SHA256,
        "trainable": "autoregressive_decoder_only",
        "objective": "remaining_human_mass_soft_cross_entropy",
        "teacher_history": "human_mass_sampled_without_replacement",
        "validation_decoding": "greedy_self_history_without_teacher_loss",
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Comparison config mismatch for {key}: {cfg.get(key)!r}")
    data = cfg.get("dataset", {})
    if data.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Comparison config has the wrong manifest hash")
    if data.get("cell_mass_sha256") != EXPECTED_CELL_MASS_SHA256:
        raise ValueError("Comparison config has the wrong cell-mass hash")
    stages = cfg.get("stages", [])
    if not stages or stages[0].get("config") != "av_gaze_stavis_supervised_k16_stage1":
        raise ValueError("Comparison config does not select the supervised K16 stage-one config")
    return cfg, sha256_file(path)


def load_stage_config(dataset_root: Path, manifest: Path, cell_mass: Path):
    config_dir = REPOSITORY_ROOT / "autogaze" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="av_gaze_stavis_supervised_k16_stage1")
    cfg.dataset.root = str(dataset_root)
    cfg.dataset.manifest_path = str(manifest)
    cfg.dataset.cell_mass_path = str(cell_mass)
    if (
        int(cfg.dataset.clip_len) != CLIP_LEN
        or int(cfg.dataset.image_size) != 224
        or int(cfg.dataset.grid_size) != 14
        or bool(cfg.dataset.load_heatmap)
        or not bool(cfg.dataset.load_rgb)
        or int(cfg.algorithm.exact_budget) != EXACT_K
        or int(cfg.algorithm.actions_per_frame) != ACTIONS_PER_FRAME
        or int(cfg.algorithm.fine_action_offset) != FINE_OFFSET
        or int(cfg.task.exact_budget) != EXACT_K
        or int(cfg.trainer.per_gpu_max_batch_size) != 1
        or not bool(cfg.trainer.freeze_gaze_vision)
        or not bool(cfg.trainer.freeze_gaze_connector)
    ):
        raise ValueError("Resolved stage-one configuration violates the preflight contract")
    return cfg


def first_stable_index_per_source(records: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []
    for source in STAVIS_SOURCES:
        candidates = [
            (str(record["clip_id"]), str(record["video_id"]), index)
            for index, record in enumerate(records)
            if record["source"] == source
        ]
        if not candidates:
            raise ValueError(f"TRAIN split has no clip for source {source}")
        selected.append(min(candidates)[2])
    return selected


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def move_tensors(inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def verify_exact_k(info: Mapping[str, Any]) -> torch.Tensor:
    counts = info.get("num_gazing_each_frame")
    if not isinstance(counts, torch.Tensor) or tuple(counts.shape) != (CLIP_LEN,):
        raise ValueError("Exact-K output has an invalid per-frame count vector")
    if not torch.equal(counts, torch.full_like(counts, EXACT_K)):
        raise ValueError("Exact-K output does not contain sixteen actions per frame")
    padded = info.get("if_padded_gazing")
    if not isinstance(padded, torch.Tensor) or padded.any():
        raise ValueError("Exact-K output contains padding or EOS")
    return global_positions_to_fine_cells(
        info["gazing_pos"],
        num_frames=CLIP_LEN,
        exact_budget=EXACT_K,
        actions_per_frame=ACTIONS_PER_FRAME,
        fine_action_offset=FINE_OFFSET,
    )


def parameter_contract(model: torch.nn.Module) -> dict[str, Any]:
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    frozen_names = [name for name, value in model.named_parameters() if not value.requires_grad]
    if not trainable_names or not frozen_names:
        raise ValueError("Expected both trainable and frozen model parameters")
    invalid_trainable = [
        name
        for name in trainable_names
        if name.startswith("gazing_model.vision_model.")
        or name.startswith("gazing_model.connector.")
    ]
    invalid_frozen = [
        name
        for name in frozen_names
        if not (
            name.startswith("gazing_model.vision_model.")
            or name.startswith("gazing_model.connector.")
        )
    ]
    if invalid_trainable or invalid_frozen:
        raise ValueError(
            f"Decoder-only boundary mismatch: trainable={invalid_trainable}, frozen={invalid_frozen}"
        )
    return {
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "trainable_tensors": len(trainable_names),
        "frozen_tensors": len(frozen_names),
        "trainable_numel": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "frozen_numel": sum(value.numel() for value in model.parameters() if not value.requires_grad),
        "trainable_names_sha256": canonical_payload_sha256(trainable_names),
        "frozen_names_sha256": canonical_payload_sha256(frozen_names),
    }


def gradient_contract(model: torch.nn.Module) -> dict[str, Any]:
    frozen_with_grad = []
    trainable_without_grad = []
    nonfinite = []
    zero_gradient = []
    squared_norm = 0.0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if not parameter.requires_grad:
            if gradient is not None:
                frozen_with_grad.append(name)
            continue
        if gradient is None:
            trainable_without_grad.append(name)
            continue
        values = gradient.detach().float()
        if not torch.isfinite(values).all():
            nonfinite.append(name)
            continue
        norm = float(values.norm().cpu())
        squared_norm += norm * norm
        if norm == 0:
            zero_gradient.append(name)
    if frozen_with_grad or trainable_without_grad or nonfinite or squared_norm <= 0:
        raise ValueError(
            "Gradient contract failed: "
            f"frozen_with_grad={frozen_with_grad}, trainable_without_grad={trainable_without_grad}, "
            f"nonfinite={nonfinite}, squared_norm={squared_norm}"
        )
    return {
        "global_l2_norm": squared_norm**0.5,
        "frozen_with_grad": frozen_with_grad,
        "trainable_without_grad": trainable_without_grad,
        "nonfinite_gradient": nonfinite,
        "zero_gradient_names": zero_gradient,
    }


def supervised_distribution_contract(algorithm, inputs, gaze_outputs) -> dict[str, Any]:
    targets, valid, available = algorithm.conditional_targets(inputs)
    log_probs = gaze_outputs["supervised_action_log_probs_all"]
    available = available.to(log_probs.device)
    row_sums = log_probs.detach().double().exp().sum(dim=-1)
    support_ok = bool(
        torch.isfinite(log_probs[available]).all()
        and torch.isneginf(log_probs[~available]).all()
    )
    normalized = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=3e-5, rtol=0))
    if not support_ok or not normalized:
        raise ValueError("Supervised fine-cell distribution support or normalization failed")
    return {
        "shape": list(log_probs.shape),
        "finite_legal_and_negative_infinity_elsewhere": support_ok,
        "normalized": normalized,
        "max_absolute_row_sum_error": float((row_sums - 1).abs().max().cpu()),
        "valid_target_rows": int(valid.sum().cpu()),
        "exhausted_target_rows": int((~valid).sum().cpu()),
        "targets_sha256": tensor_sha256(targets),
    }


def make_model(cfg, checkpoint: Path) -> tuple[AutoGaze, dict[str, Any]]:
    cfg.model.max_num_frames = cfg.dataset.clip_len
    runtime = AutoGaze(AutoGazeConfig(**OmegaConf.to_container(cfg.model, resolve=True)))
    pretrained, loading = AutoGaze.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    compatibility = verify_checkpoint_loading(pretrained, loading)
    missing, unexpected = runtime.load_state_dict(pretrained.state_dict(), strict=True)
    if missing or unexpected:
        raise ValueError(f"Runtime model copy mismatch: missing={missing}, unexpected={unexpected}")
    del pretrained
    return runtime, {"loading_info": loading, "compatibility": compatibility}


def build_dataset(cfg, processor) -> AVGazeStavisDataset:
    return instantiate(
        cfg.dataset,
        split="train",
        gaze_transform=processor,
        task_transform=None,
    )


def enforce_resources(
    device: torch.device, max_gpu_bytes: int, rss_ceiling_bytes: int
) -> dict[str, int]:
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    rss = max_rss_bytes()
    if allocated >= max_gpu_bytes:
        raise MemoryError(f"Peak GPU allocation {allocated} reached ceiling {max_gpu_bytes}")
    if rss >= rss_ceiling_bytes:
        raise MemoryError(f"Peak process RSS {rss} reached ceiling {rss_ceiling_bytes}")
    return {"peak_gpu_allocated_bytes": allocated, "peak_gpu_reserved_bytes": reserved, "peak_rss_bytes": rss}


def execute(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    started_monotonic = time.monotonic()
    source = source_identity()
    comparison, comparison_sha = validate_comparison_config(args.config)
    input_hashes = {
        "comparison_config": comparison_sha,
        "manifest": sha256_file(args.manifest),
        "cell_mass": sha256_file(args.cell_mass),
    }
    if input_hashes["manifest"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Live TRAIN manifest hash mismatch")
    if input_hashes["cell_mass"] != EXPECTED_CELL_MASS_SHA256:
        raise ValueError("Live cell-mass hash mismatch")
    checkpoint = verify_checkpoint_files(args.pretrained, PRETRAINED_FILES)
    if checkpoint["files"]["model.safetensors"] != EXPECTED_MODEL_SHA256:
        raise ValueError("Live original pretrained model hash mismatch")

    cfg = load_stage_config(args.dataset_root, args.manifest, args.cell_mass)
    resolved_stage = OmegaConf.to_container(cfg, resolve=True)
    processor_cfg = OmegaConf.to_container(cfg.model.preprocessing, resolve=True)
    processor_cfg["size"] = {"shortest_edge": 224}
    processor = AutoGazeImageProcessor(**processor_cfg)
    dataset = build_dataset(cfg, processor)
    selected_indices = first_stable_index_per_source(dataset.records)
    selected_records = [dataset.records[index] for index in selected_indices]
    if len(selected_records) != args.max_clips:
        raise ValueError(
            f"This preflight requires one clip for each of six sources; got max-clips={args.max_clips}"
        )

    before_gpu = gpu_snapshot()
    if before_gpu["compute_processes"]:
        raise RuntimeError(f"GPU already has compute processes: {before_gpu['compute_processes']}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    total_gpu_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    max_gpu_bytes = int(args.max_gpu_alloc_gib * GIB)
    if max_gpu_bytes >= total_gpu_bytes:
        raise ValueError("GPU allocation ceiling must be smaller than physical GPU memory")
    torch.cuda.set_per_process_memory_fraction(max_gpu_bytes / total_gpu_bytes, device)
    torch.cuda.reset_peak_memory_stats(device)

    model, load_record = make_model(cfg, args.pretrained)
    for parameter in model.gazing_model.vision_model.parameters():
        parameter.requires_grad = False
    for parameter in model.gazing_model.connector.parameters():
        parameter.requires_grad = False
    parameters = parameter_contract(model)
    algorithm = instantiate(cfg.algorithm)
    task = instantiate(cfg.task).to(device)
    if any(parameter.requires_grad for parameter in task.parameters()):
        raise ValueError("The supervised coverage task must remain parameter-free")

    init_file = run_dir / "single_rank_ddp_init"
    torch.distributed.init_process_group(
        backend="nccl", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    model = model.to(device).train()
    ddp = DDP(model, find_unused_parameters=False, device_ids=[device.index], output_device=device.index)
    progress_path = run_dir / "clips.jsonl"
    clip_results: list[dict[str, Any]] = []
    try:
        with progress_path.open("x", encoding="utf-8") as progress:
            for iteration, (dataset_index, record) in enumerate(
                zip(selected_indices, selected_records), start=1
            ):
                if time.monotonic() - started_monotonic >= args.deadline_seconds:
                    raise TimeoutError("Preflight reached its declared wall-time ceiling")
                iteration_started = time.monotonic()
                batch = collate_fn([dataset[dataset_index]])
                video_shape = list(batch["video"].shape)
                video_dtype = str(batch["video"].dtype)
                inputs = move_tensors(batch, device)
                supervised_inputs = algorithm.preprocess_inputs(inputs)
                teacher_fine = verify_exact_k(supervised_inputs["gt_gazing_info"])
                ddp.zero_grad(set_to_none=True)
                gaze_outputs = ddp(
                    supervised_inputs,
                    gazing_info=supervised_inputs["gt_gazing_info"],
                    return_supervised_log_probs=True,
                    **task.gaze_model_kwargs,
                )
                scored_fine = verify_exact_k(gaze_outputs)
                if not torch.equal(teacher_fine, scored_fine):
                    raise ValueError("Real model did not rescore the supplied teacher trajectory")
                distribution = supervised_distribution_contract(
                    algorithm, supervised_inputs, gaze_outputs
                )
                algorithm_outputs = algorithm(supervised_inputs, gaze_outputs, {})
                loss = algorithm_outputs["loss"].mean()
                if not torch.isfinite(loss):
                    raise ValueError("Supervised real-model loss is non-finite")
                loss.backward()
                gradients = gradient_contract(ddp.module)
                torch.cuda.synchronize(device)
                resource_use = enforce_resources(
                    device, max_gpu_bytes, int(args.max_rss_gib * GIB)
                )
                result = {
                    "iteration": iteration,
                    "dataset_index": dataset_index,
                    "source": record["source"],
                    "video_id": record["video_id"],
                    "clip_id": record["clip_id"],
                    "frame_numbers": record["frame_numbers"],
                    "video_shape": video_shape,
                    "video_dtype": video_dtype,
                    "teacher_fine_cells_sha256": tensor_sha256(teacher_fine),
                    "teacher_first_frame_cells": teacher_fine[0, 0].detach().cpu().tolist(),
                    "loss": float(loss.detach().cpu()),
                    "metrics": {
                        key: float(value.detach().cpu())
                        for key, value in algorithm_outputs["metrics"].items()
                    },
                    "distribution": distribution,
                    "gradients": gradients,
                    "resource_use": resource_use,
                    "seconds": time.monotonic() - iteration_started,
                }
                progress.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                progress.flush()
                clip_results.append(result)
                del batch, inputs, supervised_inputs, teacher_fine, gaze_outputs, scored_fine
                del algorithm_outputs, loss

        # A second, independent pass deliberately supplies only pixels. The
        # annotation tensor and teacher history never enter greedy validation.
        greedy_results: list[dict[str, Any]] = []
        ddp.module.eval()
        with torch.inference_mode():
            for dataset_index, record in zip(selected_indices, selected_records):
                batch = collate_fn([dataset[dataset_index]])
                video = batch["video"].to(device)
                gaze_outputs = ddp.module(
                    {"video": video},
                    max_gaze_tokens_each_frame=EXACT_K,
                    allowed_token_ids=list(range(FINE_OFFSET, ACTIONS_PER_FRAME)),
                    allow_eos=False,
                    min_gaze_tokens_each_frame=EXACT_K,
                    generate_only=True,
                    use_cache=False,
                )
                greedy_fine = verify_exact_k(gaze_outputs)
                greedy_results.append(
                    {
                        "source": record["source"],
                        "clip_id": record["clip_id"],
                        "fine_cells_sha256": tensor_sha256(greedy_fine),
                        "first_frame_cells": greedy_fine[0, 0].detach().cpu().tolist(),
                        "label_free_inputs": ["video"],
                    }
                )
                del batch, video, gaze_outputs, greedy_fine
        torch.cuda.synchronize(device)
        resources = enforce_resources(device, max_gpu_bytes, int(args.max_rss_gib * GIB))
    finally:
        del ddp
        torch.distributed.destroy_process_group()

    elapsed = time.monotonic() - started_monotonic
    if elapsed >= args.deadline_seconds:
        raise TimeoutError("Preflight exceeded its declared wall-time ceiling")
    if len(clip_results) < 2:
        raise ValueError("At least two consecutive DDP iterations are required")
    return {
        "schema_version": 1,
        "kind": "non_scientific_real_model_supervised_k16_preflight",
        "started_utc": args.started_utc,
        "completed_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "source": source,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "comparison_config": str(args.config),
            "comparison_config_sha256": comparison_sha,
            "comparison_contract": comparison,
            "resolved_stage_config": resolved_stage,
            "dataset_root": str(args.dataset_root),
            "manifest": str(args.manifest),
            "cell_mass": str(args.cell_mass),
            "input_hashes": input_hashes,
            "checkpoint": checkpoint,
            "checkpoint_loading": load_record,
        },
        "selection": {
            "split": "train",
            "rule": "lexicographically smallest (clip_id, video_id, dataset_index) per canonical STAViS source",
            "source_order": list(STAVIS_SOURCES),
            "clip_count": len(selected_records),
            "records": selected_records,
        },
        "execution": {
            "single_rank_ddp": True,
            "ddp_backend": "nccl",
            "find_unused_parameters": False,
            "consecutive_forward_backward_iterations": len(clip_results),
            "microbatch_size": 1,
            "optimizer_updates": 0,
            "data_loader_workers": 0,
            "cpu_thread_limit": args.cpu_threads,
            "deadline_seconds": args.deadline_seconds,
            "gpu_allocation_ceiling_bytes": max_gpu_bytes,
            "rss_ceiling_bytes": int(args.max_rss_gib * GIB),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": total_gpu_bytes,
            "peak_resources": resources,
            "occupancy_before_cuda": before_gpu,
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "packages": package_versions(),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        },
        "parameter_contract": parameters,
        "supervised_iterations": clip_results,
        "label_free_greedy": greedy_results,
        "acceptance": {
            "all_six_sources": len(clip_results) == len(STAVIS_SOURCES),
            "two_or_more_consecutive_single_rank_ddp_backward_iterations": len(clip_results) >= 2,
            "all_losses_finite": all(torch.isfinite(torch.tensor(row["loss"])) for row in clip_results),
            "decoder_only_gradients": True,
            "fine_only_exact_k16_teacher_and_greedy": True,
            "label_free_greedy_input_keys": ["video"],
            "resource_ceilings_respected": True,
            "scientific_result": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-mass", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-clips", type=int, default=6, choices=(6,))
    parser.add_argument("--cpu-threads", type=int, default=2, choices=(1, 2))
    parser.add_argument("--max-gpu-alloc-gib", type=float, default=10.0)
    parser.add_argument("--max-rss-gib", type=float, default=8.0)
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    args = parser.parse_args()
    args.started_utc = utc_now()
    for name in ("config", "pretrained", "dataset_root", "manifest", "cell_mass"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.output = args.output.expanduser().resolve()
    if args.max_gpu_alloc_gib <= 0 or args.max_gpu_alloc_gib > 10:
        parser.error("--max-gpu-alloc-gib must lie in (0, 10]")
    if args.max_rss_gib <= 0 or args.max_rss_gib > 8:
        parser.error("--max-rss-gib must lie in (0, 8]")
    if args.deadline_seconds <= 0 or args.deadline_seconds > 1800:
        parser.error("--deadline-seconds must lie in (0, 1800]")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new_json(
        args.output / "run.json",
        {
            "schema_version": 1,
            "kind": "non_scientific_real_model_supervised_k16_preflight",
            "started_utc": args.started_utc,
            "status": "started",
            "command": [sys.executable, *sys.argv],
        },
    )
    try:
        receipt = execute(args, args.output)
        write_new_json(args.output / "complete.json", receipt)
        print(json.dumps({
            "status": "complete",
            "output": str(args.output),
            "receipt_sha256": sha256_file(args.output / "complete.json"),
            "elapsed_seconds": receipt["elapsed_seconds"],
            "peak_resources": receipt["execution"]["peak_resources"],
        }, sort_keys=True))
    except Exception as error:
        write_new_json(
            args.output / "failure.json",
            {
                "schema_version": 1,
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "max_rss_bytes": max_rss_bytes(),
            },
        )
        raise


if __name__ == "__main__":
    main()
