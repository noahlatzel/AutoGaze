#!/usr/bin/env python3
"""Fail-closed source/input and checkpoint checks for supervised K16 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from autogaze.supervised_checkpoint import verify_supervised_completion


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_BASE = "d85c6558bf9e8f02ece1f3516b4709a715ebec63"
COMPARISON_SHA256 = "6730eb010584e50e8e11c20ed1c175553cbb14e6ec32e43e9b1ee7f81f301ada"
MANIFEST_SHA256 = "0ef9fa17881517ea69d9edbaf57f8571c5c58d1da517370746fbe264fe33ff10"
CELL_MASS_SHA256 = "fed5a6552ab94eeaa02a807959d4c3cda6de9dbadeadc35ee248e46ebc16dfb8"
PRETRAINED_SHA256 = "a48e6a83a198368e3798420ff5d5df42af7c0003c9230f85581c39a2ea64e9eb"
BASE_SEEDS = list(range(440826, 440832))
CONTINUATION_SEEDS = list(range(540826, 540832))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def verified_source(expected_commit: str) -> dict[str, Any]:
    head = git("rev-parse", "HEAD").stdout.strip()
    if head != expected_commit:
        raise ValueError(f"Execution source mismatch: expected {expected_commit}, found {head}")
    if git("status", "--porcelain=v1", "--untracked-files=all").stdout.strip():
        raise ValueError("Execution source checkout is dirty")
    if git("symbolic-ref", "-q", "HEAD", check=False).returncode == 0:
        raise ValueError("Execution source must be detached at the admitted commit")
    if git("merge-base", "--is-ancestor", IMPLEMENTATION_BASE, head, check=False).returncode:
        raise ValueError("Execution source is not descended from the frozen implementation")
    changes = [
        value for value in git("diff", "--name-only", f"{IMPLEMENTATION_BASE}..{head}").stdout.splitlines()
        if value
    ]
    permitted_exact = {
        "experiments/human_gaze/SUPERVISED_K16.md",
        "scripts/human_gaze/preflight_supervised_k16.py",
        "scripts/human_gaze/verify_supervised_k16_execution.py",
        "tests/test_supervised_k16_real_preflight.py",
        "tests/test_supervised_k16_execution.py",
        "experiments/human_gaze/configs/supervised_k16_execution.yaml",
        "experiments/human_gaze/slurm/run_supervised_k16_comparison_array.sbatch",
        "results/wp5_three_strand_evidence/READINESS.md",
        "results/wp5_three_strand_evidence/owner_status.csv",
        "results/wp5_three_strand_evidence/readiness.json",
    }
    permitted_prefix = "experiments/human_gaze/results/supervised_k16_real_preflight/"
    unexpected = [
        path for path in changes
        if path not in permitted_exact and not path.startswith(permitted_prefix)
    ]
    if unexpected:
        raise ValueError(f"Execution branch changes scientific source files: {unexpected}")
    return {
        "repository": git("remote", "get-url", "origin").stdout.strip(),
        "execution_commit": head,
        "implementation_base_commit": IMPLEMENTATION_BASE,
        "dirty": False,
        "detached": True,
        "changes_from_implementation_base": changes,
    }


def load_execution_config(path: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Execution config must resolve to a mapping")
    expected = {
        "schema_version": 1,
        "experiment_id": "supervised_k16_comparison",
        "implementation_base_commit": IMPLEMENTATION_BASE,
        "comparison_config_sha256": COMPARISON_SHA256,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Execution config mismatch for {key}")
    if config.get("base_seeds") != BASE_SEEDS:
        raise ValueError("Execution config has the wrong base-seed matrix")
    if config.get("continuation_seeds") != CONTINUATION_SEEDS:
        raise ValueError("Execution config has the wrong continuation-seed matrix")
    training = config.get("training", {})
    if (
        training.get("exact_k") != 16
        or training.get("stage1", {}).get("updates") != 2315
        or training.get("stage2", {}).get("updates") != 17685
        or training.get("cumulative_updates") != 20000
        or training.get("base_clip_presentations_per_seed") != 80000
    ):
        raise ValueError("Execution config violates the fixed training exposure")
    if not config.get("admission", {}).get("no_hlvid_in_this_admission"):
        raise ValueError("Production training admission must explicitly exclude HLVid")
    return config


def command_precheck(args: argparse.Namespace) -> None:
    config_path = args.execution_config.resolve(strict=True)
    config = load_execution_config(config_path)
    if not 0 <= args.array_index < len(BASE_SEEDS):
        raise ValueError("Array index must select one of the six preregistered seeds")
    comparison = (REPOSITORY_ROOT / config["comparison_config"]).resolve(strict=True)
    manifest = Path(config["data"]["manifest"]).resolve(strict=True)
    cell_mass = Path(config["data"]["cell_mass"]).resolve(strict=True)
    pretrained = Path(config["initialization"]["path"]).resolve(strict=True)
    hashes = {
        "execution_config": sha256_file(config_path),
        "comparison_config": sha256_file(comparison),
        "manifest": sha256_file(manifest),
        "cell_mass": sha256_file(cell_mass),
        "pretrained_model_safetensors": sha256_file(pretrained / "model.safetensors"),
    }
    expected_hashes = {
        "comparison_config": COMPARISON_SHA256,
        "manifest": MANIFEST_SHA256,
        "cell_mass": CELL_MASS_SHA256,
        "pretrained_model_safetensors": PRETRAINED_SHA256,
    }
    for key, expected in expected_hashes.items():
        if hashes[key] != expected:
            raise ValueError(f"Live input hash mismatch for {key}")
    source = verified_source(args.expected_commit)
    index = args.array_index
    record = {
        "schema_version": 1,
        "status": "pass",
        "verified_utc": utc_now(),
        "experiment_id": config["experiment_id"],
        "array_index": index,
        "base_seed": BASE_SEEDS[index],
        "continuation_seed": CONTINUATION_SEEDS[index],
        "source": source,
        "paths": {
            "execution_config": str(config_path),
            "comparison_config": str(comparison),
            "dataset_root": str(Path(config["data"]["root"]).resolve(strict=True)),
            "manifest": str(manifest),
            "cell_mass": str(cell_mass),
            "pretrained": str(pretrained),
        },
        "sha256": hashes,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES",
            )
        },
        "contract": {
            "stage1_updates": 2315,
            "stage2_updates": 17685,
            "cumulative_updates": 20000,
            "base_clip_presentations": 80000,
            "nominal_action_rows": 20480000,
            "hlvid": False,
        },
    }
    write_new_json(args.output, record)
    print(json.dumps({"status": "pass", "output": str(args.output), "base_seed": BASE_SEEDS[index]}))


def expected_resume_contract(stage: str, seed: int) -> dict[str, Any]:
    if stage == "stage1":
        n_epochs, schedule, learning_rate = 5, "linear_w_warmup", 1e-5
    else:
        n_epochs, schedule, learning_rate = 40, "constant", 3e-6
    return {
        "trainer_seed": seed,
        "sampler_seed": seed,
        "teacher_seed": seed,
        "configured_batch_size": 4,
        "configured_per_gpu_max_batch_size": 1,
        "loader_batch_size": 1,
        "loader_batches": 1852,
        "dataset_items": 1854,
        "gradient_accumulation_steps": 4,
        "n_epochs": n_epochs,
        "optimizer": "adam",
        "lr_schedule": schedule,
        "learning_rate": learning_rate,
        "sampler_type": "autogaze.datasets.av_gaze_stavis.BalancedSourceSampler",
        "action_contract": {
            "clip_len": 16,
            "actions_per_frame": 265,
            "fine_action_offset": 69,
            "exact_budget": 16,
        },
    }


def command_checkpoint(args: argparse.Namespace) -> None:
    seed = args.base_seed if args.stage == "stage1" else args.continuation_seed
    expected_step = 2315 if args.stage == "stage1" else 17685
    expected_cursor = (5, 0) if args.stage == "stage1" else (38, 364)
    run_dir = args.run_dir.resolve(strict=True)
    receipt = verify_supervised_completion(
        run_dir,
        expected_contract=expected_resume_contract(args.stage, seed),
    )
    if receipt["train_step"] != expected_step:
        raise ValueError(
            f"{args.stage} completion is at step {receipt['train_step']}, expected {expected_step}"
        )
    train_path = run_dir / "checkpoint_latest_train.pt"
    saved = torch.load(train_path, map_location="cpu", weights_only=True)
    if saved.get("train_step") != expected_step:
        raise ValueError("Serialized training step disagrees with completion receipt")
    if (saved.get("epoch"), saved.get("iteration")) != expected_cursor:
        raise ValueError("Serialized next-input cursor disagrees with the fixed exposure")
    algorithm = saved.get("supervised_algorithm_state", {})
    if algorithm.get("teacher_seed") != seed:
        raise ValueError("Serialized teacher seed disagrees with the paired phase seed")
    record = {
        "schema_version": 1,
        "status": "pass",
        "verified_utc": utc_now(),
        "stage": args.stage,
        "base_seed": args.base_seed,
        "continuation_seed": args.continuation_seed,
        "phase_seed": seed,
        "train_step": expected_step,
        "cumulative_train_step": expected_step if args.stage == "stage1" else 20000,
        "next_input_cursor": list(expected_cursor),
        "run_directory": str(run_dir),
        "completion_marker_sha256": sha256_file(run_dir / "checkpoint_latest_complete.json"),
        "checkpoint_files_sha256": receipt["files_sha256"],
        "resume_contract": receipt["resume_contract"],
    }
    write_new_json(args.output, record)
    print(json.dumps({"status": "pass", "stage": args.stage, "train_step": expected_step}))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def command_manifest(args: argparse.Namespace) -> None:
    paths = {
        "admission_precheck": args.precheck.resolve(strict=True),
        "stage1_completion_verification": args.stage1_verification.resolve(strict=True),
        "stage2_completion_verification": args.stage2_verification.resolve(strict=True),
        "stage1_time": args.stage1_time.resolve(strict=True),
        "stage2_time": args.stage2_time.resolve(strict=True),
        "gpu_telemetry": args.gpu_telemetry.resolve(strict=True),
    }
    precheck = read_json(paths["admission_precheck"])
    stage1 = read_json(paths["stage1_completion_verification"])
    stage2 = read_json(paths["stage2_completion_verification"])
    if any(record.get("status") != "pass" for record in (precheck, stage1, stage2)):
        raise ValueError("Execution manifest requires passing precheck and checkpoint records")
    seed_pair = (precheck.get("base_seed"), precheck.get("continuation_seed"))
    if seed_pair != (stage1.get("base_seed"), stage1.get("continuation_seed")):
        raise ValueError("Stage-one seed identity differs from the admitted pair")
    if seed_pair != (stage2.get("base_seed"), stage2.get("continuation_seed")):
        raise ValueError("Stage-two seed identity differs from the admitted pair")
    if stage1.get("train_step") != 2315 or stage2.get("train_step") != 17685:
        raise ValueError("Training phases do not reach the fixed endpoint")
    record = {
        "schema_version": 1,
        "status": "training_complete_pending_validation_and_final_sacct",
        "completed_utc": utc_now(),
        "experiment_id": "supervised_k16_comparison",
        "run_id": args.run_id,
        "array_index": precheck["array_index"],
        "base_seed": seed_pair[0],
        "continuation_seed": seed_pair[1],
        "source": precheck["source"],
        "input_sha256": precheck["sha256"],
        "fixed_endpoint": {
            "cumulative_updates": 20000,
            "base_clip_presentations": 80000,
            "teacher_trajectories": 80000,
            "nominal_action_rows": 20480000,
        },
        "checkpoint_records": {
            "stage1": stage1,
            "stage2": stage2,
        },
        "resource_records": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "slurm": precheck["slurm"],
        "final_sacct": "pending_after_allocation_exit",
        "scientific_evaluation": "pending_full_validation_all_six_seeds",
        "hlvid_executed": False,
    }
    write_new_json(args.output, record)
    print(json.dumps({"status": record["status"], "output": str(args.output)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    precheck = subparsers.add_parser("precheck")
    precheck.add_argument("--execution-config", type=Path, required=True)
    precheck.add_argument("--expected-commit", required=True)
    precheck.add_argument("--array-index", type=int, required=True)
    precheck.add_argument("--output", type=Path, required=True)
    precheck.set_defaults(function=command_precheck)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    checkpoint.add_argument("--run-dir", type=Path, required=True)
    checkpoint.add_argument("--base-seed", type=int, choices=BASE_SEEDS, required=True)
    checkpoint.add_argument(
        "--continuation-seed", type=int, choices=CONTINUATION_SEEDS, required=True
    )
    checkpoint.add_argument("--output", type=Path, required=True)
    checkpoint.set_defaults(function=command_checkpoint)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--precheck", type=Path, required=True)
    manifest.add_argument("--stage1-verification", type=Path, required=True)
    manifest.add_argument("--stage2-verification", type=Path, required=True)
    manifest.add_argument("--stage1-time", type=Path, required=True)
    manifest.add_argument("--stage2-time", type=Path, required=True)
    manifest.add_argument("--gpu-telemetry", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(function=command_manifest)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
