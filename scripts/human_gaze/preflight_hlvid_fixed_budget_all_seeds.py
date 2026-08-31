#!/usr/bin/env python3
"""Fail-closed preflight for the all-seed fixed-budget HLVid evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import safetensors
import torch
import transformers
import yaml
from safetensors import safe_open

EXPECTED_PROTOCOL = {
    "frame_source": "uniform",
    "num_video_frames": 128,
    "num_video_frames_thumbnail": 64,
    "max_tiles_video": 48,
    "tile_len": 16,
    "max_new_tokens": 16,
    "thumbnails": "full",
    "torch_dtype": "bfloat16",
    "generation": "greedy",
    "primary_metric": "exact_match_question_micro_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_schema(path: Path) -> dict[str, list[int]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: list(handle.get_slice(key).get_shape()) for key in handle.keys()}


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    failures: list[str] = []

    protocol = config["protocol"]
    for key, expected in EXPECTED_PROTOCOL.items():
        if protocol.get(key) != expected:
            failures.append(f"protocol.{key}={protocol.get(key)!r}, expected {expected!r}")
    if protocol.get("protocol_id") != "hlvid_nvila_mtv48_uniform128_thumbnail64_v1":
        failures.append("unexpected protocol_id")
    tile_policy = protocol["tile_policy"]
    if tile_policy != {
        "exact_decoder_actions_per_autogaze_frame": "budget",
        "allowed_action_ids": [69, 264],
        "unique_actions": True,
        "allow_eos": False,
    }:
        failures.append(f"unexpected tile_policy: {tile_policy}")

    reference = Path(protocol["reference_artifact"])
    if not reference.is_file():
        failures.append(f"missing reference artifact: {reference}")
        reference_summary = {}
    else:
        observed = sha256(reference)
        if observed != protocol["reference_artifact_sha256"]:
            failures.append(f"reference artifact hash {observed}")
        reference_summary = json.loads(reference.read_text())
        for key in ("num_video_frames", "num_video_frames_thumbnail", "max_tiles_video"):
            if reference_summary.get(key) != protocol[key]:
                failures.append(
                    f"reference {key}={reference_summary.get(key)!r}, config={protocol[key]!r}"
                )
        if reference_summary.get("accuracy") != protocol["reference_accuracy"]:
            failures.append("reference accuracy mismatch")

    dataset_root = Path(config["dataset"]["root"])
    parquet_path = dataset_root / config["dataset"]["parquet"]
    parquet_hash = sha256(parquet_path) if parquet_path.is_file() else None
    if parquet_hash != config["dataset"]["parquet_sha256"]:
        failures.append(f"dataset parquet hash {parquet_hash}")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if len(rows) != 268 or len({row["video_path"] for row in rows}) != 77:
        failures.append("HLVid population is not 268 questions / 77 videos")
    missing_videos = [
        row["video_path"]
        for row in rows
        if not (dataset_root / "videos" / row["video_path"]).is_file()
    ]
    if missing_videos:
        failures.append(f"missing HLVid videos: {sorted(set(missing_videos))[:5]}")

    checkpoint_root = Path(config["models"]["checkpoint_root"])
    upstream_path = Path(config["models"]["upstream_autogaze"]) / "model.safetensors"
    upstream_schema = tensor_schema(upstream_path)
    checkpoint_records = []
    seen = set()
    for seed_row in config["seed_matrix"]:
        for budget, relative_path in seed_row["checkpoints"].items():
            checkpoint = checkpoint_root / relative_path
            model_path = checkpoint / "model.safetensors"
            identity = (int(seed_row["base_seed"]), int(budget))
            if identity in seen:
                failures.append(f"duplicate seed/budget: {identity}")
            seen.add(identity)
            if not model_path.is_file():
                failures.append(f"missing checkpoint: {model_path}")
                continue
            schema = tensor_schema(model_path)
            if schema != upstream_schema:
                failures.append(f"tensor schema mismatch: {model_path}")
            checkpoint_records.append(
                {
                    "base_seed": int(seed_row["base_seed"]),
                    "continuation_seed": int(seed_row["continuation_seed"]),
                    "budget": int(budget),
                    "path": str(checkpoint),
                    "model_safetensors_sha256": sha256(model_path),
                    "config_sha256": sha256(checkpoint / "config.json"),
                    "preprocessor_config_sha256": sha256(checkpoint / "preprocessor_config.json"),
                    "tensor_count": len(schema),
                }
            )
    expected_counts = {16: 6, 24: 3, 36: 3}
    k32_present = any(row["budget"] == 32 for row in checkpoint_records)
    if k32_present:
        expected_counts[32] = 3
        if config["k32_integration"].get("status") != "handed_off":
            failures.append("K32 checkpoints require k32_integration.status=handed_off")
    observed_counts = {
        budget: sum(row["budget"] == budget for row in checkpoint_records)
        for budget in expected_counts
    }
    unexpected_budgets = sorted({row["budget"] for row in checkpoint_records} - set(expected_counts))
    if observed_counts != expected_counts or unexpected_budgets:
        failures.append(
            f"checkpoint counts {observed_counts}, expected {expected_counts}; "
            f"unexpected budgets {unexpected_budgets}"
        )

    manifest = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "implementation_files": {
            path: sha256(Path(path))
            for path in [
                "autogaze/human_gaze/hlvid.py",
                "scripts/human_gaze/evaluate_hlvid_nvila_fixed_budget.py",
                "scripts/human_gaze/aggregate_hlvid_fixed_budget_all_seeds.py",
                "experiments/human_gaze/slurm/run_hlvid_nvila_fixed_budget_all_seeds.sbatch",
            ]
        },
        "code_commit": git_value("rev-parse", "HEAD"),
        "code_dirty": bool(git_value("status", "--porcelain")),
        "verified_protocol": {key: protocol[key] for key in EXPECTED_PROTOCOL},
        "protocol_id": protocol["protocol_id"],
        "reference_artifact": {
            "path": str(reference),
            "sha256": sha256(reference),
            "num_correct": reference_summary.get("num_correct"),
            "num_examples": reference_summary.get("num_examples"),
            "accuracy": reference_summary.get("accuracy"),
        },
        "dataset": {
            "path": str(parquet_path),
            "sha256": parquet_hash,
            "questions": len(rows),
            "videos": len({row["video_path"] for row in rows}),
            "missing_video_count": len(missing_videos),
            "benchmark_split": "official_huggingface_test",
            "human_gaze_protected_test_accessed": False,
        },
        "checkpoints": checkpoint_records,
        "checkpoint_counts": observed_counts,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "pyarrow": pyarrow.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "safetensors": safetensors.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "k32_status": config["k32_integration"],
        "planned_counted_jobs": config["slurm"]["planned_array_tasks"],
        "maximum_counted_jobs": config["slurm"]["maximum_counted_jobs"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
