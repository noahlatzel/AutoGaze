# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create immutable provenance manifests for R4 treatment runs."""

import argparse
import json
from pathlib import Path


BASE_SEEDS = (440826, 440827, 440828)
TRAINING_SEED_OFFSET = 200000


def run_name(base_seed):
    return f"r4_causal_difference_base{base_seed}_seed{base_seed + TRAINING_SEED_OFFSET}"


def manifest(base_seed, job_id, task_id, code_commit):
    if base_seed not in BASE_SEEDS:
        raise ValueError(f"Unexpected base seed: {base_seed}")
    endpoint_seed = base_seed + 100000
    training_seed = base_seed + TRAINING_SEED_OFFSET
    return {
        "schema_version": 1,
        "run_id": run_name(base_seed),
        "arm": "causal_connector_difference",
        "base_seed": base_seed,
        "training_seed": training_seed,
        "source_checkpoint": (
            "outputs/human_gaze/grpo/"
            f"r2b_fresh20k_stage3_base{base_seed}_seed{endpoint_seed}/checkpoint_latest_gaze"
        ),
        "paired_control_run": (
            "outputs/human_gaze/grpo/"
            f"r3b_temporal_control_base{base_seed}_seed{training_seed}"
        ),
        "code_commit": code_commit,
        "tracked_worktree_dirty": False,
        "expected_untracked_outputs_symlink": True,
        "slurm_array_job_id": int(job_id),
        "slurm_array_task_id": int(task_id),
        "hydra_config": "autogaze/configs/av_gaze_stavis_r4_causal_difference.yaml",
        "experiment_config": "experiments/human_gaze/configs/r4_causal_connector_difference.yaml",
        "dataset_manifest": "outputs/human_gaze/d0_stavis_fold1_validated/clips.jsonl",
        "dataset_manifest_sha256": "0ef9fa17881517ea69d9edbaf57f8571c5c58d1da517370746fbe264fe33ff10",
        "cell_mass_sha256": "fed5a6552ab94eeaa02a807959d4c3cda6de9dbadeadc35ee248e46ebc16dfb8",
        "split": {"train": "optimization", "val": "comparison", "test": "unopened"},
        "fixed_updates": 10000,
        "validation_interval": 100,
        "optimizer": "Adam reset from empty state",
        "learning_rate": 3e-6,
        "schedule": "constant",
        "temporal_position_encoding": "causal_connector_difference_scalar_gate",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/human_gaze/grpo"))
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    record = manifest(args.base_seed, args.job_id, args.task_id, args.code_commit)
    path = args.output_root / record["run_id"] / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != record:
            raise ValueError(f"Refusing to overwrite different manifest: {path}")
    else:
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
