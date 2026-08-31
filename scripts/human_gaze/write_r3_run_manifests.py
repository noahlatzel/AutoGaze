# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create immutable preregistered provenance manifests for the paired R3 runs."""

import argparse
import json
from pathlib import Path


BASE_SEEDS = (440826, 440827, 440828)


def manifest(arm, base_seed, job_id, code_commit):
    endpoint_seed = base_seed + 100000
    training_seed = base_seed + 200000
    return {
        "schema_version": 1,
        "run_id": f"r3b_temporal_{arm}_base{base_seed}_seed{training_seed}",
        "arm": arm,
        "base_seed": base_seed,
        "training_seed": training_seed,
        "source_checkpoint": (
            "outputs/human_gaze/grpo/"
            f"r2b_fresh20k_stage3_base{base_seed}_seed{endpoint_seed}/checkpoint_latest_gaze"
        ),
        "code_commit": code_commit,
        "tracked_worktree_dirty": False,
        "expected_untracked_outputs_symlink": True,
        "slurm_array_job_id": int(job_id),
        "slurm_array_task_id": BASE_SEEDS.index(base_seed) + (0 if arm == "control" else 3),
        "hydra_config": "autogaze/configs/av_gaze_stavis_r3_temporal_position.yaml",
        "experiment_config": "experiments/human_gaze/configs/r3_temporal_diagnostics_gated_position.yaml",
        "dataset_manifest": "outputs/human_gaze/d0_stavis_fold1_validated/clips.jsonl",
        "dataset_manifest_sha256": "0ef9fa17881517ea69d9edbaf57f8571c5c58d1da517370746fbe264fe33ff10",
        "cell_mass_sha256": "fed5a6552ab94eeaa02a807959d4c3cda6de9dbadeadc35ee248e46ebc16dfb8",
        "split": {"train": "optimization", "val": "comparison", "test": "unopened"},
        "fixed_updates": 10000,
        "validation_interval": 100,
        "optimizer": "Adam reset from empty state",
        "learning_rate": 3e-6,
        "schedule": "constant",
        "temporal_position_encoding": (
            "none" if arm == "control" else "normalized_sinusoidal_scalar_gate"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/human_gaze/grpo"))
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    for arm in ("control", "position"):
        for seed in BASE_SEEDS:
            record = manifest(arm, seed, args.job_id, args.code_commit)
            path = args.output_root / record["run_id"] / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = json.loads(path.read_text())
                if existing != record:
                    raise ValueError(f"Refusing to overwrite different manifest: {path}")
                continue
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(path)


if __name__ == "__main__":
    main()
