# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify paired R3 artifacts before interpreting comparative results."""

import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from scripts.human_gaze.write_r3_run_manifests import BASE_SEEDS, manifest

TRAINING_SEED_OFFSET = 200000
CODE_COMMIT = "18d1055c265d41cd2f75654b6efb2b6bb3d70ffb"
JOB_ID = 1680265


def run_name(arm, seed):
    return f"r3b_temporal_{arm}_base{seed}_seed{seed + TRAINING_SEED_OFFSET}"


def nested_differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(nested_differences(left[key], right[key], path))
        return differences
    if left != right:
        return [prefix]
    return []


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_temporal_rows(rows, max_ratio):
    observed_max = 0.0
    for row in rows:
        gate = float(row["temporal_position_gate"])
        ratio = float(row["temporal_signal_to_feature_rms"])
        assert math.isfinite(gate) and math.isfinite(ratio)
        assert ratio >= 0.0
        assert abs(abs(gate) - ratio) < 1e-7
        assert ratio <= max_ratio
        observed_max = max(observed_max, ratio)
    return observed_max


def verify_run(root, arm, seed, max_ratio):
    directory = root / run_name(arm, seed)
    observed_manifest = json.loads((directory / "manifest.json").read_text())
    expected_manifest = manifest(arm, seed, JOB_ID, CODE_COMMIT)
    if observed_manifest != expected_manifest:
        raise AssertionError(f"Manifest mismatch: {directory}")
    config = OmegaConf.to_container(
        OmegaConf.load(directory / "config.yaml"), resolve=True
    )
    expected_mode = "none" if arm == "control" else "normalized_sinusoidal_scalar_gate"
    assert (
        config["model"]["gaze_model_config"]["temporal_position_encoding"]
        == expected_mode
    )
    assert int(config["trainer"]["seed"]) == seed + TRAINING_SEED_OFFSET
    assert int(config["trainer"]["max_train_steps"]) == 10000
    assert int(config["trainer"]["val_nsteps"]) == 100
    assert float(config["trainer"]["lr"]) == 3e-6
    assert config["trainer"]["lr_schedule"] == "constant"
    assert config["trainer"]["resume"] is False

    training = read_jsonl(directory / "training_metrics.jsonl")
    validation = read_jsonl(directory / "validation_metrics.jsonl")
    assert [int(row["train_step"]) for row in training] == list(range(10000))
    assert [int(row["train_step"]) for row in validation] == list(range(0, 10001, 100))
    endpoint = validation[-1]
    required = ["coverage_k16_macro_source"] + [
        f"coverage_k16_source_{source}"
        for source in ("AVAD", "Coutrot_db1", "Coutrot_db2", "DIEM", "ETMD_av", "SumMe")
    ]
    assert all(
        key in endpoint and math.isfinite(float(endpoint[key])) for key in required
    )
    assert (directory / "checkpoint_latest_gaze" / "model.safetensors").is_file()
    observed_max_ratio = None
    if arm == "position":
        observed_max_ratio = verify_temporal_rows(training + validation, max_ratio)
    return config, endpoint, observed_max_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", type=Path, default=Path("outputs/human_gaze/grpo")
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-signal-ratio", type=float, default=0.25)
    args = parser.parse_args()
    endpoints = {"control": {}, "position": {}}
    observed_max_ratios = {}
    allowed_differences = {
        "model.gaze_model_config.temporal_position_encoding",
        "trainer.exp_name",
    }
    for seed in BASE_SEEDS:
        control_config, endpoints["control"][str(seed)], _ = verify_run(
            args.run_root, "control", seed, args.max_signal_ratio
        )
        position_config, endpoints["position"][str(seed)], observed_max_ratio = (
            verify_run(args.run_root, "position", seed, args.max_signal_ratio)
        )
        observed_max_ratios[str(seed)] = observed_max_ratio
        differences = set(nested_differences(control_config, position_config))
        if differences != allowed_differences:
            raise AssertionError(
                f"Unexpected paired config differences for {seed}: {sorted(differences)}"
            )
    report = {
        "schema_version": 1,
        "status": "verified",
        "job_id": JOB_ID,
        "code_commit": CODE_COMMIT,
        "fixed_updates": 10000,
        "validation_steps": list(range(0, 10001, 100)),
        "allowed_paired_config_differences": sorted(allowed_differences),
        "max_signal_to_feature_rms": args.max_signal_ratio,
        "maximum_observed_signal_to_feature_rms_by_seed": observed_max_ratios,
        "all_temporal_training_and_validation_rows_norm_controlled": True,
        "test_split_opened": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
