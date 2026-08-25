# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify fixed R4 treatment artifacts before comparative interpretation."""

import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from scripts.human_gaze.verify_r3_temporal_position import nested_differences, read_jsonl
from scripts.human_gaze.write_r4_run_manifests import BASE_SEEDS, manifest, run_name

TRAINING_SEED_OFFSET = 200000
CODE_COMMIT = "9c3c6d27ecc5aea85419b3f9671c4a8b29a07bc9"
JOB_ID = 1681313
SOURCES = ("AVAD", "Coutrot_db1", "Coutrot_db2", "DIEM", "ETMD_av", "SumMe")


def control_name(seed):
    return f"r3b_temporal_control_base{seed}_seed{seed + TRAINING_SEED_OFFSET}"


def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def verify_common(config, seed):
    assert int(config["trainer"]["seed"]) == seed + TRAINING_SEED_OFFSET
    assert int(config["trainer"]["max_train_steps"]) == 10000
    assert int(config["trainer"]["val_nsteps"]) == 100
    assert float(config["trainer"]["lr"]) == 3e-6
    assert config["trainer"]["lr_schedule"] == "constant"
    assert config["trainer"]["resume"] is False


def verify_rows(directory, max_ratio):
    training = read_jsonl(directory / "training_metrics.jsonl")
    validation = read_jsonl(directory / "validation_metrics.jsonl")
    assert [int(row["train_step"]) for row in training] == list(range(10000))
    assert [int(row["train_step"]) for row in validation] == list(range(0, 10001, 100))
    observed_max = 0.0
    for row in training + validation:
        gate = float(row["causal_difference_gate"])
        ratio = float(row["causal_difference_signal_to_feature_rms_ceiling"])
        assert math.isfinite(gate) and math.isfinite(ratio)
        assert abs(abs(gate) - ratio) < 1e-7
        assert 0 <= ratio <= max_ratio
        observed_max = max(observed_max, ratio)
    endpoint = validation[-1]
    required = ["coverage_k16_macro_source"] + [
        f"coverage_k16_source_{source}" for source in SOURCES
    ]
    assert all(key in endpoint and math.isfinite(float(endpoint[key])) for key in required)
    assert (directory / "checkpoint_latest_gaze" / "model.safetensors").is_file()
    return observed_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("outputs/human_gaze/grpo"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-signal-ratio", type=float, default=0.25)
    args = parser.parse_args()
    allowed = {
        "model.gaze_model_config.temporal_position_encoding",
        "trainer.exp_name",
    }
    maxima = {}
    for task_id, seed in enumerate(BASE_SEEDS):
        treatment_dir = args.run_root / run_name(seed)
        observed = json.loads((treatment_dir / "manifest.json").read_text())
        expected = manifest(seed, JOB_ID, task_id, CODE_COMMIT)
        if observed != expected:
            raise AssertionError(f"Manifest mismatch: {treatment_dir}")
        treatment_config = load_config(treatment_dir / "config.yaml")
        control_config = load_config(args.run_root / control_name(seed) / "config.yaml")
        verify_common(treatment_config, seed)
        verify_common(control_config, seed)
        assert (
            treatment_config["model"]["gaze_model_config"]["temporal_position_encoding"]
            == "causal_connector_difference_scalar_gate"
        )
        assert control_config["model"]["gaze_model_config"]["temporal_position_encoding"] == "none"
        differences = set(nested_differences(control_config, treatment_config))
        if differences != allowed:
            raise AssertionError(f"Unexpected paired differences for {seed}: {sorted(differences)}")
        maxima[str(seed)] = verify_rows(treatment_dir, args.max_signal_ratio)

    report = {
        "schema_version": 1,
        "status": "verified",
        "job_id": JOB_ID,
        "code_commit": CODE_COMMIT,
        "fixed_updates": 10000,
        "validation_steps": list(range(0, 10001, 100)),
        "allowed_paired_config_differences": sorted(allowed),
        "max_signal_to_feature_rms": args.max_signal_ratio,
        "maximum_observed_signal_to_feature_rms_by_seed": maxima,
        "all_treatment_training_and_validation_rows_norm_controlled": True,
        "reused_controls": [control_name(seed) for seed in BASE_SEEDS],
        "test_split_opened": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
