# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify fixed R5 treatment artifacts before comparative interpretation."""

import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from scripts.human_gaze.verify_r3_temporal_position import nested_differences, read_jsonl
from scripts.human_gaze.write_r5_run_manifests import BASE_SEEDS, manifest, run_name

OFFSET = 200000
CODE_COMMIT = "bb27e4dd58acb4f011d5d7494bbaf2ac4a0cd0df"
JOB_ID = 1683083
SOURCES = ("AVAD", "Coutrot_db1", "Coutrot_db2", "DIEM", "ETMD_av", "SumMe")


def control_name(seed):
    return f"r3b_temporal_control_base{seed}_seed{seed + OFFSET}"


def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def verify_common(config, seed):
    assert int(config["trainer"]["seed"]) == seed + OFFSET
    assert int(config["trainer"]["max_train_steps"]) == 10000
    assert int(config["trainer"]["val_nsteps"]) == 100
    assert float(config["trainer"]["lr"]) == 3e-6
    assert config["trainer"]["lr_schedule"] == "constant"
    assert config["trainer"]["resume"] is False


def verify_rows(directory, max_gate):
    training = read_jsonl(directory / "training_metrics.jsonl")
    validation = read_jsonl(directory / "validation_metrics.jsonl")
    assert [int(row["train_step"]) for row in training] == list(range(10000))
    assert [int(row["train_step"]) for row in validation] == list(range(0, 10001, 100))
    observed_max = 0.0
    for row in training + validation:
        gate = float(row["feature_transport_logit_gate"])
        rms = float(row["feature_transport_logit_bias_rms"])
        assert math.isfinite(gate) and math.isfinite(rms)
        assert abs(abs(gate) - rms) < 1e-7
        assert 0 <= rms <= max_gate
        observed_max = max(observed_max, rms)
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
    parser.add_argument("--max-absolute-logit-gate", type=float, default=1.0)
    args = parser.parse_args()
    allowed = {
        "model.gaze_model_config.feature_transport_temperature",
        "model.gaze_model_config.temporal_position_encoding",
        "trainer.exp_name",
    }
    maxima = {}
    for task_id, seed in enumerate(BASE_SEEDS):
        treatment_dir = args.run_root / run_name(seed)
        observed_manifest = json.loads((treatment_dir / "manifest.json").read_text())
        expected_manifest = manifest(seed, JOB_ID, task_id, CODE_COMMIT)
        if observed_manifest != expected_manifest:
            raise AssertionError(f"Manifest mismatch: {treatment_dir}")
        treatment_config = load_config(treatment_dir / "config.yaml")
        control_config = load_config(args.run_root / control_name(seed) / "config.yaml")
        verify_common(treatment_config, seed)
        verify_common(control_config, seed)
        assert treatment_config["model"]["gaze_model_config"]["temporal_position_encoding"] == "causal_feature_transport_logit_bias"
        assert float(treatment_config["model"]["gaze_model_config"]["feature_transport_temperature"]) == 0.1
        assert control_config["model"]["gaze_model_config"]["temporal_position_encoding"] == "none"
        differences = set(nested_differences(control_config, treatment_config))
        if differences != allowed:
            raise AssertionError(f"Unexpected paired differences for {seed}: {sorted(differences)}")
        maxima[str(seed)] = verify_rows(treatment_dir, args.max_absolute_logit_gate)

    report = {
        "schema_version": 1,
        "status": "verified",
        "job_id": JOB_ID,
        "code_commit": CODE_COMMIT,
        "fixed_updates": 10000,
        "validation_steps": list(range(0, 10001, 100)),
        "allowed_paired_config_differences": sorted(allowed),
        "max_absolute_logit_gate": args.max_absolute_logit_gate,
        "maximum_observed_absolute_logit_gate_by_seed": maxima,
        "all_treatment_training_and_validation_rows_controlled": True,
        "reused_controls": [control_name(seed) for seed in BASE_SEEDS],
        "test_split_opened": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
