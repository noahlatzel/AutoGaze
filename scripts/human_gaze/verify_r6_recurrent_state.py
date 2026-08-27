# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify fixed R6 treatment artifacts before comparative interpretation."""

import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from scripts.human_gaze.verify_r3_temporal_position import nested_differences, read_jsonl
from scripts.human_gaze.write_r6_run_manifests import BASE_SEEDS, manifest, run_name


OFFSET = 200000
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


def verify_rows(directory, limits):
    training = read_jsonl(directory / "training_metrics.jsonl")
    validation = read_jsonl(directory / "validation_metrics.jsonl")
    assert [int(row["train_step"]) for row in training] == list(range(10000))
    assert [int(row["train_step"]) for row in validation] == list(range(0, 10001, 100))
    maximum_gate = 0.0
    maximum_state_rms = 0.0
    maximum_saturation = 0.0
    all_rows = training + validation
    for row in all_rows:
        gate = abs(float(row["recurrent_state_logit_gate"]))
        gate_abs = float(row["recurrent_state_logit_gate_abs"])
        state_rms = float(row["recurrent_state_rms_max"])
        saturation = float(row["recurrent_state_saturation_fraction"])
        finite = float(row["recurrent_state_all_finite"])
        values = (gate, gate_abs, state_rms, saturation)
        assert all(math.isfinite(value) for value in values)
        assert abs(gate - gate_abs) < 1e-7
        assert gate_abs <= limits["max_absolute_gate"]
        assert state_rms <= limits["max_state_rms"]
        assert saturation <= limits["max_state_saturation_fraction"]
        assert finite == 1.0
        maximum_gate = max(maximum_gate, gate_abs)
        maximum_state_rms = max(maximum_state_rms, state_rms)
        maximum_saturation = max(maximum_saturation, saturation)
    assert training[0]["recurrent_state_gate_grad_norm"] > 0
    assert any(row["recurrent_state_cell_grad_norm"] > 0 for row in training[1:])
    assert any(row["recurrent_state_readout_grad_norm"] > 0 for row in training[1:])
    endpoint = validation[-1]
    required = ["coverage_k16_macro_source"] + [
        f"coverage_k16_source_{source}" for source in SOURCES
    ]
    assert all(key in endpoint and math.isfinite(float(endpoint[key])) for key in required)
    assert (directory / "checkpoint_latest_gaze" / "model.safetensors").is_file()
    return {
        "maximum_absolute_gate": maximum_gate,
        "maximum_state_rms": maximum_state_rms,
        "maximum_state_saturation_fraction": maximum_saturation,
        "first_gate_gradient_norm": training[0]["recurrent_state_gate_grad_norm"],
        "maximum_cell_gradient_norm": max(row["recurrent_state_cell_grad_norm"] for row in training),
        "maximum_readout_gradient_norm": max(row["recurrent_state_readout_grad_norm"] for row in training),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("outputs/human_gaze/grpo"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    limits = {
        "max_absolute_gate": 1.0,
        "max_state_rms": 1.000001,
        "max_state_saturation_fraction": 0.25,
    }
    allowed = {
        "model.gaze_model_config.recurrent_state_hidden_dim",
        "model.gaze_model_config.temporal_position_encoding",
        "trainer.exp_name",
    }
    rows = {}
    job_id = None
    for task_id, seed in enumerate(BASE_SEEDS):
        treatment_dir = args.run_root / run_name(seed)
        observed_manifest = json.loads((treatment_dir / "manifest.json").read_text())
        if job_id is None:
            job_id = int(observed_manifest["slurm_array_job_id"])
        assert int(observed_manifest["slurm_array_job_id"]) == job_id
        expected_manifest = manifest(seed, job_id, task_id, args.code_commit)
        if observed_manifest != expected_manifest:
            raise AssertionError(f"Manifest mismatch: {treatment_dir}")
        treatment_config = load_config(treatment_dir / "config.yaml")
        control_config = load_config(args.run_root / control_name(seed) / "config.yaml")
        verify_common(treatment_config, seed)
        verify_common(control_config, seed)
        assert treatment_config["model"]["gaze_model_config"]["temporal_position_encoding"] == (
            "selection_conditioned_recurrent_state_logit_bias"
        )
        assert int(treatment_config["model"]["gaze_model_config"]["recurrent_state_hidden_dim"]) == 192
        assert control_config["model"]["gaze_model_config"]["temporal_position_encoding"] == "none"
        differences = set(nested_differences(control_config, treatment_config))
        if differences != allowed:
            raise AssertionError(f"Unexpected paired differences for {seed}: {sorted(differences)}")
        rows[str(seed)] = verify_rows(treatment_dir, limits)

    report = {
        "schema_version": 1,
        "status": "verified",
        "job_id": job_id,
        "code_commit": args.code_commit,
        "fixed_updates": 10000,
        "validation_steps": list(range(0, 10001, 100)),
        "allowed_paired_config_differences": sorted(allowed),
        "stability_limits": limits,
        "by_seed": rows,
        "all_treatment_training_and_validation_rows_controlled": True,
        "gradient_path_reaches_gate_cell_and_readout": True,
        "reused_controls": [control_name(seed) for seed in BASE_SEEDS],
        "test_split_opened": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
