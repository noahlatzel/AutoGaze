# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed verification for the exact-K32 reproduction of R2c K36."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from omegaconf import OmegaConf


SOURCES = ("AVAD", "Coutrot_db1", "Coutrot_db2", "DIEM", "ETMD_av", "SumMe")
COMMON_BASELINE_BUDGETS = (16, 24, 36)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_yaml(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def run_names(base_seed: int) -> tuple[str, str, str]:
    continuation_seed = base_seed + 100000
    return (
        f"r2c_k32_reproduction_stage1_seed{base_seed}",
        f"r2c_k32_reproduction_stage2_base{base_seed}_seed{continuation_seed}",
        f"r2c_k32_reproduction_stage3_base{base_seed}_seed{continuation_seed}",
    )


def reference_run_names(base_seed: int) -> tuple[str, str, str]:
    continuation_seed = base_seed + 100000
    return (
        f"r2c_fixedk36_stage1_seed{base_seed}",
        f"r2c_fixedk36_stage2_base{base_seed}_seed{continuation_seed}",
        f"r2c_fixedk36_stage3_base{base_seed}_seed{continuation_seed}",
    )


def normalized_for_budget_parity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized["task"]["exact_budget"] = "<FIXED_BUDGET>"
    normalized["task"]["report_budgets"] = "<PREFIXES_THROUGH_FIXED_BUDGET>"
    for key in ("exp_name", "gaze_weights", "resume"):
        normalized["trainer"][key] = "<RUN_IDENTITY>"
    return normalized


def nested_differences(left, right, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(nested_differences(left[key], right[key], path))
        return differences
    return [] if left == right else [prefix]


def assert_k32_config(config: dict, expected_seed: int) -> None:
    assert config["dataset"]["manifest_path"].endswith("clips.jsonl")
    assert config["dataset"]["cell_mass_path"].endswith("cell_mass.npy")
    assert int(config["dataset"]["clip_len"]) == 16
    assert int(config["dataset"]["image_size"]) == 224
    assert int(config["dataset"]["grid_size"]) == 14
    assert config["task"]["_target_"].endswith("HumanHeatmapCoverage")
    assert int(config["task"]["exact_budget"]) == 32
    assert list(config["task"]["report_budgets"]) == [1, 2, 4, 8, 16, 24, 32]
    assert int(config["task"]["clip_len"]) * int(config["task"]["exact_budget"]) == 512
    assert int(config["task"]["actions_per_frame"]) - int(config["task"]["fine_action_offset"]) == 196
    assert int(config["trainer"]["seed"]) == expected_seed
    assert bool(config["trainer"]["freeze_gaze_vision"])
    assert bool(config["trainer"]["freeze_gaze_connector"])
    assert bool(config["trainer"]["train_gaze"])
    assert not bool(config["trainer"]["train_task"])
    assert float(config["algorithm"]["kl_coefficient"]) == 0.0
    assert int(config["algorithm"]["group_size"]) == 4


def compare_common_baselines(candidate: dict, reference: dict) -> None:
    assert list(candidate["config"]["evaluation_splits"]) == ["val"]
    for method in ("random", "center", "prior", "oracle"):
        for budget in COMMON_BASELINE_BUDGETS:
            left = candidate["splits"]["val"][method][str(budget)]
            right = reference["splits"]["val"][method][str(budget)]
            if left != right:
                raise AssertionError(f"K{budget} {method} baseline changed from K36")


def verify_preflight(cfg: dict, preflight_run: Path, reference_root: Path) -> dict:
    run_config = load_yaml(preflight_run / "config.yaml")
    assert_k32_config(run_config, expected_seed=940832)
    assert int(run_config["trainer"]["max_train_steps"]) == 1
    assert not bool(run_config["trainer"]["validate_at_start"])
    assert not bool(run_config["trainer"]["save_at_end"])
    assert bool(run_config["trainer"]["skip_final_validation"])
    training = read_jsonl(preflight_run / "training_metrics.jsonl")
    assert [int(row["train_step"]) for row in training] == [0]
    assert all(math.isfinite(float(value)) for value in training[0].values())
    assert "coverage_k32" in training[0]

    candidate_baselines = read_json(Path(cfg["static_baselines"]))
    reference_baselines = read_json(reference_root / "baselines" / "b1_fixed_budget_baselines_val.json")
    compare_common_baselines(candidate_baselines, reference_baselines)
    for method in ("random", "center", "prior", "oracle"):
        assert math.isfinite(float(candidate_baselines["splits"]["val"][method]["32"]["macro_source_mean"]))

    pretrained = read_json(Path(cfg["pretrained"]))
    assert list(pretrained["config"]["evaluation_splits"]) == ["val"]
    assert int(pretrained["config"]["exact_budget"]) == 32
    assert list(pretrained["config"]["report_budgets"]) == [1, 2, 4, 8, 16, 24, 32]
    assert math.isfinite(float(pretrained["metrics"]["val"]["32"]["macro_source_mean"]))
    return {
        "mode": "preflight", "status": "verified",
        "exact_actions_per_frame": 32, "exact_actions_per_clip": 512,
        "one_optimizer_update_completed": True,
        "common_k16_k24_k36_baselines_bit_identical": True,
        "pretrained_k32_complete": True, "test_split_opened": False,
    }


def expected_validation_steps(stage_index: int) -> list[int]:
    if stage_index == 0:
        return [*range(0, 2301, 100), 2315]
    if stage_index == 1:
        return [*range(0, 7601, 100), 7685]
    return [*range(7700, 17601, 100), 17685]


def expected_training_steps(stage_index: int) -> list[int]:
    if stage_index == 0:
        return list(range(2315))
    if stage_index == 1:
        return list(range(7685))
    return list(range(7685, 17685))


def verify_complete(cfg: dict, run_root: Path, diagnostics_root: Path, reference_root: Path) -> dict:
    parity_rows = {}
    endpoint_rows = {}
    for base_seed in [int(value) for value in cfg["base_seeds"]]:
        k32_names = run_names(base_seed)
        k36_names = reference_run_names(base_seed)
        seed_parity = {}
        for stage_index, (k32_name, k36_name) in enumerate(zip(k32_names, k36_names)):
            k32_dir = run_root / k32_name
            k36_dir = reference_root / "grpo" / k36_name
            k32_config = load_yaml(k32_dir / "config.yaml")
            k36_config = load_yaml(k36_dir / "config.yaml")
            expected_seed = base_seed if stage_index == 0 else base_seed + 100000
            assert_k32_config(k32_config, expected_seed)
            differences = nested_differences(
                normalized_for_budget_parity(k32_config), normalized_for_budget_parity(k36_config)
            )
            if differences:
                raise AssertionError(
                    f"Unexpected K32/K36 stage-{stage_index + 1} differences for {base_seed}: {differences}"
                )
            training = read_jsonl(k32_dir / "training_metrics.jsonl")
            validation = read_jsonl(k32_dir / "validation_metrics.jsonl")
            assert [int(row["train_step"]) for row in training] == expected_training_steps(stage_index)
            assert [int(row["train_step"]) for row in validation] == expected_validation_steps(stage_index)
            assert all("coverage_k32" in row for row in training + validation)
            assert (k32_dir / "checkpoint_latest_gaze" / "model.safetensors").is_file()
            seed_parity[f"stage{stage_index + 1}"] = {
                "config_parity_except_budget_and_run_identity": True,
                "training_rows": len(training), "validation_rows": len(validation),
            }
        parity_rows[str(base_seed)] = seed_parity
        endpoint = read_jsonl(run_root / k32_names[2] / "validation_metrics.jsonl")[-1]
        required = ["coverage_k32_macro_source"] + [f"coverage_k32_source_{source}" for source in SOURCES]
        assert int(endpoint["train_step"]) == 17685
        assert all(key in endpoint and math.isfinite(float(endpoint[key])) for key in required)
        endpoint_rows[str(base_seed)] = {key: float(endpoint[key]) for key in required}

        for cumulative_step in [int(value) for value in cfg["diagnostics"]["checkpoints_cumulative"]]:
            name = k32_names[1] if cumulative_step == 10000 else k32_names[2]
            report = read_json(diagnostics_root / name / f"center_collapse_val_step{cumulative_step}.json")
            assert report["split"] == "val"
            assert int(report["processed_clips"]) == 427
            assert set(report["coverage"]) == {"actual", "center", "selection_frequency_top32", "shuffled"}
            assert all(set(value) == {"32"} for value in report["coverage"].values())

    return {
        "mode": "complete", "status": "verified",
        "exact_actions_per_frame": 32, "exact_actions_per_clip": 512,
        "fixed_cumulative_optimizer_updates": 20000,
        "validation_interval": 100, "endpoint_selection": "fixed_step_20000",
        "matched_base_seeds": [int(value) for value in cfg["base_seeds"]],
        "matched_continuation_seeds": [int(value) for value in cfg["continuation_seeds"]],
        "config_parity": parity_rows, "endpoints": endpoint_rows,
        "all_endpoint_diagnostics_complete": True, "test_split_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/human_gaze/grpo"))
    parser.add_argument("--diagnostics-root", type=Path, default=Path("outputs/human_gaze/diagnostics"))
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--preflight-run", type=Path)
    parser.add_argument("--mode", choices=("preflight", "complete"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["experiment_id"] == "r2c_k32_fixed_budget_reproduction"
    assert int(cfg["budget"]) == 32
    assert cfg["decision_rules"]["test_split_opened"] is False
    if args.mode == "preflight":
        if args.preflight_run is None:
            parser.error("--preflight-run is required in preflight mode")
        report = verify_preflight(cfg, args.preflight_run, args.reference_root)
    else:
        report = verify_complete(cfg, args.run_root, args.diagnostics_root, args.reference_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"schema_version": 1, **report}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
