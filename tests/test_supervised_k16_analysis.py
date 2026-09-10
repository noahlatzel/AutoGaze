import json
import math
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.supervised_analysis import (
    aggregate_frame_values,
    agreement_report,
    build_offcenter_manifest,
    evaluate_practical_gate,
    load_action_export,
    policy_report,
    sha256_file,
    six_seed_summary,
    source_stratified_video_bootstrap_delta,
    validate_action_array,
    within_source_different_video_indices,
)
from scripts.human_gaze.curate_supervised_k16_comparison import (
    phase_wall_at,
    resource_summary,
    validate_analysis_config,
    validate_resource_receipt,
)
from scripts.human_gaze.plot_supervised_k16_comparison import plot_convergence


BASE_SEEDS = tuple(range(440826, 440832))


def normalized_mass(num_records: int, frames: int = 2) -> np.ndarray:
    mass = np.zeros((num_records, frames, 196), dtype=np.float64)
    mass[:, :, 0] = 0.7
    mass[:, :, 50] = 0.3
    return mass


def stavis_records(frames: int = 2) -> list[dict]:
    return [
        {
            "clip_id": f"{source}/video{video}/target_000000",
            "source": source,
            "video_id": f"video{video}",
            "split": "val",
            "frame_numbers": list(range(frames)),
            "cell_mass_index": index,
        }
        for index, (source, video) in enumerate(
            (source, video) for source in STAVIS_SOURCES for video in (0, 1)
        )
    ]


def test_offcenter_manifest_uses_ceil_per_source_and_stable_ties() -> None:
    records = []
    for source, count in (("A", 5), ("B", 3)):
        for index in range(count):
            records.append(
                {
                    "clip_id": f"{source}/{chr(ord('a') + index)}",
                    "source": source,
                    "video_id": f"v{index % 2}",
                    "split": "val",
                    "cell_mass_index": len(records),
                }
            )
    center = int(center_order(14)[0])
    outside = 0
    mass = np.zeros((len(records), 2, 196), dtype=np.float64)
    mass[:, :, center] = 1.0
    strengths = [0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    for index, strength in enumerate(strengths):
        mass[index, :, center] = 1 - strength
        mass[index, :, outside] = strength

    report = build_offcenter_manifest(
        records,
        mass,
        manifest_sha256="a" * 64,
        cell_mass_sha256="b" * 64,
        sources=("A", "B"),
    )

    assert report["counts"]["total_eligible_clips"] == 3
    assert report["counts"]["per_source"]["A"]["eligible_clips"] == math.ceil(5 / 4)
    assert report["counts"]["per_source"]["B"]["eligible_clips"] == math.ceil(3 / 4)
    assert [row["clip_id"] for row in report["clips"]] == ["A/a", "A/b", "B/a"]
    assert report["model_outputs_consulted"] is False


def test_action_validation_rejects_repeats_and_non_fine_cells() -> None:
    valid = np.tile(np.arange(16), (2, 3, 1))
    assert validate_action_array(valid, num_records=2, clip_len=3).shape == (2, 3, 16)
    repeated = valid.copy()
    repeated[0, 0, 1] = repeated[0, 0, 0]
    with pytest.raises(ValueError, match="repeats"):
        validate_action_array(repeated, num_records=2, clip_len=3)
    invalid = valid.copy()
    invalid[1, 1, 0] = 196
    with pytest.raises(ValueError, match="out-of-range"):
        validate_action_array(invalid, num_records=2, clip_len=3)


def test_aggregation_is_frames_then_clips_then_videos_then_sources() -> None:
    records = [
        {"clip_id": "A/1", "source": "A", "video_id": "v1"},
        {"clip_id": "A/2", "source": "A", "video_id": "v1"},
        {"clip_id": "A/3", "source": "A", "video_id": "v2"},
        {"clip_id": "B/1", "source": "B", "video_id": "v3"},
    ]
    values = np.asarray([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float64)

    report = aggregate_frame_values(records, values, sources=("A", "B"))

    assert report["per_source"]["A"]["mean_video_value"] == 0.75
    assert report["per_source"]["B"]["mean_video_value"] == 0.0
    assert report["macro_source_mean"] == 0.375
    assert report["pooled_frame_mean"] == 0.5


def test_policy_report_covers_controls_saliency_and_structure() -> None:
    records = stavis_records()
    mass = normalized_mass(len(records))
    actions = np.tile(np.arange(16), (len(records), 2, 1))
    priors = {source: list(range(16)) for source in STAVIS_SOURCES}

    report = policy_report(
        records,
        mass,
        actions,
        offcenter_clip_ids={record["clip_id"] for record in records},
        train_source_prior_cells=priors,
    )

    assert report["coverage"]["actual"]["full_validation"]["macro_source_mean"] == pytest.approx(0.7)
    assert report["coverage"]["train_source_prior16"]["full_validation"]["macro_source_mean"] == pytest.approx(0.7)
    assert report["saliency"]["uniform_selected_density_sim"]["full_validation"]["num_frames"] == 24
    assert report["structure"]["unique_frame_selection_sets"] == 1
    assert report["content_dependence"]["actual_minus_static_macro"] == pytest.approx(0.0)


def test_different_video_shuffle_rejects_single_video_source() -> None:
    records = stavis_records()
    records[1]["video_id"] = records[0]["video_id"]
    with pytest.raises(ValueError, match="at least two videos"):
        within_source_different_video_indices(records)


def test_agreement_reports_identical_and_disjoint_sets() -> None:
    records = stavis_records()
    left = np.tile(np.arange(16), (len(records), 2, 1))
    identical = agreement_report(
        records,
        left,
        left.copy(),
        offcenter_clip_ids={record["clip_id"] for record in records},
    )
    right = np.tile(np.arange(16, 32), (len(records), 2, 1))
    disjoint = agreement_report(
        records,
        left,
        right,
        offcenter_clip_ids={record["clip_id"] for record in records},
    )

    assert identical["set_jaccard"]["full_validation"]["macro_source_mean"] == 1
    assert disjoint["intersection_over_k"]["full_validation"]["macro_source_mean"] == 0


def test_action_export_loader_checks_alignment_and_checksum(tmp_path: Path) -> None:
    records = stavis_records(frames=16)
    directory = tmp_path / "export"
    directory.mkdir()
    actions = np.tile(np.arange(16), (len(records), 16, 1))
    actions_path = directory / "actions.jsonl"
    with actions_path.open("w", encoding="utf-8") as handle:
        for record, cells in zip(records, actions):
            handle.write(
                json.dumps(
                    {
                        "clip_id": record["clip_id"],
                        "source": record["source"],
                        "video_id": record["video_id"],
                        "frame_numbers": record["frame_numbers"],
                        "fine_cells": cells.tolist(),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    provenance = {
        "checkpoint": {
            "tree_sha256": "c" * 64,
            "files_sha256": {"model.safetensors": "d" * 64},
        }
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "method": "supervised",
        "base_seed": 440826,
        "training_seed": 540826,
        "cumulative_update": 20000,
        "split": "val",
        "inputs": {
            "manifest_sha256": "a" * 64,
            "cell_mass_sha256": "b" * 64,
            "checkpoint_tree_sha256": "c" * 64,
            "model_safetensors_sha256": "d" * 64,
        },
        "checkpoint_provenance": provenance,
        "actions_sha256": sha256_file(actions_path),
        "clip_len": 16,
        "exact_k": 16,
        "num_fine_cells": 196,
        "fine_action_offset": 69,
        "allowed_local_action_ids_inclusive": [69, 264],
        "greedy": True,
        "selection_without_replacement": True,
        "num_clips": len(records),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest, loaded = load_action_export(
        directory,
        records,
        expected_manifest_sha256="a" * 64,
        expected_cell_mass_sha256="b" * 64,
        expected_method="supervised",
        expected_base_seed=440826,
        expected_training_seed=540826,
        expected_cumulative_update=20000,
        expected_checkpoint_provenance=provenance,
    )

    assert loaded_manifest["status"] == "complete"
    np.testing.assert_array_equal(loaded, actions)
    with pytest.raises(ValueError, match="training_seed"):
        load_action_export(
            directory,
            records,
            expected_manifest_sha256="a" * 64,
            expected_cell_mass_sha256="b" * 64,
            expected_training_seed=540827,
        )
    with pytest.raises(ValueError, match="checkpoint provenance"):
        load_action_export(
            directory,
            records,
            expected_manifest_sha256="a" * 64,
            expected_cell_mass_sha256="b" * 64,
            expected_checkpoint_provenance={"checkpoint": {}},
        )
    with actions_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_action_export(
            directory,
            records,
            expected_manifest_sha256="a" * 64,
            expected_cell_mass_sha256="b" * 64,
        )


def fake_policy(coverage: float, offcenter: float, static_delta: float, shuffled_delta: float) -> dict:
    return {
        "coverage": {
            "actual": {
                "full_validation": {"macro_source_mean": coverage},
                "offcenter_top_quartile": {"macro_source_mean": offcenter},
            }
        },
        "content_dependence": {
            "actual_minus_static_macro": static_delta,
            "actual_minus_shuffled_macro": shuffled_delta,
        },
    }


def gate_thresholds() -> dict:
    return {
        "route_a": {
            "minimum_supervised_minus_rl_coverage": -0.01,
            "minimum_reduction_in_nominal_training_trajectory_action_rows": 0.5,
        },
        "route_b": {
            "minimum_overall_coverage_improvement": 0.01,
            "or_minimum_offcenter_coverage_improvement": 0.02,
            "minimum_overall_supervised_minus_rl_coverage": -0.02,
        },
    }


def test_practical_gate_fails_closed_then_passes_cost_route() -> None:
    incomplete = evaluate_practical_gate(
        supervised_reports={},
        rl_reports={},
        supervised_resource_complete={},
        supervised_nominal_action_rows=20480000,
        rl_nominal_action_rows=81920000,
        thresholds=gate_thresholds(),
    )
    assert incomplete["status"] == "incomplete_required_evidence"
    assert incomplete["hlvid_admitted"] is False

    supervised = {seed: fake_policy(0.445, 0.50, 0.01, 0.02) for seed in BASE_SEEDS}
    rl = {seed: fake_policy(0.45, 0.49, 0.01, 0.01) for seed in BASE_SEEDS}
    complete = evaluate_practical_gate(
        supervised_reports=supervised,
        rl_reports=rl,
        supervised_resource_complete={seed: True for seed in BASE_SEEDS},
        supervised_nominal_action_rows=20480000,
        rl_nominal_action_rows=81920000,
        thresholds=gate_thresholds(),
    )

    assert complete["route_a_pass"] is True
    assert complete["hlvid_admitted"] is True
    assert complete["evidence"]["nominal_training_trajectory_action_row_reduction"] == 0.75


def test_six_seed_interval_and_paired_video_bootstrap() -> None:
    summary = six_seed_summary([1, 2, 3, 4, 5, 6])
    assert summary["mean"] == 3.5
    assert summary["sample_sd"] > 0
    left = []
    right = []
    for _ in range(6):
        left.append(
            {
                "per_source": {
                    source: {"video_values": {"video": 0.6}} for source in STAVIS_SOURCES
                }
            }
        )
        right.append(
            {
                "per_source": {
                    source: {"video_values": {"video": 0.5}} for source in STAVIS_SOURCES
                }
            }
        )
    bootstrap = source_stratified_video_bootstrap_delta(
        left, right, iterations=20, seed=7
    )
    assert bootstrap["difference"] == pytest.approx(0.1)
    assert bootstrap["ci90_low"] == pytest.approx(0.1)
    assert bootstrap["ci90_high"] == pytest.approx(0.1)


def test_resource_receipt_and_same_seed_recovery_fail_closed(tmp_path: Path) -> None:
    seed = 440826
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    resource_records = {}
    for name in ("stage1_time", "stage2_time", "gpu_telemetry"):
        path = seed_root / f"{name}.txt"
        path.write_text(f"{name}\n", encoding="utf-8")
        resource_records[name] = {"path": str(path), "sha256": sha256_file(path)}
    execution = {
        "status": "training_complete_pending_validation_and_final_sacct",
        "base_seed": seed,
        "continuation_seed": seed + 100000,
        "resource_records": resource_records,
    }
    (seed_root / "execution_manifest.json").write_text(
        json.dumps(execution), encoding="utf-8"
    )
    sacct = {
        "schema_version": 1,
        "status": "complete",
        "terminal_state": "COMPLETED",
        "exit_code": "0:0",
        "base_seed": seed,
        "allocated_gpu_count": 1,
        "requested_host_memory_bytes": 32 * 1024**3,
        "elapsed_seconds": 10,
        "max_rss_bytes": 100,
        "gpu_memory_peak_bytes": 200,
    }
    (seed_root / "final_sacct.json").write_text(json.dumps(sacct), encoding="utf-8")

    complete, detail = validate_resource_receipt(seed_root, seed)
    assert complete is True
    assert detail["status"] == "pass"

    recovery_stage1 = seed_root / "recovery_attempt1" / "stage1"
    recovery_stage2 = seed_root / "recovery_attempt1" / "stage2"
    recovery_stage1.mkdir(parents=True)
    recovery_stage2.mkdir()
    stage1_receipt = seed_root / "stage1_recovery_verification.json"
    stage2_receipt = seed_root / "stage2_recovery_verification.json"
    stage1_receipt.write_text("{}\n", encoding="utf-8")
    stage2_receipt.write_text("{}\n", encoding="utf-8")
    recovery = {
        "schema_version": 1,
        "status": "complete_same_seed_recovery",
        "base_seed": seed,
        "training_seed": seed + 100000,
        "immutable_source_commit": "source",
        "input_sha256": {"manifest": "manifest", "cell_mass": "cell_mass"},
        "failed_job_ids": ["1"],
        "recovery_job_ids": ["2"],
        "resolved_phase_directories": {
            "stage1": str(recovery_stage1),
            "stage2": str(recovery_stage2),
        },
        "checkpoint_verification_receipts": {
            "stage1": {"path": str(stage1_receipt), "sha256": sha256_file(stage1_receipt)},
            "stage2": {"path": str(stage2_receipt), "sha256": sha256_file(stage2_receipt)},
        },
    }
    (seed_root / "recovery_manifest.json").write_text(
        json.dumps(recovery), encoding="utf-8"
    )
    complete, detail = validate_resource_receipt(seed_root, seed)
    assert complete is False
    assert "recovery_attempt_accounting" in detail["problems"]

    sacct["attempts"] = [
        {"job_id": "1", "elapsed_seconds": 5, "allocated_gpu_count": 1},
        {"job_id": "2", "elapsed_seconds": 10, "allocated_gpu_count": 1},
    ]
    sacct["terminal_state"] = "FAILED"
    (seed_root / "final_sacct.json").write_text(json.dumps(sacct), encoding="utf-8")
    complete, detail = validate_resource_receipt(seed_root, seed)
    assert complete is False
    assert "final_sacct_terminal_state" in detail["problems"]


def test_phase_wall_time_rejects_duplicate_steps_and_elapsed_resets() -> None:
    duplicate = [
        {"train_step": 4999, "elapsed_seconds": 1000},
        {"train_step": 4999, "elapsed_seconds": 200},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        phase_wall_at(duplicate, 5000)
    reset = [
        {"train_step": 0, "elapsed_seconds": 10},
        {"train_step": 1, "elapsed_seconds": 5},
    ]
    with pytest.raises(ValueError, match="resets"):
        phase_wall_at(reset, 2)


def test_timing_figure_handles_withheld_supervised_curve_and_names_hardware(
    tmp_path: Path,
) -> None:
    coverage = {"mean": 0.4, "ci90_low": 0.3, "ci90_high": 0.5}
    metrics = {
        "convergence": {
            "hardware": {"supervised": "NVIDIA A40", "rl": "NVIDIA RTX 5000"},
            "supervised_fixed_checkpoints": [
                {
                    "base_clip_presentations": 80000,
                    "nominal_training_trajectory_action_rows": 20480000,
                    "training_wall_seconds": {"status": "withheld"},
                    "coverage": coverage,
                }
            ],
            "rl_existing_actual_logged_updates": [
                {
                    "base_clip_presentations": 80000,
                    "nominal_training_trajectory_action_rows": 81920000,
                    "training_wall_seconds": {
                        "mean": 3600,
                        "ci90_low": 3500,
                        "ci90_high": 3700,
                    },
                    "coverage": coverage,
                }
            ],
        }
    }
    plot_convergence(metrics, tmp_path)
    assert (tmp_path / "supervised_k16_convergence.png").is_file()
    assert (tmp_path / "supervised_k16_convergence.pdf").is_file()
    source = Path("scripts/human_gaze/plot_supervised_k16_comparison.py").read_text(
        encoding="utf-8"
    )
    assert 'legend_label = f"{method} ({method_hardware})"' in source


def test_analysis_config_pins_scientific_matrix() -> None:
    path = Path("experiments/human_gaze/configs/supervised_k16_analysis.yaml")
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(config, dict)
    validate_analysis_config(config)
    config["data"]["exact_k"] = 24
    with pytest.raises(ValueError, match="exact_k"):
        validate_analysis_config(config)


def test_resource_summary_counts_recovery_attempt_time() -> None:
    readiness = {"resource_receipts": {}}
    for seed in BASE_SEEDS:
        sacct = {
            "elapsed_seconds": 10,
            "allocated_gpu_count": 1,
            "max_rss_bytes": 100,
            "gpu_memory_peak_bytes": 200,
        }
        if seed == BASE_SEEDS[0]:
            sacct["attempts"] = [
                {"elapsed_seconds": 5, "allocated_gpu_count": 1},
                {"elapsed_seconds": 10, "allocated_gpu_count": 1},
            ]
        readiness["resource_receipts"][str(seed)] = {
            "complete": True,
            "sacct": sacct,
        }
    summary = resource_summary(readiness)
    assert summary["all_attempt_scheduler_elapsed_seconds"]["mean"] == pytest.approx(
        65 / 6
    )
    assert summary["all_attempt_allocated_gpu_seconds"]["mean"] == pytest.approx(
        65 / 6
    )
