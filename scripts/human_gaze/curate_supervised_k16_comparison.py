#!/usr/bin/env python3
"""Curate the complete six-seed supervised K16 comparison and practical gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autogaze.datasets.av_gaze_stavis import read_manifest
from autogaze.human_gaze.supervised_analysis import (
    agreement_report,
    cell_mass_for_records,
    evaluate_practical_gate,
    load_action_export,
    policy_report,
    selected_mass,
    sha256_file,
    six_seed_summary,
    source_prior_cells,
    source_stratified_video_bootstrap_delta,
)


FIXED_STEPS = (2315, 5000, 10000, 15000, 20000)
BASE_SEEDS = tuple(range(440826, 440832))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/human_gaze/configs/supervised_k16_analysis.yaml"),
    )
    parser.add_argument("--action-root", type=Path)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def validate_analysis_config(cfg: dict[str, Any]) -> None:
    """Reject changes to the preregistered comparison and action support."""
    if cfg.get("schema_version") != 1 or cfg.get("experiment_id") != "supervised_k16_comparison":
        raise ValueError("Unexpected analysis config identity")
    data = cfg.get("data", {})
    expected_data = {
        "split": "val",
        "clips": 427,
        "clip_len": 16,
        "fine_cells": 196,
        "exact_k": 16,
        "fine_action_offset": 69,
    }
    for key, value in expected_data.items():
        if data.get(key) != value:
            raise ValueError(f"Analysis config violates frozen data field {key}")
    if cfg.get("seeds", {}).get("base") != list(BASE_SEEDS):
        raise ValueError("Analysis config violates the frozen base-seed matrix")
    if cfg.get("seeds", {}).get("training") != [seed + 100000 for seed in BASE_SEEDS]:
        raise ValueError("Analysis config violates the frozen training-seed matrix")
    checkpoints = cfg.get("checkpoints", {})
    if (
        checkpoints.get("supervised_cumulative_updates") != list(FIXED_STEPS)
        or checkpoints.get("primary_endpoint_update") != 20000
        or checkpoints.get("rl_action_export_updates") != [20000]
    ):
        raise ValueError("Analysis config violates the frozen checkpoint matrix")
    export = cfg.get("action_export", {})
    if (
        export.get("split") != "val"
        or export.get("decoding") != "greedy_self_history"
        or export.get("exact_k") != 16
        or export.get("allowed_local_action_ids_inclusive") != [69, 264]
        or export.get("no_repeat") is not True
    ):
        raise ValueError("Analysis config violates the frozen action-export contract")
    compute = cfg.get("compute_accounting", {})
    if (
        compute.get("supervised_base_clip_presentations_per_seed") != 80000
        or compute.get("rl_base_clip_presentations_per_seed") != 80000
        or compute.get("supervised_nominal_training_trajectory_action_rows_per_seed") != 20480000
        or compute.get("rl_nominal_training_trajectory_action_rows_per_seed") != 81920000
    ):
        raise ValueError("Analysis config violates matched exposure or nominal action rows")
    offcenter = cfg.get("offcenter_subgroup", {})
    if offcenter.get("total_clips") != 110 or offcenter.get("total_videos") != 23:
        raise ValueError("Analysis config violates the frozen off-center counts")
    recovery = cfg.get("recovery", {})
    if not all(
        recovery.get(key) is True
        for key in (
            "same_seed_only",
            "require_verified_phase_receipts",
            "require_immutable_source_and_data_identity",
            "require_failed_and_recovery_job_ids",
            "require_all_attempts_in_resource_accounting",
        )
    ):
        raise ValueError("Analysis config weakens the frozen recovery contract")


def format_template(template: str, seed: int, step: int | None = None, phase: str | None = None) -> str:
    values = {
        "base_seed": seed,
        "training_seed": seed + 100000,
        "cumulative_update": step,
        "phase": phase,
    }
    return template.format(**values)


def expected_action_paths(cfg: dict, action_root: Path) -> dict[tuple[str, int, int], Path]:
    result = {}
    for seed in BASE_SEEDS:
        for step in FIXED_STEPS:
            relative = format_template(
                cfg["paths"]["supervised_action_export_template"], seed, step
            )
            result[("supervised", seed, step)] = action_root / relative
        relative = format_template(cfg["paths"]["rl_action_export_template"], seed, 20000)
        result[("rl", seed, 20000)] = action_root / relative
    return result


def validate_resource_receipt(seed_root: Path, seed: int) -> tuple[bool, dict[str, Any]]:
    execution_path = seed_root / "execution_manifest.json"
    sacct_path = seed_root / "final_sacct.json"
    missing = [str(path) for path in (execution_path, sacct_path) if not path.is_file()]
    if missing:
        return False, {"status": "missing", "missing": missing}
    execution = load_json(execution_path)
    sacct = load_json(sacct_path)
    problems = []
    if execution.get("base_seed") != seed or execution.get("continuation_seed") != seed + 100000:
        problems.append("execution_seed_identity")
    if not str(execution.get("status", "")).startswith("training_complete"):
        problems.append("execution_not_complete")
    required_sacct = {
        "schema_version": 1,
        "status": "complete",
        "terminal_state": "COMPLETED",
        "exit_code": "0:0",
        "base_seed": seed,
        "allocated_gpu_count": 1,
        "requested_host_memory_bytes": 32 * 1024**3,
    }
    for key, expected in required_sacct.items():
        if sacct.get(key) != expected:
            problems.append(f"final_sacct_{key}")
    for key in (
        "elapsed_seconds",
        "max_rss_bytes",
        "gpu_memory_peak_bytes",
    ):
        value = sacct.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"final_sacct_{key}")
    recovery_path = seed_root / "recovery_manifest.json"
    if recovery_path.is_file():
        recovery = load_json(recovery_path)
        expected_job_ids = {
            str(job_id)
            for key in ("failed_job_ids", "recovery_job_ids")
            for job_id in recovery.get(key, [])
        }
        attempts = sacct.get("attempts")
        if not expected_job_ids or not isinstance(attempts, list) or not attempts:
            problems.append("recovery_attempt_accounting")
        else:
            observed_job_ids = {str(attempt.get("job_id")) for attempt in attempts}
            if not expected_job_ids.issubset(observed_job_ids):
                problems.append("recovery_attempt_job_ids")
            for attempt in attempts:
                elapsed = attempt.get("elapsed_seconds")
                allocated = attempt.get("allocated_gpu_count")
                if (
                    isinstance(elapsed, bool)
                    or not isinstance(elapsed, (int, float))
                    or elapsed <= 0
                    or allocated != 1
                ):
                    problems.append("recovery_attempt_resources")
                    break
    records = execution.get("resource_records", {})
    for name in ("stage1_time", "stage2_time", "gpu_telemetry"):
        record = records.get(name, {})
        path = Path(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            problems.append(f"execution_resource_{name}")
    return not problems, {
        "status": "pass" if not problems else "invalid",
        "problems": problems,
        "execution_manifest": str(execution_path),
        "execution_manifest_sha256": sha256_file(execution_path),
        "final_sacct": str(sacct_path),
        "final_sacct_sha256": sha256_file(sacct_path),
        "sacct": sacct,
    }


def collect_readiness(cfg: dict, action_root: Path, training_root: Path) -> dict[str, Any]:
    action_paths = expected_action_paths(cfg, action_root)
    missing_actions = [
        {"method": method, "base_seed": seed, "cumulative_update": step, "path": str(path)}
        for (method, seed, step), path in action_paths.items()
        if not (path / "manifest.json").is_file()
    ]
    resources = {}
    seed_template = cfg["paths"]["supervised_seed_root_template"]
    for seed in BASE_SEEDS:
        seed_root = training_root / format_template(seed_template, seed)
        complete, detail = validate_resource_receipt(seed_root, seed)
        resources[str(seed)] = {"complete": complete, **detail}
    missing_resources = [seed for seed, value in resources.items() if not value["complete"]]
    return {
        "schema_version": 1,
        "experiment_id": "supervised_k16_comparison",
        "status": (
            "ready_for_complete_cpu_curation"
            if not missing_actions and not missing_resources
            else "waiting_for_declared_inputs"
        ),
        "expected_action_exports": len(action_paths),
        "available_action_exports": len(action_paths) - len(missing_actions),
        "missing_action_exports": missing_actions,
        "resource_receipts": resources,
        "missing_complete_resource_seeds": missing_resources,
        "fail_closed_hlvid_admission": bool(missing_actions or missing_resources),
    }


def load_histories(run: Path, filename: str) -> list[dict[str, Any]]:
    path = run / filename
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolved_phase_dir(
    seed_root: Path,
    seed: int,
    phase: str,
    *,
    immutable_source_commit: str,
    manifest_sha256: str,
    cell_mass_sha256: str,
) -> Path:
    """Honor an explicit same-seed recovery map without hiding the aborted run."""
    recovery_path = seed_root / "recovery_manifest.json"
    if not recovery_path.is_file():
        return seed_root / phase
    recovery = load_json(recovery_path)
    if (
        recovery.get("schema_version") != 1
        or recovery.get("status") != "complete_same_seed_recovery"
        or recovery.get("base_seed") != seed
        or recovery.get("training_seed") != seed + 100000
    ):
        raise ValueError(f"Invalid same-seed recovery manifest: {recovery_path}")
    if recovery.get("immutable_source_commit") != immutable_source_commit:
        raise ValueError("Recovery manifest source identity drift")
    expected_inputs = {"manifest": manifest_sha256, "cell_mass": cell_mass_sha256}
    if recovery.get("input_sha256") != expected_inputs:
        raise ValueError("Recovery manifest input identity drift")
    for key in ("failed_job_ids", "recovery_job_ids"):
        job_ids = recovery.get(key)
        if not isinstance(job_ids, list) or not job_ids:
            raise ValueError(f"Recovery manifest lacks {key}")
    value = recovery.get("resolved_phase_directories", {}).get(phase)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Recovery manifest lacks resolved {phase} directory")
    directory = Path(value)
    if not directory.is_absolute():
        directory = seed_root / directory
    directory = directory.resolve(strict=True)
    receipt = recovery.get("checkpoint_verification_receipts", {}).get(phase, {})
    receipt_path = Path(receipt.get("path", ""))
    if not receipt_path.is_absolute():
        receipt_path = seed_root / receipt_path
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt.get("sha256"):
        raise ValueError(f"Recovery manifest lacks a verified {phase} checkpoint receipt")
    return directory


def phase_wall_at(training_history: list[dict[str, Any]], phase_step: int) -> float:
    if phase_step == 0:
        return 0.0
    candidates = [
        row for row in training_history if int(row["train_step"]) <= phase_step - 1
    ]
    if not candidates:
        raise ValueError(f"Training history does not reach phase step {phase_step}")
    chosen = max(candidates, key=lambda row: int(row["train_step"]))
    if int(chosen["train_step"]) != phase_step - 1:
        raise ValueError(f"Training history lacks exact update preceding checkpoint {phase_step}")
    return float(chosen["elapsed_seconds"])


def supervised_training_wall(cfg: dict, training_root: Path, seed: int) -> dict[int, float]:
    seed_root = training_root / format_template(
        cfg["paths"]["supervised_seed_root_template"], seed
    )
    phase_identity = {
        "immutable_source_commit": cfg["submitted_execution"]["immutable_source_commit"],
        "manifest_sha256": cfg["data"]["manifest_sha256"],
        "cell_mass_sha256": cfg["data"]["cell_mass_sha256"],
    }
    stage1 = load_histories(
        resolved_phase_dir(seed_root, seed, "stage1", **phase_identity),
        "training_metrics.jsonl",
    )
    stage2 = load_histories(
        resolved_phase_dir(seed_root, seed, "stage2", **phase_identity),
        "training_metrics.jsonl",
    )
    stage1_final = phase_wall_at(stage1, 2315)
    return {
        2315: stage1_final,
        **{
            step: stage1_final + phase_wall_at(stage2, step - 2315)
            for step in FIXED_STEPS[1:]
        },
    }


def rl_validation_curve(cfg: dict, seed: int) -> list[dict[str, float]]:
    root = Path(cfg["paths"]["rl_root"])
    training_seed = seed + 100000
    runs = [
        root / format_template(cfg["paths"]["rl_stage1_template"], seed),
        root / format_template(cfg["paths"]["rl_stage2_template"], seed),
        root / format_template(cfg["paths"]["rl_stage3_template"], seed),
    ]
    histories = [load_histories(run, "validation_metrics.jsonl") for run in runs]
    phase_final_elapsed = [float(history[-1]["elapsed_seconds"]) for history in histories]
    records = []
    for phase, history in enumerate(histories):
        for row in history:
            phase_step = int(row["train_step"])
            if phase == 0:
                cumulative_step = phase_step
                wall = float(row["elapsed_seconds"])
            elif phase == 1:
                if phase_step == 0:
                    continue
                cumulative_step = 2315 + phase_step
                wall = phase_final_elapsed[0] + float(row["elapsed_seconds"])
            else:
                if phase_step <= 7685:
                    continue
                cumulative_step = 2315 + phase_step
                wall = sum(phase_final_elapsed[:2]) + float(row["elapsed_seconds"])
            records.append(
                {
                    "cumulative_update": cumulative_step,
                    "base_clip_presentations": cumulative_step * 4,
                    "nominal_training_trajectory_action_rows": cumulative_step * 4 * 4 * 16 * 16,
                    "training_wall_seconds": wall,
                    "training_gpu_seconds": wall,
                    "coverage_macro_source": float(row["coverage_k16_macro_source"]),
                }
            )
    by_step = {int(row["cumulative_update"]): row for row in records}
    return [by_step[step] for step in sorted(by_step)]


def summarize_convergence(
    cfg: dict,
    training_root: Path,
    supervised_reports: dict[int, dict[int, dict]],
) -> dict[str, Any]:
    supervised_by_seed = {}
    for seed in BASE_SEEDS:
        wall = supervised_training_wall(cfg, training_root, seed)
        supervised_by_seed[str(seed)] = [
            {
                "cumulative_update": step,
                "base_clip_presentations": step * 4,
                "nominal_training_trajectory_action_rows": step * 4 * 16 * 16,
                "training_wall_seconds": wall[step],
                "training_gpu_seconds": wall[step],
                "coverage_macro_source": supervised_reports[step][seed]["coverage"]["actual"]["full_validation"]["macro_source_mean"],
            }
            for step in FIXED_STEPS
        ]
    rl_by_seed = {str(seed): rl_validation_curve(cfg, seed) for seed in BASE_SEEDS}
    supervised_summary = []
    for step in FIXED_STEPS:
        rows = [
            next(row for row in supervised_by_seed[str(seed)] if row["cumulative_update"] == step)
            for seed in BASE_SEEDS
        ]
        supervised_summary.append(
            {
                "cumulative_update": step,
                "base_clip_presentations": step * 4,
                "nominal_training_trajectory_action_rows": step * 4 * 16 * 16,
                "coverage": six_seed_summary([row["coverage_macro_source"] for row in rows]),
                "training_wall_seconds": six_seed_summary([row["training_wall_seconds"] for row in rows]),
            }
        )
    common_rl_steps = sorted(
        set.intersection(
            *[
                {int(row["cumulative_update"]) for row in rl_by_seed[str(seed)]}
                for seed in BASE_SEEDS
            ]
        )
    )
    rl_summary = []
    for step in common_rl_steps:
        rows = [
            next(row for row in rl_by_seed[str(seed)] if row["cumulative_update"] == step)
            for seed in BASE_SEEDS
        ]
        rl_summary.append(
            {
                "cumulative_update": step,
                "base_clip_presentations": step * 4,
                "nominal_training_trajectory_action_rows": step * 4 * 4 * 16 * 16,
                "coverage": six_seed_summary([row["coverage_macro_source"] for row in rows]),
                "training_wall_seconds": six_seed_summary([row["training_wall_seconds"] for row in rows]),
            }
        )
    return {
        "supervised_fixed_checkpoints": supervised_summary,
        "rl_existing_actual_logged_updates": rl_summary,
        "per_seed": {"supervised": supervised_by_seed, "rl": rl_by_seed},
        "wall_time_scope": "training_process_elapsed_including_periodic_validation",
        "gpu_time_scope": "single_gpu_wall_time_on_method_specific_hardware",
        "hardware": {"supervised": "A40", "rl": cfg["compute_accounting"]["rl_hardware"]},
        "warning": "Hardware-specific wall/GPU time is descriptive; nominal action rows are not FLOPs.",
    }


def agreement_summary(
    records: list[dict],
    supervised_actions: dict[int, np.ndarray],
    rl_actions: dict[int, np.ndarray],
    offcenter_ids: set[str],
) -> dict[str, Any]:
    paired = {}
    for seed in BASE_SEEDS:
        paired[str(seed)] = agreement_report(
            records,
            supervised_actions[seed],
            rl_actions[seed],
            offcenter_clip_ids=offcenter_ids,
        )
    paired_summary = {}
    for metric in ("intersection_over_k", "set_jaccard"):
        paired_summary[metric] = {}
        for subgroup in ("full_validation", "offcenter_top_quartile"):
            paired_summary[metric][subgroup] = six_seed_summary(
                [paired[str(seed)][metric][subgroup]["macro_source_mean"] for seed in BASE_SEEDS]
            )

    rl_pairs = []
    for left, right in combinations(BASE_SEEDS, 2):
        report = agreement_report(
            records,
            rl_actions[left],
            rl_actions[right],
            offcenter_clip_ids=offcenter_ids,
        )
        rl_pairs.append(
            {
                "left_base_seed": left,
                "right_base_seed": right,
                "intersection_over_k_full_macro": report["intersection_over_k"]["full_validation"]["macro_source_mean"],
                "set_jaccard_full_macro": report["set_jaccard"]["full_validation"]["macro_source_mean"],
                "intersection_over_k_offcenter_macro": report["intersection_over_k"]["offcenter_top_quartile"]["macro_source_mean"],
                "set_jaccard_offcenter_macro": report["set_jaccard"]["offcenter_top_quartile"]["macro_source_mean"],
            }
        )
    rl_context = {}
    for key in (
        "intersection_over_k_full_macro",
        "set_jaccard_full_macro",
        "intersection_over_k_offcenter_macro",
        "set_jaccard_offcenter_macro",
    ):
        values = np.asarray([row[key] for row in rl_pairs], dtype=np.float64)
        rl_context[key] = {
            "num_unordered_seed_pairs": 15,
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "note": "Pairs share seeds and are contextual, not independent replicates.",
        }
    return {
        "paired_supervised_vs_rl": paired,
        "paired_supervised_vs_rl_summary": paired_summary,
        "rl_vs_rl_context": {"pairs": rl_pairs, "summary": rl_context},
    }


def qualitative_selection(
    fixed_manifest: dict,
    records: list[dict],
    cell_mass: np.ndarray,
    supervised_actions: np.ndarray,
    rl_actions: np.ndarray,
) -> dict[str, Any]:
    index = {record["clip_id"]: position for position, record in enumerate(records)}
    supervised_coverage = selected_mass(cell_mass, supervised_actions)
    rl_coverage = selected_mass(cell_mass, rl_actions)
    rows = []
    for example in fixed_manifest["examples"]:
        clip_id = example["clip_id"]
        if clip_id not in index:
            raise ValueError(f"Fixed qualitative clip is absent: {clip_id}")
        position = index[clip_id]
        start = int(example["window_start"])
        stop = start + int(fixed_manifest["selection"]["frames_per_clip"])
        if not 0 <= start < stop <= cell_mass.shape[1]:
            raise ValueError(f"Fixed qualitative window is invalid: {clip_id}")
        sl = float(supervised_coverage[position, start:stop].mean())
        rl = float(rl_coverage[position, start:stop].mean())
        rows.append(
            {
                "clip_id": clip_id,
                "source": example["source"],
                "video_id": example["video_id"],
                "source_rank": int(example["source_rank"]),
                "window_start": start,
                "window_stop_exclusive": stop,
                "supervised_window_coverage": sl,
                "rl_window_coverage": rl,
                "supervised_minus_rl_window_coverage": sl - rl,
            }
        )
    selections = []
    for source in sorted({row["source"] for row in rows}):
        candidates = [row for row in rows if row["source"] == source]
        median = float(np.median([row["supervised_minus_rl_window_coverage"] for row in candidates]))
        representative = min(
            candidates,
            key=lambda row: (
                abs(row["supervised_minus_rl_window_coverage"] - median),
                row["clip_id"],
            ),
        )
        failure = min(
            candidates,
            key=lambda row: (row["supervised_minus_rl_window_coverage"], row["clip_id"]),
        )
        selections.append(
            {
                "source": source,
                "source_median_delta": median,
                "representative_clip_id": representative["clip_id"],
                "failure_clip_id": failure["clip_id"],
                "roles_may_share_a_clip": representative["clip_id"] == failure["clip_id"],
            }
        )
    return {
        "fixed_panel_base_seed": 440826,
        "fixed_panel_clips": len(rows),
        "selection_rules": {
            "representative": "within-source closest to median SL-minus-RL four-frame window coverage, then clip ID",
            "failure": "within-source lowest SL-minus-RL four-frame window coverage, then clip ID",
        },
        "all_fixed_panel_rows": rows,
        "selected_roles": selections,
    }


def compact_policy_row(method: str, seed: int, step: int, report: dict) -> dict[str, Any]:
    coverage = report["coverage"]
    saliency = report["saliency"]
    return {
        "method": method,
        "base_seed": seed,
        "training_seed": seed + 100000,
        "cumulative_update": step,
        "coverage_full_macro": coverage["actual"]["full_validation"]["macro_source_mean"],
        "coverage_offcenter_macro": coverage["actual"]["offcenter_top_quartile"]["macro_source_mean"],
        "coverage_static_macro": coverage["selection_frequency_top16"]["full_validation"]["macro_source_mean"],
        "coverage_shuffled_macro": coverage["same_source_shuffled_video"]["full_validation"]["macro_source_mean"],
        "coverage_center_macro": coverage["center16"]["full_validation"]["macro_source_mean"],
        "coverage_train_prior_macro": coverage["train_source_prior16"]["full_validation"]["macro_source_mean"],
        "sim_full_macro": saliency["uniform_selected_density_sim"]["full_validation"]["macro_source_mean"],
        "cc_full_macro": saliency["uniform_selected_density_cc"]["full_validation"]["macro_source_mean"],
        "sim_offcenter_macro": saliency["uniform_selected_density_sim"]["offcenter_top_quartile"]["macro_source_mean"],
        "cc_offcenter_macro": saliency["uniform_selected_density_cc"]["offcenter_top_quartile"]["macro_source_mean"],
        "center_overlap_full_macro": report["structure"]["center16_overlap_fraction"]["full_validation"]["macro_source_mean"],
        "normalized_selection_entropy": report["structure"]["normalized_selection_entropy"],
        "actual_minus_static_macro": report["content_dependence"]["actual_minus_static_macro"],
        "actual_minus_shuffled_macro": report["content_dependence"]["actual_minus_shuffled_macro"],
    }


def endpoint_diagnostic_summary(
    supervised_reports: dict[int, dict], rl_reports: dict[int, dict]
) -> dict[str, Any]:
    result = {}
    for method, reports in (("supervised", supervised_reports), ("rl", rl_reports)):
        result[method] = {
            "coverage_full": six_seed_summary(
                [reports[seed]["coverage"]["actual"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "coverage_offcenter": six_seed_summary(
                [reports[seed]["coverage"]["actual"]["offcenter_top_quartile"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "coverage_static": six_seed_summary(
                [reports[seed]["coverage"]["selection_frequency_top16"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "coverage_shuffled": six_seed_summary(
                [reports[seed]["coverage"]["same_source_shuffled_video"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "coverage_center": six_seed_summary(
                [reports[seed]["coverage"]["center16"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "coverage_train_prior": six_seed_summary(
                [reports[seed]["coverage"]["train_source_prior16"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "sim_full": six_seed_summary(
                [reports[seed]["saliency"]["uniform_selected_density_sim"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "cc_full": six_seed_summary(
                [reports[seed]["saliency"]["uniform_selected_density_cc"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "center_overlap": six_seed_summary(
                [reports[seed]["structure"]["center16_overlap_fraction"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
            ),
            "selection_entropy": six_seed_summary(
                [reports[seed]["structure"]["normalized_selection_entropy"] for seed in BASE_SEEDS]
            ),
            "per_source": {
                source: {
                    "coverage_full": six_seed_summary(
                        [reports[seed]["coverage"]["actual"]["full_validation"]["per_source"][source]["mean_video_value"] for seed in BASE_SEEDS]
                    ),
                    "coverage_offcenter": six_seed_summary(
                        [reports[seed]["coverage"]["actual"]["offcenter_top_quartile"]["per_source"][source]["mean_video_value"] for seed in BASE_SEEDS]
                    ),
                }
                for source in sorted(reports[BASE_SEEDS[0]]["coverage"]["actual"]["full_validation"]["per_source"])
            },
        }
    return result


def resource_summary(readiness: dict[str, Any]) -> dict[str, Any]:
    elapsed = []
    gpu_seconds = []
    max_rss = []
    gpu_peak = []
    for seed in BASE_SEEDS:
        receipt = readiness["resource_receipts"][str(seed)]
        if not receipt.get("complete"):
            raise ValueError("Resource summary requires all six complete seed receipts")
        sacct = receipt["sacct"]
        attempts = sacct.get("attempts")
        if isinstance(attempts, list) and attempts:
            elapsed.append(float(sum(row["elapsed_seconds"] for row in attempts)))
            gpu_seconds.append(
                float(
                    sum(
                        row["elapsed_seconds"] * row["allocated_gpu_count"]
                        for row in attempts
                    )
                )
            )
        else:
            elapsed.append(float(sacct["elapsed_seconds"]))
            gpu_seconds.append(
                float(sacct["elapsed_seconds"] * sacct["allocated_gpu_count"])
            )
        max_rss.append(float(sacct["max_rss_bytes"]))
        gpu_peak.append(float(sacct["gpu_memory_peak_bytes"]))
    return {
        "hardware": "NVIDIA A40",
        "allocated_gpus_per_attempt": 1,
        "requested_host_memory_bytes_per_attempt": 32 * 1024**3,
        "all_attempt_scheduler_elapsed_seconds": six_seed_summary(elapsed),
        "all_attempt_allocated_gpu_seconds": six_seed_summary(gpu_seconds),
        "per_seed_max_rss_bytes": six_seed_summary(max_rss),
        "per_seed_gpu_memory_peak_bytes": six_seed_summary(gpu_peak),
        "recovery_attempts_included_when_present": True,
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_result_readme(path: Path, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    endpoint = metrics["endpoint_coverage"]
    lines = [
        "# S1 supervised human-gaze K16 comparison",
        "",
        "Status: complete six-seed fixed-endpoint validation curation.",
        "",
        (
            "Primary source/video-macro K16 coverage: "
            f"supervised {endpoint['supervised']['mean']:.6f}, "
            f"RL {endpoint['rl']['mean']:.6f}, paired delta "
            f"{endpoint['paired_supervised_minus_rl']['mean']:+.6f}."
        ),
        (
            "Human-only off-center paired coverage delta: "
            f"{endpoint['offcenter_paired_supervised_minus_rl']['mean']:+.6f}."
        ),
        "",
        f"Practical HLVid gate: `{gate['decision']}`.",
        "",
        "Intervals in `metrics.json` are descriptive and unadjusted; validation was historically explored.",
        "Nominal training trajectory action rows are an exposure/accounting quantity, not FLOPs or measured runtime.",
    ]
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def git_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()

    return {
        "repository": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain=v1", "--untracked-files=all")),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace curation output: {args.output_dir}")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Analysis config must resolve to a mapping")
    validate_analysis_config(cfg)
    action_root = args.action_root or Path(cfg["paths"]["action_export_root"])
    training_root = args.training_root or Path(cfg["paths"]["supervised_training_root"])
    args.output_dir.mkdir(parents=True)
    readiness = collect_readiness(cfg, action_root, training_root)
    write_json(args.output_dir / "readiness.json", readiness)
    if args.readiness_only:
        print(json.dumps({"status": readiness["status"], "output": str(args.output_dir)}))
        return
    if readiness["status"] != "ready_for_complete_cpu_curation":
        gate = evaluate_practical_gate(
            supervised_reports={},
            rl_reports={},
            supervised_resource_complete={},
            supervised_nominal_action_rows=20480000,
            rl_nominal_action_rows=81920000,
            thresholds=cfg["practical_hlvid_gate"],
        )
        write_json(args.output_dir / "gate.json", gate)
        raise SystemExit("Required action/resource inputs are incomplete; HLVid remains closed")

    manifest_path = Path(cfg["data"]["manifest"])
    cell_mass_path = Path(cfg["data"]["cell_mass"])
    if sha256_file(manifest_path) != cfg["data"]["manifest_sha256"]:
        raise ValueError("STAViS manifest hash drift")
    if sha256_file(cell_mass_path) != cfg["data"]["cell_mass_sha256"]:
        raise ValueError("STAViS cell-mass hash drift")
    offcenter_path = resolve_repo_path(cfg["offcenter_subgroup"]["manifest"])
    if sha256_file(offcenter_path) != cfg["offcenter_subgroup"]["manifest_sha256"]:
        raise ValueError("Frozen off-center manifest hash drift")
    offcenter = load_json(offcenter_path)
    if offcenter.get("selection_sha256") != cfg["offcenter_subgroup"]["selection_sha256"]:
        raise ValueError("Frozen off-center selection identity drift")
    offcenter_ids = {row["clip_id"] for row in offcenter["clips"]}
    all_records = read_manifest(manifest_path)
    cache = np.load(cell_mass_path, mmap_mode="r")
    all_mass = cell_mass_for_records(all_records, cache)
    prior_cells = source_prior_cells(all_records, all_mass)
    records = [record for record in all_records if record["split"] == "val"]
    mass = cell_mass_for_records(records, cache)

    action_paths = expected_action_paths(cfg, action_root)
    supervised_reports: dict[int, dict[int, dict]] = {step: {} for step in FIXED_STEPS}
    supervised_actions: dict[int, np.ndarray] = {}
    rl_reports: dict[int, dict] = {}
    rl_actions: dict[int, np.ndarray] = {}
    action_manifests = []
    compact_rows = []
    for (method, seed, step), directory in sorted(action_paths.items()):
        export_manifest, actions = load_action_export(
            directory,
            records,
            expected_manifest_sha256=cfg["data"]["manifest_sha256"],
            expected_cell_mass_sha256=cfg["data"]["cell_mass_sha256"],
            expected_method=method,
            expected_base_seed=seed,
            expected_training_seed=seed + 100000,
            expected_cumulative_update=step,
        )
        report = policy_report(
            records,
            mass,
            actions,
            offcenter_clip_ids=offcenter_ids,
            train_source_prior_cells=prior_cells,
        )
        if method == "supervised":
            supervised_reports[step][seed] = report
            if step == 20000:
                supervised_actions[seed] = actions
        else:
            rl_reports[seed] = report
            rl_actions[seed] = actions
        compact_rows.append(compact_policy_row(method, seed, step, report))
        action_manifests.append(
            {
                "method": method,
                "base_seed": seed,
                "cumulative_update": step,
                "path": str(directory / "manifest.json"),
                "sha256": sha256_file(directory / "manifest.json"),
                "checkpoint_tree_sha256": export_manifest["inputs"]["checkpoint_tree_sha256"],
                "actions_sha256": export_manifest["actions_sha256"],
                "runtime": export_manifest["runtime"],
            }
        )

    endpoint_sl = supervised_reports[20000]
    endpoint_coverage = {
        "supervised": six_seed_summary(
            [endpoint_sl[seed]["coverage"]["actual"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
        ),
        "rl": six_seed_summary(
            [rl_reports[seed]["coverage"]["actual"]["full_validation"]["macro_source_mean"] for seed in BASE_SEEDS]
        ),
        "paired_supervised_minus_rl": six_seed_summary(
            [
                endpoint_sl[seed]["coverage"]["actual"]["full_validation"]["macro_source_mean"]
                - rl_reports[seed]["coverage"]["actual"]["full_validation"]["macro_source_mean"]
                for seed in BASE_SEEDS
            ]
        ),
        "offcenter_paired_supervised_minus_rl": six_seed_summary(
            [
                endpoint_sl[seed]["coverage"]["actual"]["offcenter_top_quartile"]["macro_source_mean"]
                - rl_reports[seed]["coverage"]["actual"]["offcenter_top_quartile"]["macro_source_mean"]
                for seed in BASE_SEEDS
            ]
        ),
    }
    endpoint_diagnostics = endpoint_diagnostic_summary(endpoint_sl, rl_reports)
    bootstrap = {
        "full_validation": source_stratified_video_bootstrap_delta(
            [endpoint_sl[seed]["coverage"]["actual"]["full_validation"] for seed in BASE_SEEDS],
            [rl_reports[seed]["coverage"]["actual"]["full_validation"] for seed in BASE_SEEDS],
            iterations=int(cfg["uncertainty"]["bootstrap_iterations"]),
            seed=int(cfg["uncertainty"]["bootstrap_seed"]),
        ),
        "offcenter_top_quartile": source_stratified_video_bootstrap_delta(
            [endpoint_sl[seed]["coverage"]["actual"]["offcenter_top_quartile"] for seed in BASE_SEEDS],
            [rl_reports[seed]["coverage"]["actual"]["offcenter_top_quartile"] for seed in BASE_SEEDS],
            iterations=int(cfg["uncertainty"]["bootstrap_iterations"]),
            seed=int(cfg["uncertainty"]["bootstrap_seed"]),
        ),
    }
    agreement = agreement_summary(
        records, supervised_actions, rl_actions, offcenter_ids
    )
    fixed_qualitative_path = resolve_repo_path(cfg["qualitative"]["fixed_manifest"])
    if sha256_file(fixed_qualitative_path) != cfg["qualitative"]["fixed_manifest_sha256"]:
        raise ValueError("Fixed 24-clip qualitative manifest hash drift")
    qualitative = qualitative_selection(
        load_json(fixed_qualitative_path),
        records,
        mass,
        supervised_actions[440826],
        rl_actions[440826],
    )
    convergence = summarize_convergence(cfg, training_root, supervised_reports)
    resources = resource_summary(readiness)
    resource_complete = {
        seed: readiness["resource_receipts"][str(seed)]["complete"] for seed in BASE_SEEDS
    }
    gate = evaluate_practical_gate(
        supervised_reports=endpoint_sl,
        rl_reports=rl_reports,
        supervised_resource_complete=resource_complete,
        supervised_nominal_action_rows=int(
            cfg["compute_accounting"]["supervised_nominal_training_trajectory_action_rows_per_seed"]
        ),
        rl_nominal_action_rows=int(
            cfg["compute_accounting"]["rl_nominal_training_trajectory_action_rows_per_seed"]
        ),
        thresholds=cfg["practical_hlvid_gate"],
    )
    metrics = {
        "schema_version": 1,
        "status": "complete",
        "primary_metric": cfg["metrics"]["primary"],
        "endpoint_coverage": endpoint_coverage,
        "endpoint_diagnostics": endpoint_diagnostics,
        "paired_video_bootstrap": bootstrap,
        "agreement": agreement,
        "convergence": convergence,
        "resources": resources,
        "qualitative_selection": qualitative,
        "per_seed_checkpoint_rows": compact_rows,
        "offcenter_counts": offcenter["counts"],
        "interpretation": "Descriptive and unadjusted; validation was historically explored.",
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "gate.json", gate)
    OmegaConf.save(
        config=OmegaConf.create(cfg),
        f=args.output_dir / "config.yaml",
    )
    write_result_readme(args.output_dir / "README.md", metrics, gate)
    with (args.output_dir / "per_seed_checkpoint_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact_rows[0]))
        writer.writeheader()
        writer.writerows(compact_rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": "supervised_k16_comparison",
        "source": git_identity(),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "resolved_config": str(args.output_dir / "config.yaml"),
        "resolved_config_sha256": sha256_file(args.output_dir / "config.yaml"),
        "data_sha256": {
            "manifest": cfg["data"]["manifest_sha256"],
            "cell_mass": cfg["data"]["cell_mass_sha256"],
            "offcenter_manifest": cfg["offcenter_subgroup"]["manifest_sha256"],
            "fixed_qualitative_manifest": cfg["qualitative"]["fixed_manifest_sha256"],
        },
        "action_exports": action_manifests,
        "resource_receipts": readiness["resource_receipts"],
        "outputs": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "readiness.json",
                "README.md",
                "config.yaml",
                "metrics.json",
                "gate.json",
                "per_seed_checkpoint_metrics.csv",
            )
        },
        "hlvid_executed": False,
        "existing_rl_hlvid_reused_only_if_gate_passes": cfg["existing_rl_hlvid_context"],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output_dir),
                "hlvid_admitted": gate["hlvid_admitted"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
