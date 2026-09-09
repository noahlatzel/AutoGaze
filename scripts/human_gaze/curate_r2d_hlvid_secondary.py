#!/usr/bin/env python3
"""Curate the three-seed R2d HLVid actual-versus-forced comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autogaze.human_gaze.r2d_hlvid import merge_raw_allocations, summarize_raw_allocation
from scripts.runners.hlvid_evidence import (
    load_resume_state,
    stable_example_key,
    stable_video_key,
)

SEEDS = (440826, 440827, 440828)
MODES = ("variable", "forced_k16")
EXPECTED_VIDEOS = 77
ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
EXPECTED_PROTOCOL = {
    "num_examples": 268,
    "frame_source": "uniform",
    "num_video_frames": 128,
    "num_video_frames_thumbnail": 64,
    "thumbnail_sampling": "full",
    "max_tiles_video": 48,
    "tile_len": 16,
    "delta_gap": 0,
    "max_new_tokens": 16,
    "metric": "exact_match_multiple_choice_accuracy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.read_bytes().endswith(b"\n"):
        raise ValueError(f"JSONL is missing its terminal newline: {path}")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_letter(text: str) -> str:
    match = ANSWER_RE.search(text.strip())
    return match.group(1).upper() if match else ""


def macro_video(rows: list[dict]) -> float:
    videos = defaultdict(list)
    for row in rows:
        videos[row["video_path"]].append(float(row["is_correct"]))
    return float(np.mean([np.mean(values) for values in videos.values()]))


def process_counter_scope_complete(adapter: dict) -> bool:
    """Only accept process-exit counters with explicit full QA observation coverage."""

    coverage = adapter.get("observation_coverage") or {}
    return (
        coverage.get("complete") is True
        and int(coverage.get("qa_examples_observed_by_process_counters", -1))
        == EXPECTED_PROTOCOL["num_examples"]
        and int(coverage.get("resume_prefix_examples", -1)) == 0
    )


def load_variable_replay(
    replay_root: Path,
    *,
    seed: int,
    qa_rows: list[dict],
    qa_adapter: dict,
) -> tuple[dict, list[Path]]:
    replay_dir = replay_root / f"seed{seed}_variable"
    summary_path = replay_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Variable allocation is unavailable without a complete validated replay: {summary_path}"
        )
    summary = json.loads(summary_path.read_text())
    expected = {
        "schema_version": 1,
        "base_seed": seed,
        "training_seed": int(qa_adapter["training_seed"]),
        "mode": "variable",
        "record_type": "r2d_hlvid_allocation_replay_summary",
        "status": "complete",
        "nvila_generation_calls": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Variable replay identity/status mismatch for seed {seed}: {mismatches}")
    coverage = summary.get("coverage") or {}
    if coverage != {
        "unique_videos_observed": EXPECTED_VIDEOS,
        "unique_videos_expected": EXPECTED_VIDEOS,
        "question_keys_observed": EXPECTED_PROTOCOL["num_examples"],
        "question_keys_expected": EXPECTED_PROTOCOL["num_examples"],
        "complete": True,
    }:
        raise ValueError(f"Variable replay has incomplete observation coverage for seed {seed}: {coverage}")
    validation = summary.get("validation") or {}
    accepted_validation = {
        "exact_allocation_statistics_match",
        "shared_replay_implementation_validated_by_exact_variable_reference",
    }
    if validation.get("status") != "pass" or validation.get("criterion") not in accepted_validation:
        raise ValueError(f"Variable replay lacks exact uninterrupted-VARIABLE validation for seed {seed}")
    for key in ("model_sha256", "config_sha256"):
        if summary["checkpoint"].get(key) != qa_adapter["checkpoint"].get(key):
            raise ValueError(f"Variable replay checkpoint {key} mismatch for seed {seed}")
    if summary["calibration"].get("sha256") != qa_adapter["eos_calibration"].get("sha256"):
        raise ValueError(f"Variable replay calibration mismatch for seed {seed}")
    protocol = summary.get("protocol") or {}
    protocol_mismatches = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in EXPECTED_PROTOCOL.items()
        if protocol.get(key) != value
    }
    if protocol_mismatches:
        raise ValueError(f"Variable replay protocol mismatch for seed {seed}: {protocol_mismatches}")

    results_path = Path(summary["results_jsonl"])
    evidence_path = Path(summary["evidence_jsonl"])
    for path, hash_key in (
        (results_path, "results_jsonl_sha256"),
        (evidence_path, "evidence_jsonl_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != summary[hash_key]:
            raise ValueError(f"Variable replay artifact/hash mismatch for seed {seed}: {path}")
    replay_rows = read_jsonl(results_path)
    if len(replay_rows) != EXPECTED_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_VIDEOS} replay videos for seed {seed}")
    qa_by_video = defaultdict(list)
    for row in qa_rows:
        qa_by_video[row["video_path"]].append(int(row["question_id"]))
    replay_by_video = {}
    for row in replay_rows:
        video = row["video_path"]
        if video in replay_by_video:
            raise ValueError(f"Duplicate variable replay video for seed {seed}: {video}")
        if row.get("state") != "completed_processor_observation":
            raise ValueError(f"Incomplete variable replay record for seed {seed}: {video}")
        if row.get("compatibility_key") != summary.get("compatibility_key"):
            raise ValueError(f"Variable replay resume identity mismatch for seed {seed}: {video}")
        if row.get("video_key") != stable_video_key(video, "test"):
            raise ValueError(f"Variable replay stable video key mismatch for seed {seed}: {video}")
        expected_question_keys = [stable_example_key(qid, "test") for qid in row["question_ids"]]
        if row.get("question_keys") != expected_question_keys:
            raise ValueError(f"Variable replay stable QA keys mismatch for seed {seed}: {video}")
        if sorted(row["question_ids"]) != sorted(qa_by_video[video]):
            raise ValueError(f"Variable replay QA-key mismatch for seed {seed}: {video}")
        if int(row["question_count"]) != len(qa_by_video[video]):
            raise ValueError(f"Variable replay question weight mismatch for seed {seed}: {video}")
        replay_by_video[video] = row
    if set(replay_by_video) != set(qa_by_video):
        raise ValueError(f"Variable replay video set mismatch for seed {seed}")
    recomputed_raw = merge_raw_allocations(
        (row["allocation_raw"], int(row["question_count"])) for row in replay_rows
    )
    if recomputed_raw != summary["weighted_allocation_raw"]:
        raise ValueError(f"Variable replay aggregate mismatch for seed {seed}")
    if summarize_raw_allocation(recomputed_raw) != summary["weighted_allocation_statistics"]:
        raise ValueError(f"Variable replay summary mismatch for seed {seed}")
    evidence = load_resume_state(evidence_path, summary["compatibility_key"])
    if len(evidence["completed"]) != EXPECTED_PROTOCOL["num_examples"]:
        raise ValueError(f"Variable replay durable QA evidence is incomplete for seed {seed}")
    qa_by_key = {stable_example_key(int(row["question_id"]), "test"): row for row in qa_rows}
    if set(evidence["completed"]) != set(qa_by_key):
        raise ValueError(f"Variable replay durable QA key set mismatch for seed {seed}")
    for example_key, record in evidence["completed"].items():
        qa_row = qa_by_key[example_key]
        if record.get("video_key") != stable_video_key(qa_row["video_path"], "test"):
            raise ValueError(f"Variable replay evidence video key mismatch for seed {seed}")
        if record.get("measurement_origin") != "processor_replay":
            raise ValueError(f"Variable replay measurement origin mismatch for seed {seed}")
        if record.get("answer", {}).get("source_row") != qa_row:
            raise ValueError(f"Variable replay did not preserve the durable QA row for seed {seed}")
        for counter_name in (
            "decoder_spatial_actions",
            "retained_patches",
            "expanded_visual_tokens",
            "expanded_context_tokens",
        ):
            counter = record.get("counters", {}).get(counter_name) or {}
            if counter.get("availability") != "complete" or counter.get("observed_sum") is None:
                raise ValueError(
                    f"Variable replay counter {counter_name} is incomplete for seed {seed}"
                )
    counters = summary.get("counters") or {}
    for counter_name in (
        "decoder_spatial_actions",
        "retained_patches",
        "expanded_visual_tokens",
        "expanded_context_tokens",
    ):
        counter = counters.get(counter_name) or {}
        if (
            counter.get("availability") != "complete"
            or int(counter.get("observed_count", -1)) <= 0
            or counter.get("observed_sum") is None
        ):
            raise ValueError(f"Variable replay summary counter {counter_name} is incomplete for seed {seed}")
    audit = summary.get("protocol_audit") or {}
    audit_dir = Path(audit.get("directory", ""))
    audit_paths = [
        (audit_dir / "summary.json", "summary_sha256"),
        (audit_dir / "protocol_runtime_manifest.json", "manifest_sha256"),
        (audit_dir / "decode_audit.jsonl", "decode_audit_sha256"),
        (audit_dir / "live_runtime_supplement.json", "supplement_sha256"),
    ]
    for path, hash_key in audit_paths:
        if not path.is_file() or sha256_file(path) != audit.get(hash_key):
            raise ValueError(f"Variable replay protocol-audit artifact mismatch for seed {seed}: {path}")
    return summary, [summary_path, results_path, evidence_path, *(path for path, _ in audit_paths)]


def bootstrap_delta(per_video: dict[tuple[int, str], dict[str, float]], video_ids: list[str]) -> dict:
    rng = np.random.default_rng(20260901)
    values = np.empty(10000, dtype=np.float64)
    for index in range(values.size):
        sampled = rng.choice(video_ids, size=len(video_ids), replace=True)
        seed_deltas = []
        for seed in SEEDS:
            seed_deltas.append(
                np.mean(
                    [
                        per_video[(seed, "variable")][video]
                        - per_video[(seed, "forced_k16")][video]
                        for video in sampled
                    ]
                )
            )
        values[index] = np.mean(seed_deltas)
    return {
        "method": "paired_video_cluster_percentile_bootstrap",
        "confidence": 0.90,
        "replicates": int(values.size),
        "seed": 20260901,
        "interval": [float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))],
    }


def seed_t95(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    half_width = 4.3026527299 * array.std(ddof=1) / math.sqrt(len(array))
    return [float(array.mean() - half_width), float(array.mean() + half_width)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--allocation-replay-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    summaries = {}
    rows_by_arm = {}
    allocation_by_arm = {}
    artifact_files = []
    reference = None
    for seed in SEEDS:
        for mode in MODES:
            arm_dir = args.artifact_root / args.run_id / "rollouts" / f"seed{seed}_{mode}"
            summary_path = arm_dir / "summary.json"
            results_path = arm_dir / "results.jsonl"
            if not summary_path.is_file() or not results_path.is_file():
                raise FileNotFoundError(f"Incomplete HLVid arm: {arm_dir}")
            summary = json.loads(summary_path.read_text())
            adapter = summary.get("r2d_hlvid_adapter") or {}
            if adapter.get("mode") != mode or int(adapter.get("base_seed", -1)) != seed:
                raise ValueError(f"Arm identity mismatch under {arm_dir}")
            execution = adapter.get("execution") or {}
            if execution.get("code_commit") != args.code_commit or execution.get("code_dirty") is not False:
                raise ValueError(f"Unfrozen execution provenance under {arm_dir}: {execution}")
            protocol = adapter.get("benchmark_protocol") or {}
            mismatches = {
                key: {"expected": value, "actual": protocol.get(key)}
                for key, value in EXPECTED_PROTOCOL.items()
                if protocol.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Protocol mismatch under {arm_dir}: {mismatches}")
            rows = read_jsonl(results_path)
            if len(rows) != 268 or [int(row["question_id"]) for row in rows] != list(range(268)):
                raise ValueError(f"Expected ordered HLVid question IDs 0--267 under {arm_dir}")
            for row in rows:
                parsed = extract_letter(str(row["prediction"]))
                answer = str(row["answer"]).strip().upper()
                if row.get("prediction_letter") != parsed or bool(row.get("is_correct")) != (
                    parsed == answer
                ):
                    raise ValueError(
                        f"Prediction parser/correctness mismatch under {arm_dir}, "
                        f"question_id={row['question_id']}"
                    )
            identity = [(row["question_id"], row["video_path"], row["answer"]) for row in rows]
            if reference is None:
                reference = identity
            elif identity != reference:
                raise ValueError("HLVid arm question order/content mismatch")
            summaries[(seed, mode)] = summary
            rows_by_arm[(seed, mode)] = rows
            artifact_files.extend([summary_path, results_path])
            process_stats_complete = process_counter_scope_complete(adapter)
            if mode == "variable":
                replay, replay_files = load_variable_replay(
                    args.allocation_replay_root,
                    seed=seed,
                    qa_rows=rows,
                    qa_adapter=adapter,
                )
                artifact_files.extend(replay_files)
                allocation_by_arm[(seed, mode)] = {
                    "source": "validated_question_agnostic_processor_replay",
                    "process_summary_counter_scope_complete": process_stats_complete,
                    "statistics": replay["weighted_allocation_statistics"],
                    "mean_visual_tokens": replay["visual_tokens"]["mean"],
                    "mean_expanded_context_tokens": replay["expanded_context_tokens"]["mean"],
                    "replay_compatibility_key": replay["compatibility_key"],
                }
            else:
                allocation_by_arm[(seed, mode)] = {
                    "source": "coherent_exact_forced_k16_action_contract",
                    "process_summary_counter_scope_complete": process_stats_complete,
                    "statistics": {
                        "decoder": {
                            "spatial_actions_per_frame_mean": 16.0,
                            "eos_action_rate": 0.0,
                        },
                        "post_resolution_adaptation": {
                            "retained_patches_per_frame_mean": 64.0,
                        },
                    },
                    "mean_visual_tokens": None,
                    "mean_expanded_context_tokens": None,
                    "missing_resource_fields_reason": (
                        "The exact-K contract establishes actions and recovered patches, but the legacy "
                        "QA stream did not persist per-question expanded token/context counts."
                    ),
                }

    per_seed = []
    per_category = []
    per_video = {}
    paired_rows = []
    for seed in SEEDS:
        mode_metrics = {}
        for mode in MODES:
            rows = rows_by_arm[(seed, mode)]
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["video_path"]].append(float(row["is_correct"]))
            per_video[(seed, mode)] = {video: float(np.mean(values)) for video, values in grouped.items()}
            allocation = allocation_by_arm[(seed, mode)]
            stats = allocation["statistics"]["decoder"]
            post_stats = allocation["statistics"]["post_resolution_adaptation"]
            mode_metrics[mode] = {
                "num_correct": int(sum(bool(row["is_correct"]) for row in rows)),
                "question_micro_accuracy": float(np.mean([row["is_correct"] for row in rows])),
                "macro_video_accuracy": macro_video(rows),
                "num_videos": len(grouped),
                "mean_decoder_spatial_actions": stats["spatial_actions_per_frame_mean"],
                "decoder_eos_action_rate": stats["eos_action_rate"],
                "mean_retained_patches": post_stats["retained_patches_per_frame_mean"],
                "mean_visual_tokens": allocation["mean_visual_tokens"],
                "mean_expanded_context_tokens": allocation["mean_expanded_context_tokens"],
                "allocation_evidence_source": allocation["source"],
                "legacy_process_counter_scope_complete": allocation[
                    "process_summary_counter_scope_complete"
                ],
            }
            for category in sorted({row["category"] for row in rows}):
                category_rows = [row for row in rows if row["category"] == category]
                per_category.append(
                    {
                        "base_seed": seed,
                        "mode": mode,
                        "category": category,
                        "num_examples": len(category_rows),
                        "num_correct": int(sum(bool(row["is_correct"]) for row in category_rows)),
                        "question_micro_accuracy": float(
                            np.mean([row["is_correct"] for row in category_rows])
                        ),
                    }
                )
        per_seed.append(
            {
                "base_seed": seed,
                **mode_metrics,
                "variable_minus_forced_k16_macro_video": (
                    mode_metrics["variable"]["macro_video_accuracy"]
                    - mode_metrics["forced_k16"]["macro_video_accuracy"]
                ),
                "variable_minus_forced_k16_question_micro": (
                    mode_metrics["variable"]["question_micro_accuracy"]
                    - mode_metrics["forced_k16"]["question_micro_accuracy"]
                ),
            }
        )
        for variable, forced in zip(rows_by_arm[(seed, "variable")], rows_by_arm[(seed, "forced_k16")]):
            paired_rows.append(
                {
                    "base_seed": seed,
                    "question_id": variable["question_id"],
                    "video_path": variable["video_path"],
                    "variable_correct": int(variable["is_correct"]),
                    "forced_k16_correct": int(forced["is_correct"]),
                    "correctness_delta": int(variable["is_correct"]) - int(forced["is_correct"]),
                    "variable_prediction": variable["prediction_letter"],
                    "forced_k16_prediction": forced["prediction_letter"],
                }
            )

    video_ids = sorted(per_video[(SEEDS[0], MODES[0])])
    if any(sorted(per_video[key]) != video_ids for key in per_video):
        raise ValueError("Video clusters differ across arms")
    aggregate = {}
    for mode in MODES:
        visual_tokens = [row[mode]["mean_visual_tokens"] for row in per_seed]
        expanded_contexts = [row[mode]["mean_expanded_context_tokens"] for row in per_seed]
        macro_values = [row[mode]["macro_video_accuracy"] for row in per_seed]
        micro_values = [row[mode]["question_micro_accuracy"] for row in per_seed]
        aggregate[mode] = {
            "mean_macro_video_accuracy_over_seeds": float(
                np.mean(macro_values)
            ),
            "macro_video_accuracy_seed_sd": float(np.std(macro_values, ddof=1)),
            "macro_video_accuracy_seed_t95_interval": seed_t95(macro_values),
            "mean_question_micro_accuracy_over_seeds": float(
                np.mean(micro_values)
            ),
            "question_micro_accuracy_seed_sd": float(np.std(micro_values, ddof=1)),
            "question_micro_accuracy_seed_t95_interval": seed_t95(micro_values),
            "mean_decoder_spatial_actions_over_seeds": float(
                np.mean([row[mode]["mean_decoder_spatial_actions"] for row in per_seed])
            ),
            "mean_retained_patches_over_seeds": float(
                np.mean([row[mode]["mean_retained_patches"] for row in per_seed])
            ),
            "mean_visual_tokens_over_seeds": (
                float(np.mean(visual_tokens)) if all(value is not None for value in visual_tokens) else None
            ),
            "mean_expanded_context_tokens_over_seeds": (
                float(np.mean(expanded_contexts))
                if all(value is not None for value in expanded_contexts)
                else None
            ),
        }
    deltas = np.array([row["variable_minus_forced_k16_macro_video"] for row in per_seed])
    aggregate["variable_minus_forced_k16"] = {
        "mean_macro_video_accuracy_difference": float(deltas.mean()),
        "paired_video_bootstrap": bootstrap_delta(per_video, video_ids),
        "seed_t95_interval": seed_t95(deltas.tolist()),
        "mean_question_micro_accuracy_difference": float(
            np.mean([row["variable_minus_forced_k16_question_micro"] for row in per_seed])
        ),
        "mean_decoder_spatial_actions_difference": float(
            aggregate["variable"]["mean_decoder_spatial_actions_over_seeds"] - 16.0
        ),
    }

    metrics = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "complete_secondary_descriptive",
        "benchmark": "HLVid official test",
        "human_gaze_protected_test_accessed": False,
        "interpretation_boundary": (
            "Separate secondary deployment evidence; does not alter the primary fixed-budget conclusion "
            "and does not isolate allocation from changed later spatial ordering."
        ),
        "primary_estimand": "equal-seed_mean_macro_video_exact_accuracy",
        "secondary_estimand": "equal-seed_mean_question_micro_exact_accuracy",
        "eos_calibration_transfer": [
            {
                "base_seed": seed,
                "training_split": summaries[(seed, "variable")]["r2d_hlvid_adapter"][
                    "eos_calibration"
                ]["split"],
                "target_mean_spatial_actions": summaries[(seed, "variable")][
                    "r2d_hlvid_adapter"
                ]["eos_calibration"]["target_mean_tokens"],
                "calibration_mean_spatial_actions": summaries[(seed, "variable")][
                    "r2d_hlvid_adapter"
                ]["eos_calibration"]["selected_mean_tokens"],
                "hlvid_mean_spatial_actions": next(
                    row["variable"]["mean_decoder_spatial_actions"]
                    for row in per_seed
                    if row["base_seed"] == seed
                ),
            }
            for seed in SEEDS
        ],
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    shutil.copyfile(args.config, args.output_dir / "config.yaml")
    with (args.output_dir / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "base_seed", "mode", "num_correct", "question_micro_accuracy",
            "macro_video_accuracy", "num_videos", "mean_decoder_spatial_actions",
            "decoder_eos_action_rate", "mean_retained_patches", "mean_visual_tokens",
            "mean_expanded_context_tokens", "allocation_evidence_source",
            "legacy_process_counter_scope_complete",
        ])
        writer.writeheader()
        for record in per_seed:
            for mode in MODES:
                writer.writerow({"base_seed": record["base_seed"], "mode": mode, **record[mode]})
    with (args.output_dir / "paired_questions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    with (args.output_dir / "per_category.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_category[0]))
        writer.writeheader()
        writer.writerows(per_category)
    with (args.output_dir / "paired_videos.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "base_seed",
                "video_path",
                "variable_macro_accuracy",
                "forced_k16_macro_accuracy",
                "accuracy_delta",
            ],
        )
        writer.writeheader()
        for seed in SEEDS:
            for video in video_ids:
                variable = per_video[(seed, "variable")][video]
                forced = per_video[(seed, "forced_k16")][video]
                writer.writerow(
                    {
                        "base_seed": seed,
                        "video_path": video,
                        "variable_macro_accuracy": variable,
                        "forced_k16_macro_accuracy": forced,
                        "accuracy_delta": variable - forced,
                    }
                )

    figure, accuracy_axis = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(SEEDS))
    accuracy_axis.plot(x, [row["variable"]["macro_video_accuracy"] for row in per_seed], "o-", label="actual variable EOS")
    accuracy_axis.plot(x, [row["forced_k16"]["macro_video_accuracy"] for row in per_seed], "s--", label="coherent forced K16")
    accuracy_axis.set_xticks(x, [str(seed) for seed in SEEDS])
    accuracy_axis.set_xlabel("R2d base seed")
    accuracy_axis.set_ylabel("HLVid macro-video exact accuracy")
    accuracy_axis.grid(axis="y", alpha=0.25)
    allocation_axis = accuracy_axis.twinx()
    allocation_axis.bar(x, [row["variable"]["mean_decoder_spatial_actions"] for row in per_seed], width=0.18, alpha=0.25, color="tab:green", label="actual mean K")
    allocation_axis.axhline(16, color="tab:green", linewidth=1, linestyle=":")
    allocation_axis.set_ylabel("actual decoder spatial actions/frame")
    handles1, labels1 = accuracy_axis.get_legend_handles_labels()
    handles2, labels2 = allocation_axis.get_legend_handles_labels()
    accuracy_axis.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "actual_vs_forced_k16.png", dpi=300)
    figure.savefig(args.output_dir / "actual_vs_forced_k16.pdf")
    plt.close(figure)

    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "code_commit": args.code_commit,
        "heavy_artifact_root": str((args.artifact_root / args.run_id).resolve()),
        "benchmark_protocol": EXPECTED_PROTOCOL,
        "artifact_inputs": {
            str(path.resolve()): sha256_file(path) for path in artifact_files
        },
        "allocation_evidence": {
            f"seed{seed}_{mode}": allocation_by_arm[(seed, mode)]
            for seed in SEEDS
            for mode in MODES
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = f"""# R2f R2d HLVid secondary evaluation

This bundle compares each completed R2d checkpoint under its actual calibrated
variable-EOS policy and its coherent exact forced-K16 rollout. It uses the
established HLVid protocol: 128 uniform video frames, 64 full thumbnails,
`max_tiles_video=48`, 16-frame tiles, all 268 test questions, and exact answer
letter scoring. This is secondary evidence and is separate from the primary
fixed-budget comparison.

Primary descriptive estimand: equal-seed mean of macro-video exact accuracy.
Question-micro accuracy is also retained. `metrics.json` contains paired
video-bootstrap and three-seed intervals; `paired_questions.csv` preserves the
auditable within-question contrasts, while `paired_videos.csv` is the source
table for clustered comparisons.

The variable allocation/resource fields come from a validated processor-only
replay over the same source videos; the replay makes zero NVILA generation
calls and reproduces an uninterrupted VARIABLE preflight exactly. Legacy
process-exit counters are not accepted unless their observation coverage is
explicitly complete. Forced-K16 action and recovered-patch lengths are nominal
from the coherent exact-K contract; unavailable legacy per-question context
fields remain null rather than being inferred.

This is a deployment-policy contrast, not an isolated allocation experiment:
stopping at EOS also changes which later spatial actions are emitted. The
earlier R2d validation analysis isolates allocation only by holding the common
forced-K36 spatial ordering fixed and comparing actual against shuffled
lengths. Its small allocation signal does not override the observed in-domain
coverage degradation or require a positive HLVid story.
"""
    (args.output_dir / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
