#!/usr/bin/env python3
"""Curate the three-seed R2d HLVid actual-versus-forced comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEEDS = (440826, 440827, 440828)
MODES = ("variable", "forced_k16")
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
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def macro_video(rows: list[dict]) -> float:
    videos = defaultdict(list)
    for row in rows:
        videos[row["video_path"]].append(float(row["is_correct"]))
    return float(np.mean([np.mean(values) for values in videos.values()]))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    summaries = {}
    rows_by_arm = {}
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
            if len(rows) != 268 or len({row["question_id"] for row in rows}) != 268:
                raise ValueError(f"Expected 268 unique HLVid questions under {arm_dir}")
            identity = [(row["question_id"], row["video_path"], row["answer"]) for row in rows]
            if reference is None:
                reference = identity
            elif identity != reference:
                raise ValueError("HLVid arm question order/content mismatch")
            summaries[(seed, mode)] = summary
            rows_by_arm[(seed, mode)] = rows
            artifact_files.extend([summary_path, results_path])

    per_seed = []
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
            stats = summaries[(seed, mode)]["r2d_hlvid_adapter"]["allocation_statistics"]["decoder"]
            mode_metrics[mode] = {
                "num_correct": int(sum(bool(row["is_correct"]) for row in rows)),
                "question_micro_accuracy": float(np.mean([row["is_correct"] for row in rows])),
                "macro_video_accuracy": macro_video(rows),
                "num_videos": len(grouped),
                "mean_decoder_spatial_actions": stats["spatial_actions_per_frame_mean"],
                "decoder_eos_action_rate": stats["eos_action_rate"],
            }
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
        aggregate[mode] = {
            "mean_macro_video_accuracy_over_seeds": float(
                np.mean([row[mode]["macro_video_accuracy"] for row in per_seed])
            ),
            "mean_question_micro_accuracy_over_seeds": float(
                np.mean([row[mode]["question_micro_accuracy"] for row in per_seed])
            ),
            "mean_decoder_spatial_actions_over_seeds": float(
                np.mean([row[mode]["mean_decoder_spatial_actions"] for row in per_seed])
            ),
        }
    deltas = np.array([row["variable_minus_forced_k16_macro_video"] for row in per_seed])
    aggregate["variable_minus_forced_k16"] = {
        "mean_macro_video_accuracy_difference": float(deltas.mean()),
        "paired_video_bootstrap": bootstrap_delta(per_video, video_ids),
        "seed_t95_interval": [
            float(deltas.mean() - 4.3026527299 * deltas.std(ddof=1) / math.sqrt(3)),
            float(deltas.mean() + 4.3026527299 * deltas.std(ddof=1) / math.sqrt(3)),
        ],
        "mean_question_micro_accuracy_difference": float(
            np.mean([row["variable_minus_forced_k16_question_micro"] for row in per_seed])
        ),
    }

    metrics = {
        "run_id": args.run_id,
        "status": "complete_secondary_descriptive",
        "interpretation_boundary": (
            "Separate secondary transfer evidence; does not alter the primary fixed-budget conclusion."
        ),
        "primary_estimand": "equal-seed_mean_macro_video_exact_accuracy",
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
            "decoder_eos_action_rate",
        ])
        writer.writeheader()
        for record in per_seed:
            for mode in MODES:
                writer.writerow({"base_seed": record["base_seed"], "mode": mode, **record[mode]})
    with (args.output_dir / "paired_questions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

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
    figure.savefig(args.output_dir / "actual_vs_forced_k16.png", dpi=180)
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
auditable within-question contrasts.
"""
    (args.output_dir / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
