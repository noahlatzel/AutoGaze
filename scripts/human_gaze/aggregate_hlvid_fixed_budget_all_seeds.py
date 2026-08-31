#!/usr/bin/env python3
"""Validate, aggregate, and plot the all-seed fixed-budget HLVid matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

T90 = {2: 2.919985580, 5: 2.015048373}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def accuracy(rows: list[dict]) -> float:
    return float(np.mean([bool(row["is_correct"]) for row in rows]))


def macro_video_accuracy(rows: list[dict]) -> float:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_path"]].append(bool(row["is_correct"]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def seed_interval(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    df = len(values) - 1
    if df not in T90:
        raise ValueError(f"No frozen 90% t critical value for df={df}")
    mean = float(np.mean(values))
    half = T90[df] * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean - half, mean + half


def cluster_bootstrap_difference(
    rows_a: list[list[dict]],
    rows_b: list[list[dict]],
    *,
    iterations: int,
    seed: int,
) -> dict:
    by_video = defaultdict(lambda: [0.0, 0])
    for seed_rows_a, seed_rows_b in zip(rows_a, rows_b, strict=True):
        for row_a, row_b in zip(seed_rows_a, seed_rows_b, strict=True):
            if row_a["question_id"] != row_b["question_id"]:
                raise ValueError("Question ordering mismatch in paired bootstrap")
            key = row_a["video_path"]
            by_video[key][0] += float(row_a["is_correct"]) - float(row_b["is_correct"])
            by_video[key][1] += 1
    videos = sorted(by_video)
    sums = np.asarray([by_video[video][0] for video in videos], dtype=np.float64)
    counts = np.asarray([by_video[video][1] for video in videos], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(videos), size=len(videos))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    point = sums.sum() / counts.sum()
    return {
        "difference": float(point),
        "ci90_low": float(np.quantile(draws, 0.05)),
        "ci90_high": float(np.quantile(draws, 0.95)),
        "cluster_unit": "video",
        "num_video_clusters": len(videos),
        "iterations": iterations,
        "bootstrap_seed": seed,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    raw_root = Path(config["outputs"]["raw_root"])
    protocol = config["protocol"]
    preflight_path = args.output_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "pass":
        raise ValueError(f"Preflight did not pass: {preflight['failures']}")
    expected_checkpoints = {
        (int(row["base_seed"]), int(row["budget"])): row
        for row in preflight["checkpoints"]
    }
    expected_protocol = {
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "tile_len": 16,
        "max_new_tokens": 16,
        "thumbnails": "full",
        "image_processor": "saved slow processor (use_fast=false)",
        "generation": "greedy",
        "torch_dtype": "bfloat16",
        "max_batch_size_autogaze": 16,
        "max_batch_size_siglip": 32,
        "prompt_template": "<video>\n\nQuestion: {benchmark_question_with_choices_and_answer_instruction}",
        "answer_parser": "first standalone A/B/C/D letter",
    }
    runs = {}
    canonical = None
    per_seed = []
    per_category = []
    for seed_row in config["seed_matrix"]:
        seed = int(seed_row["base_seed"])
        for budget_value in seed_row["checkpoints"]:
            budget = int(budget_value)
            run_dir = raw_root / f"fixedk{budget}_seed{seed}"
            results_path = run_dir / "results.jsonl"
            summary_path = run_dir / "summary.json"
            if not results_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"Missing completed run: {run_dir}")
            rows = read_jsonl(results_path)
            summary = json.loads(summary_path.read_text())
            if len(rows) != 268 or len({int(row["question_id"]) for row in rows}) != 268:
                raise ValueError(f"Incomplete or duplicate HLVid rows: {results_path}")
            rows.sort(key=lambda row: int(row["question_id"]))
            identity = [
                (int(row["question_id"]), row["video_path"], row["category"], row["answer"])
                for row in rows
            ]
            if canonical is None:
                canonical = identity
            elif identity != canonical:
                raise ValueError(f"Benchmark identity/order drift: {results_path}")
            if summary.get("dataset_parquet_sha256") != config["dataset"]["parquet_sha256"]:
                raise ValueError(f"Dataset drift: {summary_path}")
            if summary.get("benchmark_split") != "official_huggingface_test":
                raise ValueError(f"Benchmark split drift: {summary_path}")
            if summary.get("human_gaze_split_accessed") is not False:
                raise ValueError(f"Protected split boundary violated: {summary_path}")
            checkpoint = expected_checkpoints.get((seed, budget))
            if checkpoint is None:
                raise ValueError(f"Run was not fingerprinted in preflight: seed={seed}, K={budget}")
            if summary.get("autogaze_checkpoint_sha256") != checkpoint["model_safetensors_sha256"]:
                raise ValueError(f"Checkpoint fingerprint drift: {summary_path}")
            if Path(summary.get("autogaze_model_id", "")).resolve() != Path(checkpoint["path"]).resolve():
                raise ValueError(f"Checkpoint path drift: {summary_path}")
            if summary.get("base_seed") != seed or summary.get("continuation_seed") != int(seed_row["continuation_seed"]):
                raise ValueError(f"Seed identity drift: {summary_path}")
            compatibility = summary.get("checkpoint_compatibility", {})
            if compatibility.get("accepted_runtime_only_keys") != [
                "gazing_model.gaze_decoder.output_token_logit_bias"
            ] or compatibility.get("max_absolute_value") != 0.0:
                raise ValueError(f"Checkpoint compatibility drift: {summary_path}")
            for key, expected in expected_protocol.items():
                observed = summary["protocol"].get(key)
                if observed != expected:
                    raise ValueError(f"Protocol drift in {summary_path}: {key}={observed!r}")
            adapter = summary["fixed_budget_adapter"]
            if adapter["exact_decoder_actions_per_autogaze_frame"] != budget:
                raise ValueError(f"Budget adapter drift: {summary_path}")
            if adapter["allowed_action_ids"] != [69, 264] or adapter["allow_eos"]:
                raise ValueError(f"Action contract drift: {summary_path}")
            call_stats = adapter["post_resolution_adaptation_current_process"]
            if call_stats["model_calls"] <= 0 or call_stats["frame_observations"] <= 0:
                raise ValueError(f"Fixed-budget adapter was not exercised: {summary_path}")
            micro = accuracy(rows)
            macro = macro_video_accuracy(rows)
            if abs(summary["accuracy"] - micro) > 1e-12 or abs(summary["macro_video_accuracy"] - macro) > 1e-12:
                raise ValueError(f"Summary metric mismatch: {summary_path}")
            runs[(seed, budget)] = rows
            per_seed.append(
                {
                    "budget": budget,
                    "base_seed": seed,
                    "continuation_seed": int(seed_row["continuation_seed"]),
                    "num_correct": int(sum(bool(row["is_correct"]) for row in rows)),
                    "question_micro_accuracy": micro,
                    "macro_video_accuracy": macro,
                    "invalid_predictions": int(sum(not row["prediction_letter"] for row in rows)),
                    "checkpoint_sha256": summary["autogaze_checkpoint_sha256"],
                    "results_sha256": sha256(results_path),
                }
            )
            for category in sorted({row["category"] for row in rows}):
                subset = [row for row in rows if row["category"] == category]
                per_category.append(
                    {
                        "budget": budget,
                        "base_seed": seed,
                        "category": category,
                        "num_examples": len(subset),
                        "num_correct": int(sum(bool(row["is_correct"]) for row in subset)),
                        "accuracy": accuracy(subset),
                    }
                )

    budget_summaries = []
    budgets = sorted({row["budget"] for row in per_seed})
    for budget in budgets:
        subset = [row for row in per_seed if row["budget"] == budget]
        micro_values = [row["question_micro_accuracy"] for row in subset]
        macro_values = [row["macro_video_accuracy"] for row in subset]
        micro_ci = seed_interval(micro_values)
        macro_ci = seed_interval(macro_values)
        budget_summaries.append(
            {
                "budget": budget,
                "num_seeds": len(subset),
                "base_seeds": [row["base_seed"] for row in subset],
                "question_micro_accuracy_mean": float(np.mean(micro_values)),
                "question_micro_accuracy_seed_sd": float(np.std(micro_values, ddof=1)),
                "question_micro_accuracy_ci90_low": micro_ci[0],
                "question_micro_accuracy_ci90_high": micro_ci[1],
                "macro_video_accuracy_mean": float(np.mean(macro_values)),
                "macro_video_accuracy_seed_sd": float(np.std(macro_values, ddof=1)),
                "macro_video_accuracy_ci90_low": macro_ci[0],
                "macro_video_accuracy_ci90_high": macro_ci[1],
            }
        )

    matched_seeds = [int(value) for value in config["analysis"]["matched_budget_seeds"]]
    matched_budgets = [
        budget for budget in budgets if all((seed, budget) in runs for seed in matched_seeds)
    ]
    differences = []
    for budget_a, budget_b in combinations(matched_budgets, 2):
        result = cluster_bootstrap_difference(
            [runs[(seed, budget_a)] for seed in matched_seeds],
            [runs[(seed, budget_b)] for seed in matched_seeds],
            iterations=int(config["analysis"]["bootstrap_iterations"]),
            seed=int(config["analysis"]["bootstrap_seed"] + budget_a * 100 + budget_b),
        )
        differences.append({"budget_a": budget_a, "budget_b": budget_b, **result})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "per_seed.csv",
        sorted(per_seed, key=lambda row: (row["budget"], row["base_seed"])),
        list(per_seed[0]),
    )
    write_csv(
        args.output_dir / "per_category.csv",
        sorted(per_category, key=lambda row: (row["budget"], row["base_seed"], row["category"])),
        list(per_category[0]),
    )
    write_csv(
        args.output_dir / "matched_budget_differences.csv",
        differences,
        list(differences[0]) if differences else ["budget_a", "budget_b"],
    )

    metrics = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "HLVid official test",
        "primary_metric": "exact-match question-micro accuracy",
        "protocol_id": protocol["protocol_id"],
        "verified_protocol": expected_protocol,
        "preflight_sha256": sha256(preflight_path),
        "validated_checkpoint_fingerprints": len(runs),
        "human_gaze_protected_test_accessed": False,
        "budget_summaries": budget_summaries,
        "matched_budget_seeds": matched_seeds,
        "matched_budget_differences": differences,
        "upstream_reference": {
            "accuracy": protocol["reference_accuracy"],
            "artifact": protocol["reference_artifact"],
            "artifact_sha256": protocol["reference_artifact_sha256"],
            "comparison_boundary": "descriptive; not budget-matched or seed-matched",
        },
        "k32_status": config["k32_integration"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(matched_seeds)))
    for color, seed in zip(colors, matched_seeds, strict=True):
        points = [row for row in per_seed if row["base_seed"] == seed]
        points.sort(key=lambda row: row["budget"])
        axis.plot(
            [row["budget"] for row in points],
            [100 * row["question_micro_accuracy"] for row in points],
            marker="o",
            linewidth=1.2,
            alpha=0.75,
            color=color,
            label=f"seed {seed}",
        )
    for summary in budget_summaries:
        x = summary["budget"]
        mean = 100 * summary["question_micro_accuracy_mean"]
        low = 100 * summary["question_micro_accuracy_ci90_low"]
        high = 100 * summary["question_micro_accuracy_ci90_high"]
        axis.errorbar(
            [x], [mean], yerr=[[mean - low], [high - mean]], fmt="D", color="black",
            capsize=4, markersize=6, zorder=5,
        )
    axis.axhline(100 * protocol["reference_accuracy"], color="0.4", linestyle="--", linewidth=1.2,
                 label="upstream AutoGaze (descriptive)")
    axis.set_xlabel("Exact fine-action budget K per AutoGaze frame")
    axis.set_ylabel("HLVid exact-match accuracy (%)")
    axis.set_xticks(budgets)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    axis.set_title("Human-gaze AutoGaze transfer under NVILA MTV48")
    figure.savefig(args.output_dir / "hlvid_fixed_budget_all_seeds.png", dpi=220)
    figure.savefig(args.output_dir / "hlvid_fixed_budget_all_seeds.pdf")
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
