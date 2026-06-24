# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run dry proxy-sampling sweeps for HLVid temporal metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET_ROOT = "/storage/slurm/latn/data/AutoGaze/HLVid"
DEFAULT_OUTPUT_ROOT = "outputs/hlvid/proxy_dry_sweeps/full_128_log"
DEFAULT_DELTAS = "1,2,4,8,16,32,64,128,192,256,320"
DEFAULT_LAMBDAS = ("0", "0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1")


SWEEP_CONFIGS = [
    {"name": "mean_novelty_pixel_mse", "proxy_metric": "mean_novelty", "proxy_base_metric": "pixel_mse"},
    {"name": "mean_novelty_edge_l1", "proxy_metric": "mean_novelty", "proxy_base_metric": "edge_l1"},
    {"name": "within_diversity_pixel_mse", "proxy_metric": "within_diversity", "proxy_base_metric": "pixel_mse"},
    {"name": "within_diversity_edge_l1", "proxy_metric": "within_diversity", "proxy_base_metric": "edge_l1"},
    {"name": "mean_pairwise_change_pixel_mse", "proxy_metric": "mean_pairwise_change", "proxy_base_metric": "pixel_mse"},
    {"name": "mean_pairwise_change_edge_l1", "proxy_metric": "mean_pairwise_change", "proxy_base_metric": "edge_l1"},
    {"name": "ssim_distance", "proxy_metric": "ssim_distance"},
    {"name": "siglip_semantic_mean_novelty", "proxy_metric": "semantic_mean_novelty", "semantic_model": "siglip"},
    {"name": "dinov2_semantic_mean_novelty", "proxy_metric": "semantic_mean_novelty", "semantic_model": "dinov2"},
    {"name": "siglip_semantic_diversity", "proxy_metric": "semantic_diversity", "semantic_model": "siglip"},
    {"name": "dinov2_semantic_diversity", "proxy_metric": "semantic_diversity", "semantic_model": "dinov2"},
    {"name": "siglip_cosine", "proxy_metric": "siglip_cosine"},
    {"name": "dinov2_cosine", "proxy_metric": "dinov2_cosine"},
    {"name": "siglip_qgain", "proxy_metric": "siglip_qgain"},
    {"name": "question_relevance", "proxy_metric": "question_relevance"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sampling", default="128")
    parser.add_argument("--tile-len", type=int, default=16)
    parser.add_argument("--deltas", default=DEFAULT_DELTAS)
    parser.add_argument("--lambdas", nargs="+", default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--temporal-cost", default="log", choices=["none", "linear", "log"])
    parser.add_argument("--proxy-device", default="cuda")
    parser.add_argument("--proxy-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-batch-size-semantic", type=int, default=8)
    parser.add_argument("--max-batch-size-siglip", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_summary_path = output_root / "sweep_summary.jsonl"

    for config in SWEEP_CONFIGS:
        for lambda_cost in args.lambdas:
            run_name = f"{config['name']}_l{format_lambda(lambda_cost)}_{args.temporal_cost}"
            run_dir = output_root / run_name
            result_path = run_dir / "results.jsonl"
            summary_path = run_dir / "summary.json"

            if summary_path.exists() and not args.overwrite:
                summary = json.loads(summary_path.read_text())
                append_sweep_summary(sweep_summary_path, run_name, config, lambda_cost, summary, skipped=True)
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/evaluate_hlvid_proxy_sampling.py",
                "--dataset-root",
                args.dataset_root,
                "--output",
                str(result_path),
                "--summary-output",
                str(summary_path),
                "--dry-run-sampling",
                "--sampling",
                args.sampling,
                "--tile-len",
                str(args.tile_len),
                "--selection",
                "proxy",
                "--proxy-metric",
                config["proxy_metric"],
                "--lambda-cost",
                lambda_cost,
                "--temporal-cost",
                args.temporal_cost,
                "--deltas",
                args.deltas,
                "--proxy-device",
                args.proxy_device,
                "--proxy-dtype",
                args.proxy_dtype,
                "--max-batch-size-semantic",
                str(args.max_batch_size_semantic),
                "--max-batch-size-siglip",
                str(args.max_batch_size_siglip),
                "--resume",
            ]
            if "proxy_base_metric" in config:
                cmd.extend(["--proxy-base-metric", config["proxy_base_metric"]])
            if "semantic_model" in config:
                cmd.extend(["--semantic-model", config["semantic_model"]])
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])

            print(f"[dry-sweep] {run_name}", flush=True)
            subprocess.run(cmd, check=True)
            summary = json.loads(summary_path.read_text())
            append_sweep_summary(sweep_summary_path, run_name, config, lambda_cost, summary, skipped=False)


def format_lambda(value: str) -> str:
    return value.replace(".", "p")


def append_sweep_summary(
    path: Path,
    run_name: str,
    config: dict,
    lambda_cost: str,
    summary: dict,
    *,
    skipped: bool,
) -> None:
    row = {
        "run_name": run_name,
        "proxy_metric": config["proxy_metric"],
        "proxy_base_metric": config.get("proxy_base_metric"),
        "semantic_model": config.get("semantic_model"),
        "lambda_cost": float(lambda_cost),
        "temporal_cost": summary.get("temporal_cost"),
        "num_examples": summary.get("num_examples"),
        "mean_visited_frames": summary.get("mean_visited_frames"),
        "total_visited_frames": summary.get("total_visited_frames"),
        "mean_proxy_seconds_per_score": summary.get("timing_count_summary", {})
        .get("proxy_seconds_per_score", {})
        .get("mean"),
        "mean_proxy_num_candidate_scores": summary.get("timing_count_summary", {})
        .get("proxy_num_candidate_scores", {})
        .get("mean"),
        "selected_delta_counts": summary.get("selected_delta_counts", {}),
        "skipped": skipped,
    }
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
