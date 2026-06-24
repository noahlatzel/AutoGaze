# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast in-process dry proxy sweep for HLVid temporal sampling."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from autogaze.temporal_sampling import (
    Dinov2SemanticEncoder,
    SEMANTIC_PAIRWISE_METRICS,
    SEMANTIC_WINDOW_METRICS,
    SiglipSemanticEncoder,
    score_semantic_pairwise_metric,
    score_semantic_window_metric,
    score_window_metric,
    select_best_action,
    thumbnail_tensor,
)
from hlvid_common import load_rows, load_video_samples, parse_csv_ints


DEFAULT_MODEL_ROOT = "/storage/slurm/latn/data/AutoGaze/hf_models"
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
    parser.add_argument("--dataset-root", default="/storage/slurm/latn/data/AutoGaze/HLVid")
    parser.add_argument("--output-root", default="outputs/hlvid/proxy_dry_sweeps/full_128_log_fast")
    parser.add_argument("--sampling", default="128")
    parser.add_argument("--tile-len", type=int, default=16)
    parser.add_argument("--deltas", default=DEFAULT_DELTAS)
    parser.add_argument("--lambdas", nargs="+", default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--temporal-cost", default="log", choices=["none", "linear", "log"])
    parser.add_argument("--proxy-device", default="cuda")
    parser.add_argument("--proxy-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--max-batch-size-semantic", type=int, default=8)
    parser.add_argument("--siglip-model", default=f"{DEFAULT_MODEL_ROOT}/google--siglip2-base-patch16-224")
    parser.add_argument("--dinov2-model", default=f"{DEFAULT_MODEL_ROOT}/facebook--dinov2-with-registers-base")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


class FastProxySweeper:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dtype = dtype_from_name(args.proxy_dtype)
        self.siglip_encoder: Optional[SiglipSemanticEncoder] = None
        self.dinov2_encoder: Optional[Dinov2SemanticEncoder] = None

    def score(
        self,
        config: dict,
        current_start: int,
        candidate_start: int,
        frames: np.ndarray,
        caches: dict,
        question: str,
    ) -> float:
        metric = config["proxy_metric"]

        def get_tile(start: int) -> np.ndarray:
            if start not in caches["tiles"]:
                caches["tiles"][start] = frames[start : start + self.args.tile_len]
            return caches["tiles"][start]

        def get_resized(start: int) -> torch.Tensor:
            if start not in caches["resized"]:
                caches["resized"][start] = thumbnail_tensor(get_tile(start), self.args.resize)
            return caches["resized"][start]

        def get_semantic(model_name: str, start: int) -> torch.Tensor:
            if start not in caches["semantic"][model_name]:
                encoder = self.get_semantic_encoder(model_name)
                caches["semantic"][model_name][start] = encoder.encode_frames(get_tile(start))
            return caches["semantic"][model_name][start]

        def get_question_embedding() -> torch.Tensor:
            if question not in caches["text"]:
                caches["text"][question] = self.get_semantic_encoder("siglip").encode_text(question)
            return caches["text"][question]

        if metric in ("pixel_mse", "ssim_distance", "edge_l1"):
            return float(
                score_window_metric(
                    get_resized(current_start),
                    get_resized(candidate_start),
                    "mean_pairwise_change",
                    base_metric=metric,
                ).item()
            )

        if metric in ("mean_novelty", "within_diversity", "mean_pairwise_change"):
            return float(
                score_window_metric(
                    get_resized(current_start),
                    get_resized(candidate_start),
                    metric,
                    base_metric=config.get("proxy_base_metric", "pixel_mse"),
                ).item()
            )

        if metric in SEMANTIC_PAIRWISE_METRICS:
            model_name = semantic_model_for_config(config)
            question_embedding = get_question_embedding() if metric == "siglip_qgain" else None
            return float(
                score_semantic_pairwise_metric(
                    get_semantic(model_name, current_start),
                    get_semantic(model_name, candidate_start),
                    metric,
                    question_embedding=question_embedding,
                ).item()
            )

        if metric in SEMANTIC_WINDOW_METRICS:
            model_name = semantic_model_for_config(config)
            question_embedding = get_question_embedding() if metric == "question_relevance" else None
            return float(
                score_semantic_window_metric(
                    get_semantic(model_name, current_start),
                    get_semantic(model_name, candidate_start),
                    metric,
                    question_embedding=question_embedding,
                ).item()
            )

        raise ValueError(f"Unsupported metric for fast dry sweep: {metric}")

    def get_semantic_encoder(self, model_name: str):
        if model_name == "siglip":
            if self.siglip_encoder is None:
                self.siglip_encoder = SiglipSemanticEncoder(
                    self.args.siglip_model,
                    self.args.proxy_device,
                    self.dtype,
                    self.args.max_batch_size_semantic,
                )
            return self.siglip_encoder
        if model_name == "dinov2":
            if self.dinov2_encoder is None:
                self.dinov2_encoder = Dinov2SemanticEncoder(
                    self.args.dinov2_model,
                    self.args.proxy_device,
                    self.dtype,
                    self.args.max_batch_size_semantic,
                )
            return self.dinov2_encoder
        raise ValueError(f"Unknown semantic model: {model_name}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.dataset_root))
    if args.limit is not None:
        rows = rows[: args.limit]

    deltas = parse_csv_ints(args.deltas)
    lambdas = [float(value) for value in args.lambdas]
    sweeper = FastProxySweeper(args)
    aggregates = {run_key(config, lambda_cost, args.temporal_cost): make_aggregate(config, lambda_cost) for config in SWEEP_CONFIGS for lambda_cost in lambdas}

    for row in tqdm(rows, desc="HLVid fast dry sweep"):
        video_path = Path(args.dataset_root) / "videos" / row["video_path"]
        frames, _, _ = load_video_samples(video_path, args.sampling, args.tile_len, None)
        caches = {
            "tiles": {},
            "resized": {},
            "semantic": {"siglip": {}, "dinov2": {}},
            "text": {},
            "scores": {},
        }
        for config in SWEEP_CONFIGS:
            for lambda_cost in lambdas:
                key = run_key(config, lambda_cost, args.temporal_cost)
                starts, decisions, seconds = select_starts(
                    frames,
                    config,
                    lambda_cost,
                    deltas,
                    sweeper,
                    caches,
                    row["question"],
                    args.temporal_cost,
                )
                update_aggregate(aggregates[key], starts, decisions, seconds, args.tile_len)

        write_outputs(output_root, aggregates, args, rows_processed=sum(value["num_examples"] for value in aggregates.values()) // len(aggregates))

    write_outputs(output_root, aggregates, args, rows_processed=len(rows))


def select_starts(
    frames: np.ndarray,
    config: dict,
    lambda_cost: float,
    deltas: list[int],
    sweeper: FastProxySweeper,
    caches: dict,
    question: str,
    temporal_cost: str,
) -> tuple[list[int], list[dict], float]:
    starts = [0] if len(frames) >= sweeper.args.tile_len else []
    decisions = []
    proxy_seconds = 0.0

    while starts:
        current_start = starts[-1]
        base_next_start = current_start + sweeper.args.tile_len
        valid = [delta for delta in deltas if base_next_start + delta + sweeper.args.tile_len <= len(frames)]
        if not valid:
            break

        scores = {}
        started = time.perf_counter()
        for delta in valid:
            candidate_start = base_next_start + delta
            score_key = (config["name"], current_start, candidate_start)
            if score_key not in caches["scores"]:
                caches["scores"][score_key] = sweeper.score(config, current_start, candidate_start, frames, caches, question)
            scores[int(delta)] = caches["scores"][score_key]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        proxy_seconds += time.perf_counter() - started

        selected_delta, utilities = select_best_action(scores, lambda_cost=lambda_cost, cost=temporal_cost)
        next_start = base_next_start + selected_delta
        if next_start + sweeper.args.tile_len > len(frames):
            break
        decisions.append(
            {
                "selected_delta": selected_delta,
                "num_valid_deltas": len(valid),
                "scores": scores,
                "utilities": utilities,
            }
        )
        starts.append(next_start)

    return starts, decisions, proxy_seconds


def update_aggregate(aggregate: dict, starts: list[int], decisions: list[dict], seconds: float, tile_len: int) -> None:
    aggregate["num_examples"] += 1
    aggregate["total_visited_tiles"] += len(starts)
    aggregate["total_visited_frames"] += len(starts) * tile_len
    aggregate["total_proxy_seconds"] += seconds
    aggregate["total_candidate_scores"] += sum(len(decision["scores"]) for decision in decisions)
    aggregate["selected_delta_counts"].update(str(decision["selected_delta"]) for decision in decisions)


def write_outputs(output_root: Path, aggregates: dict, args: argparse.Namespace, rows_processed: int) -> None:
    summary_path = output_root / "sweep_summary.jsonl"
    tmp_path = output_root / "sweep_summary.jsonl.tmp"
    with tmp_path.open("w") as summary_file:
        for key, aggregate in aggregates.items():
            row = finalize_aggregate(key, aggregate, args, rows_processed)
            run_dir = output_root / key
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
            summary_file.write(json.dumps(row) + "\n")
    tmp_path.replace(summary_path)


def finalize_aggregate(key: str, aggregate: dict, args: argparse.Namespace, rows_processed: int) -> dict:
    num_examples = aggregate["num_examples"]
    mean_proxy_seconds_per_score = (
        aggregate["total_proxy_seconds"] / aggregate["total_candidate_scores"]
        if aggregate["total_candidate_scores"]
        else None
    )
    return {
        "run_name": key,
        "rows_processed": rows_processed,
        "num_examples": num_examples,
        "proxy_metric": aggregate["proxy_metric"],
        "proxy_base_metric": aggregate.get("proxy_base_metric"),
        "semantic_model": aggregate.get("semantic_model"),
        "lambda_cost": aggregate["lambda_cost"],
        "temporal_cost": args.temporal_cost,
        "sampling": args.sampling,
        "tile_len": args.tile_len,
        "mean_visited_tiles": aggregate["total_visited_tiles"] / max(num_examples, 1),
        "mean_visited_frames": aggregate["total_visited_frames"] / max(num_examples, 1),
        "total_visited_frames": aggregate["total_visited_frames"],
        "mean_proxy_num_candidate_scores": aggregate["total_candidate_scores"] / max(num_examples, 1),
        "mean_proxy_seconds_per_score": mean_proxy_seconds_per_score,
        "selected_delta_counts": dict(aggregate["selected_delta_counts"]),
    }


def make_aggregate(config: dict, lambda_cost: float) -> dict:
    return {
        "num_examples": 0,
        "proxy_metric": config["proxy_metric"],
        "proxy_base_metric": config.get("proxy_base_metric"),
        "semantic_model": config.get("semantic_model"),
        "lambda_cost": lambda_cost,
        "total_visited_tiles": 0,
        "total_visited_frames": 0,
        "total_proxy_seconds": 0.0,
        "total_candidate_scores": 0,
        "selected_delta_counts": Counter(),
    }


def run_key(config: dict, lambda_cost: float, temporal_cost: str) -> str:
    return f"{config['name']}_l{str(lambda_cost).replace('.', 'p')}_{temporal_cost}"


def semantic_model_for_config(config: dict) -> str:
    metric = config["proxy_metric"]
    if metric in ("siglip_cosine", "siglip_qgain", "question_relevance"):
        return "siglip"
    if metric == "dinov2_cosine":
        return "dinov2"
    return config["semantic_model"]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


if __name__ == "__main__":
    main()
