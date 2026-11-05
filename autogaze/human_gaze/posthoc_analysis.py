"""Post-hoc CPU helpers for fixed-K16 human-gaze and HLVid answer analyses.

These functions only re-read saved actions, cached human cell mass and saved
QA predictions. They complement ``supervised_analysis`` and reuse its
aggregation contract (frames -> clips -> videos -> equal-weight sources).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
from autogaze.human_gaze.supervised_analysis import _validate_cell_mass, action_intersection


def pooled_prior_cells(
    records: Sequence[Mapping], cell_mass: np.ndarray, *, exact_k: int = 16
) -> list[int]:
    """Top-k cells by total training attention mass pooled over all sources."""
    mass = _validate_cell_mass(cell_mass, len(records))
    indices = [i for i, record in enumerate(records) if record["split"] == "train"]
    if not indices:
        raise ValueError("Pooled prior requires training records")
    summed = mass[indices].sum(axis=(0, 1), dtype=np.float64)
    return sorted(range(mass.shape[2]), key=lambda cell: (-summed[cell], cell))[:exact_k]


def overlap_values(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame intersection/K and set Jaccard of two exact-K action arrays."""
    intersection = action_intersection(left, right).astype(np.float64)
    k = left.shape[2]
    return {
        "intersection_over_k": intersection / k,
        "set_jaccard": intersection / (2 * k - intersection),
    }


def report_from_video_values(
    video_values_by_source: Mapping[str, Mapping[str, float]],
    sources: Sequence[str] = STAVIS_SOURCES,
) -> dict[str, Any]:
    """Wrap saved per-video values in the shape used by the paired bootstrap."""
    per_source = {}
    for source in sources:
        values = {str(k): float(v) for k, v in video_values_by_source[source].items()}
        if not values:
            raise ValueError(f"No videos for source {source}")
        per_source[source] = {
            "mean_video_value": float(np.mean(list(values.values()))),
            "num_videos": len(values),
            "video_values": values,
        }
    return {
        "macro_source_mean": float(np.mean([v["mean_video_value"] for v in per_source.values()])),
        "per_source": per_source,
    }


def video_pooled_bootstrap_delta(
    left_reports: Sequence[Mapping],
    right_reports: Sequence[Mapping],
    *,
    iterations: int = 10000,
    seed: int = 20261105,
    sources: Sequence[str] = STAVIS_SOURCES,
) -> dict[str, Any]:
    """Equal-weight-per-video paired difference, seed-averaged.

    Videos are resampled within source (as in the primary bootstrap) but each
    resampled video receives equal weight, so large sources dominate.
    """
    if len(left_reports) != len(right_reports) or not left_reports:
        raise ValueError("Paired reports are required")
    blocks = []
    for source in sources:
        ids = sorted(left_reports[0]["per_source"][source]["video_values"])
        rows = []
        for left, right in zip(left_reports, right_reports):
            lv = left["per_source"][source]["video_values"]
            rv = right["per_source"][source]["video_values"]
            if sorted(lv) != ids or sorted(rv) != ids:
                raise ValueError("Paired reports cover different videos")
            rows.append([float(lv[v]) - float(rv[v]) for v in ids])
        blocks.append(np.asarray(rows, dtype=np.float64).mean(axis=0))
    point = float(np.concatenate(blocks).mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        drawn = [block[rng.integers(0, len(block), size=len(block))] for block in blocks]
        samples[iteration] = np.concatenate(drawn).mean()
    low, high = np.quantile(samples, (0.05, 0.95))
    return {
        "difference": point,
        "ci90_low": float(low),
        "ci90_high": float(high),
        "iterations": iterations,
        "bootstrap_seed": seed,
        "num_videos": int(sum(len(block) for block in blocks)),
        "cluster_unit": "video_resampled_within_source_equal_video_weight_seed_averaged",
        "interpretation": "post_hoc_descriptive_unadjusted",
    }


def pairwise_disagreement(
    answers: Mapping[str, Sequence[Any]], pairs: Sequence[tuple[str, str]]
) -> np.ndarray:
    """Per-question disagreement indicators, shape [pairs, questions]."""
    lengths = {len(answers[name]) for pair in pairs for name in pair}
    if len(lengths) != 1:
        raise ValueError("All answer vectors must cover the same questions")
    return np.asarray(
        [[a != b for a, b in zip(answers[left], answers[right])] for left, right in pairs],
        dtype=np.float64,
    )


def within_pairs(names: Sequence[str]) -> list[tuple[str, str]]:
    return list(combinations(names, 2))


def video_cluster_bootstrap_mean_difference(
    left: np.ndarray,
    right: np.ndarray,
    clusters: Sequence[str],
    *,
    iterations: int = 10000,
    seed: int = 20261105,
) -> dict[str, Any]:
    """Question-micro mean(left) - mean(right) with paired video-cluster bootstrap."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.shape != (len(clusters),):
        raise ValueError("Per-question vectors must align with clusters")
    keys = sorted(set(clusters))
    index = {key: i for i, key in enumerate(keys)}
    cluster_ids = np.asarray([index[c] for c in clusters])
    sums = np.zeros((len(keys), 2))
    np.add.at(sums[:, 0], cluster_ids, left - right)
    np.add.at(sums[:, 1], cluster_ids, 1.0)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations)
    for iteration in range(iterations):
        drawn = sums[rng.integers(0, len(keys), size=len(keys))]
        samples[iteration] = drawn[:, 0].sum() / drawn[:, 1].sum()
    low, high = np.quantile(samples, (0.05, 0.95))
    return {
        "difference": float((left - right).mean()),
        "ci90_low": float(low),
        "ci90_high": float(high),
        "iterations": iterations,
        "bootstrap_seed": seed,
        "num_clusters": len(keys),
        "cluster_unit": "video",
        "interpretation": "post_hoc_descriptive_unadjusted",
    }
