# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure analysis helpers for the matched supervised-versus-RL K16 study.

The functions in this module operate on frozen manifest records, cached human
cell mass and exported exact-K actions.  Model inference stays in the thin
export script so all downstream aggregation and gate decisions are CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
from autogaze.human_gaze.coverage import center_order


ACTION_EXPORT_SCHEMA_VERSION = 1
OFFCENTER_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash regular files by relative path and content, rejecting symlinks."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected a non-symbolic checkpoint directory: {root}")
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Checkpoint directory is empty: {root}")
    digest = hashlib.sha256()
    for candidate in files:
        if candidate.is_symlink():
            raise ValueError("Checkpoint tree contains a symbolic file")
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_cell_mass(cell_mass: np.ndarray, num_records: int) -> np.ndarray:
    mass = np.asarray(cell_mass, dtype=np.float64)
    if mass.ndim != 3 or mass.shape[0] != num_records or mass.shape[2] != 196:
        raise ValueError("cell_mass must have shape [records, frames, 196]")
    if mass.shape[1] <= 0 or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("cell_mass must be finite, nonnegative and contain frames")
    totals = mass.sum(axis=2)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("Every cached human cell-mass frame must sum to one")
    return mass


def cell_mass_for_records(records: Sequence[Mapping], cache: np.ndarray) -> np.ndarray:
    """Materialize cached mass in manifest-record order with strict indices."""
    indices = []
    for record in records:
        index = record.get("cell_mass_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"Invalid cell_mass_index for {record.get('clip_id')}")
        if index < 0 or index >= len(cache):
            raise ValueError(f"Out-of-range cell_mass_index for {record.get('clip_id')}")
        indices.append(index)
    return _validate_cell_mass(np.asarray(cache[indices]), len(records))


def build_offcenter_manifest(
    records: Sequence[Mapping],
    cell_mass: np.ndarray,
    *,
    manifest_sha256: str,
    cell_mass_sha256: str,
    fraction: float = 0.25,
    center_budget: int = 32,
    sources: Sequence[str] = STAVIS_SOURCES,
) -> dict[str, Any]:
    """Freeze the human-only top fraction of validation clips per source."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    if not 1 <= center_budget < 196:
        raise ValueError("center_budget must lie in [1, 195]")
    if len({record.get("clip_id") for record in records}) != len(records):
        raise ValueError("clip_id values must be unique")
    if any(record.get("split") != "val" for record in records):
        raise ValueError("The off-center subgroup must be frozen from validation only")

    mass = _validate_cell_mass(cell_mass, len(records))
    center_cells = center_order(14)[:center_budget].tolist()
    outside_cells = sorted(set(range(196)) - set(center_cells))
    outside_per_frame = mass[:, :, outside_cells].sum(axis=2, dtype=np.float64)
    scores = outside_per_frame.mean(axis=1, dtype=np.float64)

    by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
    for validation_index, (record, score, frame_scores) in enumerate(
        zip(records, scores, outside_per_frame)
    ):
        source = str(record["source"])
        if source not in by_source:
            raise ValueError(f"Unexpected validation source: {source}")
        by_source[source].append(
            {
                "validation_index": validation_index,
                "cell_mass_index": int(record["cell_mass_index"]),
                "clip_id": str(record["clip_id"]),
                "source": source,
                "video_id": str(record["video_id"]),
                "mean_human_mass_outside_center32": float(score),
                "human_mass_outside_center32_per_frame": [
                    float(value) for value in frame_scores
                ],
            }
        )

    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for source in sources:
        candidates = sorted(
            by_source[source],
            key=lambda row: (
                -row["mean_human_mass_outside_center32"],
                row["clip_id"],
            ),
        )
        if not candidates:
            raise ValueError(f"Validation source has no clips: {source}")
        count = math.ceil(fraction * len(candidates))
        source_selected = []
        for rank, row in enumerate(candidates[:count], start=1):
            source_selected.append({**row, "source_rank": rank})
        selected.extend(source_selected)
        counts[source] = {
            "eligible_clips": len(source_selected),
            "eligible_videos": len({row["video_id"] for row in source_selected}),
            "validation_clips": len(candidates),
            "validation_videos": len({row["video_id"] for row in candidates}),
        }

    selected = sorted(selected, key=lambda row: (sources.index(row["source"]), row["source_rank"]))
    selection_sha256 = canonical_sha256(selected)
    return {
        "schema_version": OFFCENTER_SCHEMA_VERSION,
        "status": "frozen_before_supervised_model_outputs",
        "experiment_id": "supervised_k16_comparison",
        "split": "val",
        "model_outputs_consulted": False,
        "selection": {
            "signal": "cached_human_annotation_cell_mass_only",
            "ranking": "mean_human_mass_outside_center32_descending_within_source",
            "fraction_per_source": fraction,
            "count_rule": "ceil(fraction * validation_clips_in_source)",
            "tie_breaker": "clip_id_ascending",
            "center_definition": "32 cells nearest the 14x14 grid center using center_order",
            "center_cells": center_cells,
            "aggregation": "eligible_clips_to_video_means_to_equal_source_means",
        },
        "inputs": {
            "manifest_sha256": manifest_sha256,
            "cell_mass_sha256": cell_mass_sha256,
        },
        "counts": {
            "total_eligible_clips": len(selected),
            "total_eligible_videos": len(
                {(row["source"], row["video_id"]) for row in selected}
            ),
            "per_source": counts,
        },
        "selection_sha256": selection_sha256,
        "clips": selected,
    }


def validate_action_array(
    actions: np.ndarray,
    *,
    num_records: int,
    clip_len: int = 16,
    exact_k: int = 16,
    num_cells: int = 196,
) -> np.ndarray:
    array = np.asarray(actions)
    if array.shape != (num_records, clip_len, exact_k):
        raise ValueError(
            f"Expected actions shape {(num_records, clip_len, exact_k)}, got {array.shape}"
        )
    if array.dtype.kind not in "iu":
        raise ValueError("Actions must be integer fine-cell indices")
    array = array.astype(np.int64, copy=False)
    if np.any(array < 0) or np.any(array >= num_cells):
        raise ValueError("Action export contains an out-of-range fine cell")
    if np.any(np.sort(array, axis=2)[:, :, 1:] == np.sort(array, axis=2)[:, :, :-1]):
        raise ValueError("Action export repeats a fine cell within a frame")
    return array


def load_action_export(
    directory: Path,
    records: Sequence[Mapping],
    *,
    expected_manifest_sha256: str,
    expected_cell_mass_sha256: str,
    expected_method: str | None = None,
    expected_base_seed: int | None = None,
    expected_training_seed: int | None = None,
    expected_cumulative_update: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Load a complete exact-action export and reject drift or partial files."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != ACTION_EXPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported action-export schema")
    if manifest.get("status") != "complete":
        raise ValueError("Action export is not marked complete")
    expected = {
        "method": expected_method,
        "base_seed": expected_base_seed,
        "training_seed": expected_training_seed,
        "cumulative_update": expected_cumulative_update,
    }
    for key, value in expected.items():
        if value is not None and manifest.get(key) != value:
            raise ValueError(f"Action export mismatch for {key}")
    inputs = manifest.get("inputs", {})
    if inputs.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Action export uses a different STAViS manifest")
    if inputs.get("cell_mass_sha256") != expected_cell_mass_sha256:
        raise ValueError("Action export uses a different human cell-mass cache")
    frozen_contract = {
        "split": "val",
        "clip_len": 16,
        "exact_k": 16,
        "num_fine_cells": 196,
        "fine_action_offset": 69,
        "allowed_local_action_ids_inclusive": [69, 264],
        "greedy": True,
        "selection_without_replacement": True,
    }
    for key, value in frozen_contract.items():
        if manifest.get(key) != value:
            raise ValueError(f"Action export violates the frozen {key} contract")

    actions_path = root / "actions.jsonl"
    if sha256_file(actions_path) != manifest.get("actions_sha256"):
        raise ValueError("Action export checksum mismatch")
    rows = []
    with actions_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if raw_line.strip():
                try:
                    rows.append(json.loads(raw_line))
                except ValueError as error:
                    raise ValueError(f"Invalid action JSON at line {line_number}") from error
    if len(rows) != len(records):
        raise ValueError("Action export does not cover every validation clip exactly once")
    action_values = []
    for index, (row, record) in enumerate(zip(rows, records)):
        for key in ("clip_id", "source", "video_id"):
            if row.get(key) != record.get(key):
                raise ValueError(f"Action export record {index} differs for {key}")
        if row.get("frame_numbers") != record.get("frame_numbers"):
            raise ValueError(f"Action export record {index} has different frame geometry")
        action_values.append(row.get("fine_cells"))
    actions = validate_action_array(
        np.asarray(action_values),
        num_records=len(records),
        clip_len=16,
        exact_k=16,
        num_cells=196,
    )
    if manifest.get("num_clips") != len(records):
        raise ValueError("Action-export manifest clip count is inconsistent")
    return manifest, actions


def within_source_different_video_indices(records: Sequence[Mapping]) -> list[int]:
    """Pair clips to stable same-source, different-video content where possible."""
    source_videos: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        source_videos[str(record["source"])][str(record["video_id"])].append(index)
    shuffled: list[int | None] = [None] * len(records)
    for source, videos in source_videos.items():
        video_ids = sorted(videos)
        if len(video_ids) == 1:
            raise ValueError(
                f"Different-video shuffle requires at least two videos for source {source}"
            )
        for video_position, video_id in enumerate(video_ids):
            target = videos[video_ids[(video_position + 1) % len(video_ids)]]
            for clip_position, index in enumerate(videos[video_id]):
                shuffled[index] = target[clip_position % len(target)]
    if any(index is None for index in shuffled):
        raise AssertionError("Failed to construct the deterministic content shuffle")
    return [int(index) for index in shuffled]


def aggregate_frame_values(
    records: Sequence[Mapping],
    frame_values: np.ndarray,
    *,
    eligible_clip_ids: set[str] | None = None,
    sources: Sequence[str] = STAVIS_SOURCES,
) -> dict[str, Any]:
    """Aggregate frames -> clips -> videos -> equal-weight sources."""
    values = np.asarray(frame_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(records):
        raise ValueError("frame_values must have shape [records, frames]")
    if not np.isfinite(values).all():
        raise ValueError("frame_values must be finite")
    selected_indices = [
        index
        for index, record in enumerate(records)
        if eligible_clip_ids is None or str(record["clip_id"]) in eligible_clip_ids
    ]
    if eligible_clip_ids is not None:
        observed = {str(records[index]["clip_id"]) for index in selected_indices}
        if observed != eligible_clip_ids:
            raise ValueError("Eligible clip IDs are missing from the validation manifest")
    if not selected_indices:
        raise ValueError("Aggregation subset is empty")

    per_source: dict[str, Any] = {}
    source_means = []
    for source in sources:
        videos: dict[str, list[float]] = defaultdict(list)
        frame_count = 0
        clip_count = 0
        for index in selected_indices:
            record = records[index]
            if record["source"] != source:
                continue
            videos[str(record["video_id"])].append(float(values[index].mean()))
            frame_count += values.shape[1]
            clip_count += 1
        if not videos:
            raise ValueError(f"Aggregation subset lacks source {source}")
        video_means = {
            video_id: float(np.mean(clip_values))
            for video_id, clip_values in sorted(videos.items())
        }
        source_mean = float(np.mean(list(video_means.values())))
        source_means.append(source_mean)
        per_source[source] = {
            "mean_video_value": source_mean,
            "num_videos": len(video_means),
            "num_clips": clip_count,
            "num_frames": frame_count,
            "video_values": video_means,
        }
    subset_values = values[selected_indices]
    return {
        "macro_source_mean": float(np.mean(source_means)),
        "pooled_frame_mean": float(subset_values.mean()),
        "num_sources": len(sources),
        "num_videos": sum(value["num_videos"] for value in per_source.values()),
        "num_clips": len(selected_indices),
        "num_frames": int(subset_values.size),
        "per_source": per_source,
    }


def selected_mass(cell_mass: np.ndarray, actions: np.ndarray) -> np.ndarray:
    mass = _validate_cell_mass(cell_mass, len(cell_mass))
    action_array = validate_action_array(
        actions,
        num_records=len(mass),
        clip_len=mass.shape[1],
        exact_k=actions.shape[2],
        num_cells=mass.shape[2],
    )
    return np.take_along_axis(mass, action_array, axis=2).sum(axis=2)


def uniform_density_saliency(
    cell_mass: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return frame-wise SIM and CC for a uniform density on selected cells."""
    mass = _validate_cell_mass(cell_mass, len(cell_mass))
    action_array = validate_action_array(
        actions,
        num_records=len(mass),
        clip_len=mass.shape[1],
        exact_k=actions.shape[2],
        num_cells=mass.shape[2],
    )
    predicted = np.zeros_like(mass, dtype=np.float64)
    np.put_along_axis(
        predicted,
        action_array,
        np.full(action_array.shape, 1.0 / action_array.shape[2], dtype=np.float64),
        axis=2,
    )
    sim = np.minimum(predicted, mass).sum(axis=2)
    pred_centered = predicted - predicted.mean(axis=2, keepdims=True)
    mass_centered = mass - mass.mean(axis=2, keepdims=True)
    denominator = np.sqrt(
        np.square(pred_centered).sum(axis=2) * np.square(mass_centered).sum(axis=2)
    )
    if np.any(denominator <= 0):
        raise ValueError("CC is undefined for a uniform human or prediction density")
    cc = (pred_centered * mass_centered).sum(axis=2) / denominator
    return sim, cc


def action_intersection(actions_left: np.ndarray, actions_right: np.ndarray) -> np.ndarray:
    if actions_left.shape != actions_right.shape:
        raise ValueError("Agreement requires identically shaped action arrays")
    left = validate_action_array(
        actions_left,
        num_records=actions_left.shape[0],
        clip_len=actions_left.shape[1],
        exact_k=actions_left.shape[2],
    )
    right = validate_action_array(
        actions_right,
        num_records=actions_right.shape[0],
        clip_len=actions_right.shape[1],
        exact_k=actions_right.shape[2],
    )
    return (left[:, :, :, None] == right[:, :, None, :]).any(axis=3).sum(axis=2)


def agreement_report(
    records: Sequence[Mapping],
    actions_left: np.ndarray,
    actions_right: np.ndarray,
    *,
    offcenter_clip_ids: set[str],
) -> dict[str, Any]:
    intersection = action_intersection(actions_left, actions_right).astype(np.float64)
    exact_k = actions_left.shape[2]
    values = {
        "intersection_over_k": intersection / exact_k,
        "set_jaccard": intersection / (2 * exact_k - intersection),
    }
    return {
        metric: {
            "full_validation": aggregate_frame_values(records, frame_values),
            "offcenter_top_quartile": aggregate_frame_values(
                records,
                frame_values,
                eligible_clip_ids=offcenter_clip_ids,
            ),
        }
        for metric, frame_values in values.items()
    }


def source_prior_cells(
    records: Sequence[Mapping],
    cell_mass: np.ndarray,
    *,
    exact_k: int = 16,
    sources: Sequence[str] = STAVIS_SOURCES,
) -> dict[str, list[int]]:
    mass = _validate_cell_mass(cell_mass, len(records))
    result = {}
    for source in sources:
        indices = [
            index
            for index, record in enumerate(records)
            if record["split"] == "train" and record["source"] == source
        ]
        if not indices:
            raise ValueError(f"Training split lacks source {source}")
        summed = mass[indices].sum(axis=(0, 1), dtype=np.float64)
        cells = sorted(range(196), key=lambda cell: (-summed[cell], cell))[:exact_k]
        result[source] = cells
    return result


def _constant_actions(
    records: Sequence[Mapping],
    cells_by_source: Mapping[str, Sequence[int]],
    *,
    clip_len: int,
) -> np.ndarray:
    return np.asarray(
        [
            [list(cells_by_source[str(record["source"])]) for _ in range(clip_len)]
            for record in records
        ],
        dtype=np.int64,
    )


def policy_report(
    records: Sequence[Mapping],
    cell_mass: np.ndarray,
    actions: np.ndarray,
    *,
    offcenter_clip_ids: set[str],
    train_source_prior_cells: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Compute coverage, center/content controls, SIM/CC and structure."""
    mass = _validate_cell_mass(cell_mass, len(records))
    action_array = validate_action_array(
        actions,
        num_records=len(records),
        clip_len=mass.shape[1],
        exact_k=actions.shape[2],
    )
    exact_k = action_array.shape[2]
    center_cells = center_order(14)[:exact_k].tolist()
    center_actions = np.broadcast_to(
        np.asarray(center_cells, dtype=np.int64), action_array.shape
    )
    counts = np.bincount(action_array.reshape(-1), minlength=196)
    static_cells = sorted(range(196), key=lambda cell: (-counts[cell], cell))[:exact_k]
    static_actions = np.broadcast_to(
        np.asarray(static_cells, dtype=np.int64), action_array.shape
    )
    prior_actions = _constant_actions(
        records,
        train_source_prior_cells,
        clip_len=mass.shape[1],
    )
    shuffled_indices = within_source_different_video_indices(records)
    shuffled_actions = action_array[shuffled_indices]
    methods = {
        "actual": action_array,
        "center16": center_actions,
        "selection_frequency_top16": static_actions,
        "same_source_shuffled_video": shuffled_actions,
        "train_source_prior16": prior_actions,
    }
    coverage = {}
    for method, method_actions in methods.items():
        values = selected_mass(mass, method_actions)
        coverage[method] = {
            "full_validation": aggregate_frame_values(records, values),
            "offcenter_top_quartile": aggregate_frame_values(
                records,
                values,
                eligible_clip_ids=offcenter_clip_ids,
            ),
        }

    sim, cc = uniform_density_saliency(mass, action_array)
    center_intersection = action_intersection(action_array, center_actions) / exact_k
    shuffled_intersection = action_intersection(action_array, shuffled_actions) / exact_k
    probabilities = counts.astype(np.float64) / counts.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / math.log(196))
    unique_sets = {
        tuple(sorted(int(cell) for cell in row))
        for clip in action_array
        for row in clip
    }
    return {
        "schema_version": 1,
        "action_contract": {
            "clip_len": mass.shape[1],
            "exact_k": exact_k,
            "num_fine_cells": 196,
            "fine_action_offset": 69,
            "selection_without_replacement": True,
        },
        "coverage": coverage,
        "saliency": {
            "uniform_selected_density_sim": {
                "full_validation": aggregate_frame_values(records, sim),
                "offcenter_top_quartile": aggregate_frame_values(
                    records, sim, eligible_clip_ids=offcenter_clip_ids
                ),
            },
            "uniform_selected_density_cc": {
                "full_validation": aggregate_frame_values(records, cc),
                "offcenter_top_quartile": aggregate_frame_values(
                    records, cc, eligible_clip_ids=offcenter_clip_ids
                ),
            },
        },
        "content_dependence": {
            "actual_minus_static_macro": (
                coverage["actual"]["full_validation"]["macro_source_mean"]
                - coverage["selection_frequency_top16"]["full_validation"]["macro_source_mean"]
            ),
            "actual_minus_shuffled_macro": (
                coverage["actual"]["full_validation"]["macro_source_mean"]
                - coverage["same_source_shuffled_video"]["full_validation"]["macro_source_mean"]
            ),
            "actual_minus_center_macro": (
                coverage["actual"]["full_validation"]["macro_source_mean"]
                - coverage["center16"]["full_validation"]["macro_source_mean"]
            ),
        },
        "structure": {
            "center16_overlap_fraction": {
                "full_validation": aggregate_frame_values(records, center_intersection),
                "offcenter_top_quartile": aggregate_frame_values(
                    records,
                    center_intersection,
                    eligible_clip_ids=offcenter_clip_ids,
                ),
            },
            "same_source_shuffled_overlap_fraction": {
                "full_validation": aggregate_frame_values(records, shuffled_intersection),
                "offcenter_top_quartile": aggregate_frame_values(
                    records,
                    shuffled_intersection,
                    eligible_clip_ids=offcenter_clip_ids,
                ),
            },
            "normalized_selection_entropy": entropy,
            "unique_frame_selection_sets": len(unique_sets),
            "unique_frame_selection_fraction": len(unique_sets) / (len(records) * mass.shape[1]),
            "selection_frequency_top16_cells": static_cells,
            "center16_cells": center_cells,
            "train_source_prior16_cells": {
                source: list(train_source_prior_cells[source]) for source in STAVIS_SOURCES
            },
            "selection_frequency_per_frame": (
                counts.astype(np.float64) / (len(records) * mass.shape[1])
            ).tolist(),
        },
    }


def six_seed_summary(values: Sequence[float]) -> dict[str, Any]:
    """Return the preregistered descriptive six-seed 90% t interval."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError("A complete summary requires exactly six finite seed values")
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1))
    half_width = 2.0150483733330233 * sample_sd / math.sqrt(6)
    return {
        "num_seeds": 6,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci90_low": mean - half_width,
        "ci90_high": mean + half_width,
        "interval": "descriptive_two_sided_t_df5_unadjusted",
    }


def source_stratified_video_bootstrap_delta(
    left_reports: Sequence[Mapping],
    right_reports: Sequence[Mapping],
    *,
    iterations: int = 10000,
    seed: int = 20260910,
) -> dict[str, Any]:
    """Paired source-stratified video bootstrap, averaged across six seeds."""
    if len(left_reports) != 6 or len(right_reports) != 6:
        raise ValueError("Bootstrap requires six paired seed reports")
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    source_video_deltas: dict[str, np.ndarray] = {}
    for source in STAVIS_SOURCES:
        rows = []
        expected_ids = None
        for left, right in zip(left_reports, right_reports):
            left_videos = left["per_source"][source]["video_values"]
            right_videos = right["per_source"][source]["video_values"]
            if set(left_videos) != set(right_videos):
                raise ValueError("Paired bootstrap reports cover different videos")
            video_ids = sorted(left_videos)
            if expected_ids is None:
                expected_ids = video_ids
            elif expected_ids != video_ids:
                raise ValueError("All paired seeds must cover the same videos")
            rows.append([float(left_videos[key]) - float(right_videos[key]) for key in video_ids])
        source_video_deltas[source] = np.asarray(rows, dtype=np.float64)

    point = float(
        np.mean(
            [values.mean(axis=1).mean() for values in source_video_deltas.values()]
        )
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        source_values = []
        for values in source_video_deltas.values():
            sampled = rng.integers(0, values.shape[1], size=values.shape[1])
            source_values.append(float(values[:, sampled].mean(axis=1).mean()))
        samples[iteration] = np.mean(source_values)
    low, high = np.quantile(samples, (0.05, 0.95))
    return {
        "difference": point,
        "ci90_low": float(low),
        "ci90_high": float(high),
        "iterations": iterations,
        "bootstrap_seed": seed,
        "cluster_unit": "video_stratified_by_source_with_paired_methods_and_seeds",
        "interpretation": "descriptive_unadjusted",
    }


def evaluate_practical_gate(
    *,
    supervised_reports: Mapping[int, Mapping],
    rl_reports: Mapping[int, Mapping],
    supervised_resource_complete: Mapping[int, bool],
    supervised_nominal_action_rows: int,
    rl_nominal_action_rows: int,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the practical HLVid gate; missing evidence never admits HLVid."""
    expected_seeds = tuple(range(440826, 440832))
    missing = []
    for seed in expected_seeds:
        if seed not in supervised_reports:
            missing.append(f"supervised_endpoint_seed_{seed}")
        if seed not in rl_reports:
            missing.append(f"rl_endpoint_seed_{seed}")
        if not supervised_resource_complete.get(seed, False):
            missing.append(f"supervised_resource_accounting_seed_{seed}")
    if supervised_nominal_action_rows <= 0 or rl_nominal_action_rows <= 0:
        missing.append("positive_nominal_action_row_counts")
    if missing:
        return {
            "schema_version": 1,
            "status": "incomplete_required_evidence",
            "decision": "do_not_admit_hlvid",
            "hlvid_admitted": False,
            "missing": sorted(missing),
            "route_a_pass": False,
            "route_b_pass": False,
        }

    supervised_coverage = np.asarray(
        [
            supervised_reports[seed]["coverage"]["actual"]["full_validation"]["macro_source_mean"]
            for seed in expected_seeds
        ]
    )
    rl_coverage = np.asarray(
        [
            rl_reports[seed]["coverage"]["actual"]["full_validation"]["macro_source_mean"]
            for seed in expected_seeds
        ]
    )
    supervised_offcenter = np.asarray(
        [
            supervised_reports[seed]["coverage"]["actual"]["offcenter_top_quartile"]["macro_source_mean"]
            for seed in expected_seeds
        ]
    )
    rl_offcenter = np.asarray(
        [
            rl_reports[seed]["coverage"]["actual"]["offcenter_top_quartile"]["macro_source_mean"]
            for seed in expected_seeds
        ]
    )
    dynamic_static = float(
        np.mean(
            [supervised_reports[seed]["content_dependence"]["actual_minus_static_macro"] for seed in expected_seeds]
        )
    )
    dynamic_shuffled = float(
        np.mean(
            [supervised_reports[seed]["content_dependence"]["actual_minus_shuffled_macro"] for seed in expected_seeds]
        )
    )
    overall_delta = float(np.mean(supervised_coverage - rl_coverage))
    offcenter_delta = float(np.mean(supervised_offcenter - rl_offcenter))
    reduction = 1.0 - supervised_nominal_action_rows / rl_nominal_action_rows
    dynamic_positive = dynamic_static > 0 and dynamic_shuffled > 0
    route_a = (
        overall_delta >= float(thresholds["route_a"]["minimum_supervised_minus_rl_coverage"])
        and reduction
        >= float(thresholds["route_a"]["minimum_reduction_in_nominal_training_trajectory_action_rows"])
        and dynamic_positive
    )
    route_b = (
        (
            overall_delta >= float(thresholds["route_b"]["minimum_overall_coverage_improvement"])
            or offcenter_delta
            >= float(thresholds["route_b"]["or_minimum_offcenter_coverage_improvement"])
        )
        and overall_delta
        >= float(thresholds["route_b"]["minimum_overall_supervised_minus_rl_coverage"])
        and dynamic_positive
    )
    admit = bool(route_a or route_b)
    return {
        "schema_version": 1,
        "status": "complete_practical_descriptive_gate",
        "decision": "admit_all_six_supervised_endpoints_to_hlvid" if admit else "do_not_admit_hlvid",
        "hlvid_admitted": admit,
        "route_a_pass": bool(route_a),
        "route_b_pass": bool(route_b),
        "evidence": {
            "supervised_minus_rl_full_validation_macro_coverage": overall_delta,
            "supervised_minus_rl_offcenter_macro_coverage": offcenter_delta,
            "supervised_minus_static_macro_coverage": dynamic_static,
            "supervised_minus_shuffled_macro_coverage": dynamic_shuffled,
            "nominal_training_trajectory_action_row_reduction": reduction,
            "supervised_nominal_training_trajectory_action_rows": supervised_nominal_action_rows,
            "rl_nominal_training_trajectory_action_rows": rl_nominal_action_rows,
            "complete_supervised_resource_accounting": True,
        },
        "interpretation": (
            "Practical escalation rule only; this is not a significance or formal "
            "non-inferiority decision."
        ),
    }
