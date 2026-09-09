#!/usr/bin/env python3
"""Replay R2d's question-agnostic HLVid processor without NVILA generation.

The legacy evaluator persisted QA answers per question but only persisted
process-local allocation counters at normal process exit. This replay recovers
allocation and expanded-context evidence from the identical video inputs. It
processes each unique video once, then weights that observation by the number
of HLVid questions for the video. No answer generation or scoring occurs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from autogaze.human_gaze.r2d_hlvid import (
    install_r2d_hlvid_forward,
    merge_raw_allocations,
    summarize_raw_allocation,
)
from scripts.human_gaze.evaluate_hlvid_nvila_r2d import (
    EXPECTED_COMMON_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_RUNNER_SHA256,
    ESTABLISHED_PROTOCOL,
    sha256_file,
    validate_calibration,
    validate_checkpoint,
)
from scripts.runners.hlvid_evidence import (
    canonical_json,
    export_evidence_jsonl,
    fingerprint,
    load_resume_state,
    resume_compatibility_key,
    stable_example_key,
    stable_video_key,
    write_evidence_record,
)


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2d-mode", choices=["variable", "forced_k16"], required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--checkpoint-model-sha256", required=True)
    parser.add_argument("--checkpoint-config-sha256", required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--eos-logit-bias", type=float, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--autogaze-model-id", type=Path, required=True)
    parser.add_argument(
        "--legacy-runner-dir",
        type=Path,
        default=Path("/home/stud/latn/AutoGaze/scripts/runners"),
    )
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--qa-results", type=Path, required=True)
    parser.add_argument("--protocol-audit-dir", type=Path, required=True)
    parser.add_argument("--evidence-records-dir", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--video-records-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--validation-reference-summary", type=Path)
    parser.add_argument("--validated-replay-summary", type=Path)
    parser.add_argument("--expected-code-commit", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_replay_by_video(
    records: list[dict[str, Any]], expected_compatibility_key: str
) -> dict[str, dict[str, Any]]:
    """Validate completed records before resume skipping or aggregation."""

    completed: dict[str, dict[str, Any]] = {}
    for record in records:
        video = record["video_path"]
        if video in completed:
            raise ValueError(f"Duplicate completed replay video: {video}")
        if record.get("compatibility_key") != expected_compatibility_key:
            raise ValueError(f"Resume identity mismatch for completed video: {video}")
        if record.get("state") != "completed_processor_observation":
            raise ValueError(f"Non-complete record found in completed replay output: {video}")
        completed[video] = record
    return completed


def write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish one immutable replay observation without replacing an event."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable replay record: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_video_records(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def export_video_records(directory: Path, output: Path) -> Path:
    records = read_video_records(directory)
    records.sort(key=lambda row: row["video_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-export-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def git_execution(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            text=True,
        ).strip()
    )
    if commit != expected_commit or dirty:
        raise ValueError(
            f"Allocation replay requires clean immutable commit {expected_commit}; "
            f"found commit={commit}, dirty={dirty}"
        )
    device = torch.cuda.get_device_properties(0)
    return {
        "code_commit": commit,
        "code_dirty": False,
        "cuda_device_name": device.name,
        "cuda_total_memory_bytes": int(device.total_memory),
    }


def tensor_structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (list, tuple)):
        return [tensor_structure(item) for item in value]
    if isinstance(value, dict):
        return {str(key): tensor_structure(item) for key, item in sorted(value.items())}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__}


def frame_content_sha256(frames: list[Any]) -> str:
    digest = hashlib.sha256()
    for position, frame in enumerate(frames):
        array = np.asarray(frame)
        digest.update(position.to_bytes(4, byteorder="little", signed=False))
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(np.ascontiguousarray(array)))
    return digest.hexdigest()


def decodable_uniform_indices(video_path: Path, num_frames: int) -> tuple[int, list[int]]:
    """Reproduce the legacy frame-count and round(linspace) index rule."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if capture.grab():
            break
        frame_count -= 1
    capture.release()
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    return frame_count, [int(index) for index in indices]


def context_records(processor, rows: list[dict], visual_tokens: int) -> list[dict[str, int]]:
    records = []
    video_token = processor.tokenizer.video_token
    for row in rows:
        prompt = f"{video_token}\n\nQuestion: {row['question']}"
        encoded = processor._preprocess_text(
            [prompt],
            image_token_padding_strategy=[],
            video_token_padding_strategy=[[visual_tokens]],
            return_tensors="pt",
        )
        total = int(encoded["input_ids"].shape[-1])
        records.append(
            {
                "question_id": int(row["question_id"]),
                "expanded_input_tokens": total,
                "visual_tokens": visual_tokens,
                "text_and_special_tokens": total - visual_tokens,
            }
        )
    return records


def raw_counter(
    *,
    observed_count: int,
    observed_sum: int | float,
    minimum: int | float | None,
    maximum: int | float | None,
    unit: str,
    source: str = "processor_replay",
) -> dict[str, Any]:
    """Express additive adapter totals in the shared WP0 counter schema."""

    if observed_count < 0:
        raise ValueError("Counter observations must be nonnegative")
    if observed_count == 0 and (minimum is not None or maximum is not None):
        raise ValueError("Empty counters cannot have extrema")
    return {
        "availability": "complete",
        "observed_count": int(observed_count),
        "expected_observations": int(observed_count),
        "observed_sum": observed_sum,
        "mean": observed_sum / observed_count if observed_count else None,
        "min": minimum,
        "max": maximum,
        "unit": unit,
        "source": source,
    }


def evidence_counters(raw: dict[str, Any], visual_tokens: int, context_tokens: int) -> dict[str, Any]:
    decoder = raw["decoder"]
    post = raw["post_resolution_adaptation"]
    return {
        "decoder_spatial_actions": raw_counter(
            observed_count=int(decoder["frame_observations"]),
            observed_sum=int(decoder["spatial_actions_sum"]),
            minimum=decoder["spatial_actions_min"],
            maximum=decoder["spatial_actions_max"],
            unit="actions_per_frame_observation",
        ),
        "decoder_valid_action_tokens": raw_counter(
            observed_count=int(decoder["frame_observations"]),
            observed_sum=int(decoder["valid_action_tokens"]),
            minimum=None,
            maximum=None,
            unit="tokens_per_frame_observation",
        ),
        "decoder_padded_slots": raw_counter(
            observed_count=int(decoder["frame_observations"]),
            observed_sum=int(decoder["padded_position_slots"]),
            minimum=None,
            maximum=None,
            unit="slots_per_frame_observation",
        ),
        "decoder_eos_actions": raw_counter(
            observed_count=int(decoder["frame_observations"]),
            observed_sum=int(decoder["eos_actions"]),
            minimum=None,
            maximum=None,
            unit="eos_per_frame_observation",
        ),
        "retained_patches": raw_counter(
            observed_count=int(post["frame_observations"]),
            observed_sum=int(post["retained_patches_sum"]),
            minimum=post["retained_patches_min"],
            maximum=post["retained_patches_max"],
            unit="patches_per_frame_observation",
        ),
        "post_adaptation_padded_slots": raw_counter(
            observed_count=int(post["frame_observations"]),
            observed_sum=int(post["padded_position_slots"]),
            minimum=None,
            maximum=None,
            unit="slots_per_frame_observation",
        ),
        "expanded_visual_tokens": raw_counter(
            observed_count=1,
            observed_sum=int(visual_tokens),
            minimum=int(visual_tokens),
            maximum=int(visual_tokens),
            unit="tokens_per_question",
        ),
        "expanded_context_tokens": raw_counter(
            observed_count=1,
            observed_sum=int(context_tokens),
            minimum=int(context_tokens),
            maximum=int(context_tokens),
            unit="tokens_per_question",
        ),
    }


def load_protocol_audit(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    summary_path = directory / "summary.json"
    manifest_path = directory / "protocol_runtime_manifest.json"
    records_path = directory / "decode_audit.jsonl"
    supplement_path = directory / "audit_supplement.json"
    for path in (summary_path, manifest_path, records_path, supplement_path):
        if not path.is_file():
            raise FileNotFoundError(f"Protocol audit artifact is missing: {path}")
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    supplement = json.loads(supplement_path.read_text())
    if manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise ValueError("Protocol audit dataset identity mismatch")
    if manifest.get("legacy_loader_sha256") != EXPECTED_RUNNER_SHA256:
        raise ValueError("Protocol audit legacy-loader identity mismatch")
    if manifest.get("protocol", {}).get("num_video_frames") != 128:
        raise ValueError("Protocol audit did not use the frozen 128-frame grid")
    if summary.get("reuse_qualification") != "no_current_requested_decode_failures":
        raise ValueError(
            "Protocol audit requires affected-subset inspection before replay: "
            f"{summary.get('affected_question_rows')}"
        )
    supplement_hashes = supplement.get("artifact_hashes") or {}
    for name, path in (
        ("summary", summary_path),
        ("protocol_runtime_manifest", manifest_path),
        ("decode_audit", records_path),
    ):
        if supplement_hashes.get(name, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"Protocol audit supplement hash mismatch: {name}")
    if supplement.get("reuse_qualification") != summary.get("reuse_qualification"):
        raise ValueError("Protocol audit supplement qualification mismatch")
    records = read_jsonl(records_path)
    by_video = {str(record["manifest_path"]): record for record in records}
    if len(by_video) != len(records) or len(by_video) != int(summary.get("video_count", -1)):
        raise ValueError("Protocol audit video coverage is incomplete or duplicated")
    provenance = {
        "directory": str(directory.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "decode_audit_sha256": sha256_file(records_path),
        "supplement_sha256": sha256_file(supplement_path),
        "reuse_qualification": summary["reuse_qualification"],
        "historical_certification": bool(summary.get("historical_certification")),
        "live_model_files_verified": bool(manifest.get("live_model_files_verified")),
        "runtime": supplement.get("runtime"),
        "audit_git": supplement.get("git"),
    }
    return summary, by_video, provenance


def load_complete_qa_results(path: Path, expected_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    qa_rows = read_jsonl(path)
    expected_ids = [int(row["question_id"]) for row in expected_rows]
    actual_ids = [int(row["question_id"]) for row in qa_rows]
    if actual_ids != expected_ids:
        raise ValueError("Allocation replay requires the complete ordered durable 268-row QA stream")
    by_id = {int(row["question_id"]): row for row in qa_rows}
    if len(by_id) != len(qa_rows):
        raise ValueError("QA results contain duplicate question IDs")
    for source in expected_rows:
        result = by_id[int(source["question_id"])]
        if str(result.get("video_path")) != str(source["video_path"]):
            raise ValueError("QA result video identity differs from the benchmark row")
    return by_id


def small_model_file_manifest(model_path: Path) -> dict[str, str]:
    """Hash the executable model/processor metadata without rehashing weight shards."""

    names = (
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "configuration_nvila.py",
        "generation_config.json",
        "merges.txt",
        "modeling_nvila.py",
        "preprocessor_config.json",
        "processing_nvila.py",
        "processor_config.json",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    missing = [name for name in names if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"NVILA runtime identity files are missing: {missing}")
    return {name: sha256_file(model_path / name) for name in names}


def legacy_comparable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    decoder = stats["decoder"]
    post = stats["post_resolution_adaptation"]
    return {
        "model_calls": stats["model_calls"],
        "decoder": {
            key: decoder[key]
            for key in (
                "calls",
                "frame_observations",
                "spatial_actions_per_frame_mean",
                "spatial_actions_per_frame_min",
                "spatial_actions_per_frame_max",
                "eos_action_rate",
                "spatial_action_histogram",
            )
        },
        "post_resolution_adaptation": {
            key: post[key]
            for key in (
                "frame_observations",
                "retained_patches_per_frame_mean",
                "retained_patches_per_frame_min",
                "retained_patches_per_frame_max",
            )
        },
    }


def validate_reference(
    records: list[dict[str, Any]],
    reference_summary_path: Path | None,
    validated_replay_summary_path: Path | None,
) -> dict[str, Any]:
    if reference_summary_path is not None and validated_replay_summary_path is not None:
        raise ValueError("Choose either direct VARIABLE validation or a validated replay summary")
    if validated_replay_summary_path is not None:
        validator = json.loads(validated_replay_summary_path.read_text())
        direct = validator.get("validation") or {}
        if direct.get("status") != "pass" or direct.get("criterion") != "exact_allocation_statistics_match":
            raise ValueError("Replay validator was not established by an exact VARIABLE reference")
        if not records:
            raise ValueError("Cannot validate an empty replay")
        current = records[0]
        if validator.get("mode") != "variable" or current.get("mode") != "variable":
            raise ValueError("Shared replay validation is only valid for VARIABLE mode")
        if validator.get("execution", {}).get("code_commit") != current.get("execution", {}).get("code_commit"):
            raise ValueError("Replay validator code commit differs from the current replay")
        if validator.get("protocol") != current.get("protocol"):
            raise ValueError("Replay validator protocol/processor identity differs from the current replay")
        return {
            "status": "pass",
            "criterion": "shared_replay_implementation_validated_by_exact_variable_reference",
            "validated_replay_summary": str(validated_replay_summary_path.resolve()),
            "validated_replay_summary_sha256": sha256_file(validated_replay_summary_path),
            "direct_reference": direct,
        }
    if reference_summary_path is None:
        return {"status": "not_requested"}
    reference = json.loads(reference_summary_path.read_text())
    adapter = reference.get("r2d_hlvid_adapter") or {}
    if int(reference.get("num_examples", -1)) != 1:
        raise ValueError("Replay validation reference must be an uninterrupted one-question run")
    if adapter.get("mode") != "variable":
        raise ValueError("Replay validation must use a VARIABLE reference, not a forced-K arm")
    reference_result = reference_summary_path.with_name("results.jsonl")
    result_rows = read_jsonl(reference_result)
    if len(result_rows) != 1:
        raise ValueError("Replay validation reference must have exactly one QA row")
    target = result_rows[0]
    matches = [record for record in records if target["question_id"] in record["question_ids"]]
    if len(matches) != 1 or matches[0]["video_path"] != target["video_path"]:
        raise ValueError("Replay output does not contain the VARIABLE reference video/question")
    matched = matches[0]
    if matched["checkpoint"]["model_sha256"] != adapter["checkpoint"]["model_sha256"]:
        raise ValueError("Replay checkpoint does not match the VARIABLE validation reference")
    if matched["calibration"]["sha256"] != adapter["eos_calibration"]["sha256"]:
        raise ValueError("Replay EOS calibration does not match the VARIABLE validation reference")
    replay_stats = legacy_comparable_stats(summarize_raw_allocation(matched["allocation_raw"]))
    reference_stats = legacy_comparable_stats(adapter["allocation_statistics"])
    if replay_stats != reference_stats:
        raise ValueError("Processor replay does not exactly reproduce the uninterrupted VARIABLE reference")
    return {
        "status": "pass",
        "criterion": "exact_allocation_statistics_match",
        "reference_summary": str(reference_summary_path.resolve()),
        "reference_summary_sha256": sha256_file(reference_summary_path),
        "reference_results_sha256": sha256_file(reference_result),
        "question_id": int(target["question_id"]),
        "video_path": target["video_path"],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Faithful R2d allocation replay requires CUDA")
    execution = git_execution(REPO_ROOT, args.expected_code_commit)

    runner_dir = args.legacy_runner_dir.resolve()
    runner_path = runner_dir / "evaluate_hlvid_nvila.py"
    common_path = runner_dir / "hlvid_common.py"
    if sha256_file(runner_path) != EXPECTED_RUNNER_SHA256:
        raise ValueError("Established HLVid runner SHA-256 mismatch")
    if sha256_file(common_path) != EXPECTED_COMMON_SHA256:
        raise ValueError("Established HLVid common-helper SHA-256 mismatch")
    sys.path.insert(0, str(runner_dir))
    import evaluate_hlvid_nvila as runner

    dataset_parquet = args.dataset_root / "data" / "test-00000-of-00001.parquet"
    if sha256_file(dataset_parquet) != EXPECTED_DATASET_SHA256:
        raise ValueError("Protected HLVid benchmark split SHA-256 mismatch")
    checkpoint = validate_checkpoint(
        args.autogaze_model_id,
        args.checkpoint_model_sha256,
        args.checkpoint_config_sha256,
    )
    calibration = validate_calibration(
        args.calibration_json,
        expected_sha256=args.calibration_sha256,
        checkpoint=args.autogaze_model_id,
        eos_logit_bias=args.eos_logit_bias,
    )
    actual_protocol = {
        "num_video_frames": args.num_video_frames,
        "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
        "max_tiles_video": args.max_tiles_video,
    }
    mismatches = {
        key: {"expected": ESTABLISHED_PROTOCOL[key], "actual": value}
        for key, value in actual_protocol.items()
        if value != ESTABLISHED_PROTOCOL[key]
    }
    if mismatches:
        raise ValueError(f"HLVid established-protocol mismatch: {mismatches}")

    rows = runner.load_rows(args.dataset_root)
    if len(rows) != ESTABLISHED_PROTOCOL["num_examples"]:
        raise ValueError("Expected the complete 268-question HLVid benchmark")
    qa_by_id = load_complete_qa_results(args.qa_results, rows)
    _, audit_by_video, protocol_audit = load_protocol_audit(args.protocol_audit_dir)
    rows_by_video: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        rows_by_video.setdefault(str(row["video_path"]), []).append(row)
    selected_videos = list(rows_by_video)
    if args.limit_videos is not None:
        if args.limit_videos <= 0:
            raise ValueError("--limit-videos must be positive")
        selected_videos = selected_videos[: args.limit_videos]

    protocol_identity = {
        **ESTABLISHED_PROTOCOL,
        **actual_protocol,
        "dataset_parquet_sha256": EXPECTED_DATASET_SHA256,
        "legacy_runner_sha256": EXPECTED_RUNNER_SHA256,
        "legacy_common_sha256": EXPECTED_COMMON_SHA256,
        "processor_source_sha256": sha256_file(args.model_path / "processing_nvila.py"),
        "nvila_runtime_files": small_model_file_manifest(args.model_path),
        "protocol_audit_identity": {
            key: protocol_audit[key]
            for key in (
                "summary_sha256",
                "manifest_sha256",
                "decode_audit_sha256",
                "supplement_sha256",
                "reuse_qualification",
                "runtime",
            )
        },
        "allocation_scope": "one_question_agnostic_processor_call_per_unique_video_weighted_by_question_count",
        "nvila_generation_calls": 0,
        "processor_arguments": {
            "gazing_ratio_tile": [0.2] + [0.06] * 15,
            "gazing_ratio_thumbnail": 1,
            "task_loss_requirement_tile": 0.6,
            "task_loss_requirement_thumbnail": None,
            "max_batch_size_autogaze": args.max_batch_size_autogaze,
        },
        "prompt_template": "<video_token>\\n\\nQuestion: {question}",
        "generation": {"performed": False, "source_qa_runner_sha256": EXPECTED_RUNNER_SHA256},
        "parser_and_scoring": {
            "implementation_sha256": EXPECTED_COMMON_SHA256,
            "metric": ESTABLISHED_PROTOCOL["metric"],
        },
        "recovery": {
            "resolution_adapter": "NVILA processing_nvila.py",
            "decoder_action_ids": [69, 264],
            "eos_action_id": 265,
            "variable_min_spatial_actions": 4,
            "variable_max_generation_tokens": 36,
            "forced_spatial_actions": 16,
        },
    }
    compatibility_identity = {
        "schema_version": SCHEMA_VERSION,
        "base_seed": args.base_seed,
        "training_seed": args.training_seed,
        "mode": args.r2d_mode,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "protocol": protocol_identity,
        "qa_source": {
            "run_id": "20260901-0156_r2f-r2d-hlvid-secondary_ecf1535",
            "results_jsonl_sha256": sha256_file(args.qa_results),
        },
    }
    compatibility_key = resume_compatibility_key(compatibility_identity)

    replay_paths = (
        args.evidence_records_dir,
        args.evidence_output,
        args.video_records_dir,
        args.output,
        args.summary_output,
    )
    if not args.resume and any(path.exists() for path in replay_paths):
        raise FileExistsError("Refusing to overwrite existing allocation-replay evidence")
    existing = read_video_records(args.video_records_dir) if args.resume else []
    existing_by_video = completed_replay_by_video(existing, compatibility_key)
    resume_state = load_resume_state(args.evidence_records_dir, compatibility_key)

    processor = runner.AutoProcessor.from_pretrained(
        args.model_path,
        autogaze_model_id=args.autogaze_model_id,
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
        gazing_ratio_tile=[0.2] + [0.06] * 15,
        gazing_ratio_thumbnail=1,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=None,
        max_batch_size_autogaze=args.max_batch_size_autogaze,
        trust_remote_code=True,
    )
    stats = install_r2d_hlvid_forward(
        processor._autogaze_model,
        mode=args.r2d_mode,
        eos_logit_bias=args.eos_logit_bias if args.r2d_mode == "variable" else None,
    )
    video_token = processor.tokenizer.video_token

    for video_index, video in enumerate(selected_videos):
        if video in existing_by_video:
            continue
        video_rows = rows_by_video[video]
        video_path = args.dataset_root / "videos" / video
        video_sha256 = sha256_file(video_path)
        video_key = stable_video_key(video, "test")
        question_keys = [stable_example_key(int(row["question_id"]), "test") for row in video_rows]
        stable_key_payload = {
            "compatibility_key": compatibility_key,
            "video_path": video,
            "video_sha256": video_sha256,
        }
        replay_key = fingerprint(stable_key_payload)
        attempt_id = fingerprint(
            {"replay_key": replay_key, "started_time_ns": time.time_ns(), "pid": os.getpid()}
        )
        for row, example_key in zip(video_rows, question_keys):
            if example_key in resume_state["completed"]:
                continue
            write_evidence_record(
                args.evidence_records_dir,
                {
                    "schema_version": SCHEMA_VERSION,
                    "example_key": example_key,
                    "video_key": video_key,
                    "attempt_id": attempt_id,
                    "compatibility_key": compatibility_key,
                    "state": "attempt_started",
                    "question_id": int(row["question_id"]),
                    "question_index": int(row["question_id"]),
                    "split": "test",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "measurement_origin": "processor_replay",
                },
            )

        started = time.perf_counter()
        decodable_count, source_indices = decodable_uniform_indices(video_path, args.num_video_frames)
        frames = runner.load_uniform_frames(video_path, args.num_video_frames)
        if len(frames) != args.num_video_frames:
            raise ValueError(f"Legacy loader did not return {args.num_video_frames} frames for {video}")
        frames_sha256 = frame_content_sha256(frames)
        decode = audit_by_video.get(video)
        if decode is None:
            raise ValueError(f"Protocol audit does not cover replay video: {video}")
        if not decode.get("decode_usable") or decode.get("legacy_output_substituted"):
            raise ValueError(f"Protocol audit found a material decode mismatch for replay video: {video}")
        if decode.get("video_key") != video_key:
            raise ValueError(f"Protocol audit stable video key differs from the replay key: {video}")
        if [int(index) for index in decode["intended_indices"]] != source_indices:
            raise ValueError(f"Replay source-index grid differs from the protocol audit: {video}")
        representative_prompt = f"{video_token}\n\nQuestion: {video_rows[0]['question']}"
        inputs = processor(text=representative_prompt, videos=[frames], return_tensors="pt")
        torch.cuda.synchronize()
        allocation_raw = stats.pop_raw_dict()
        allocation_summary = summarize_raw_allocation(allocation_raw)
        input_ids = inputs["input_ids"]
        video_token_id = int(processor.tokenizer.convert_tokens_to_ids(video_token))
        visual_tokens = int((input_ids == video_token_id).sum().item())
        contexts = context_records(processor, video_rows, visual_tokens)
        representative_context = next(
            row for row in contexts if row["question_id"] == int(video_rows[0]["question_id"])
        )
        if representative_context["expanded_input_tokens"] != int(input_ids.shape[-1]):
            raise ValueError("Independent context reconstruction does not match processor output")
        elapsed = time.perf_counter() - started
        record = {
            **compatibility_identity,
            "compatibility_identity": compatibility_identity,
            "compatibility_key": compatibility_key,
            "execution": execution,
            "record_type": "r2d_hlvid_allocation_replay_video",
            "replay_key": replay_key,
            "state": "completed_processor_observation",
            "video_index": video_index,
            "video_path": video,
            "video_key": video_key,
            "video_sha256": video_sha256,
            "question_ids": [int(row["question_id"]) for row in video_rows],
            "question_keys": question_keys,
            "question_count": len(video_rows),
            "source_frames": {
                "decodable_frame_count": decodable_count,
                "requested_source_indices": source_indices,
                "effective_source_indices": decode["effective_indices"],
                "legacy_loaded_frame_count": len(frames),
                "legacy_loaded_frames_sha256": frames_sha256,
                "protocol_audit_video_key": decode["video_key"],
            },
            "processor_tensors": tensor_structure(inputs),
            "expanded_context_by_question": contexts,
            "visual_tokens_per_question": visual_tokens,
            "allocation_raw": allocation_raw,
            "allocation_summary": allocation_summary,
            "measurement_origin": "processor_replay",
            "processor_replay_wall_seconds": elapsed,
            "nvila_generation_calls": 0,
            "qa_reuse": {
                "reused": True,
                "source_run_id": "20260901-0156_r2f-r2d-hlvid-secondary_ecf1535",
                "equivalence_verified": True,
                "equivalence_criteria": [
                    "same_dataset_and_split",
                    "same_legacy_loader_and_uniform_128_frames",
                    "same_nvila_processor_source_and_128_64_max_tiles_video48_settings",
                    "same_r2d_checkpoint_and_eos_calibration",
                    "no_nvila_generation_or_answer_change",
                ],
            },
        }
        contexts_by_id = {int(context["question_id"]): context for context in contexts}
        for row, example_key in zip(video_rows, question_keys):
            if example_key in resume_state["completed"]:
                continue
            question_id = int(row["question_id"])
            context = contexts_by_id[question_id]
            write_evidence_record(
                args.evidence_records_dir,
                {
                    "schema_version": SCHEMA_VERSION,
                    "example_key": example_key,
                    "video_key": video_key,
                    "attempt_id": attempt_id,
                    "compatibility_key": compatibility_key,
                    "state": "completed_answer",
                    "question_id": question_id,
                    "question_index": question_id,
                    "split": "test",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "measurement_origin": "processor_replay",
                    "answer": {
                        "source": "existing_durable_qa_jsonl",
                        "source_row": qa_by_id[question_id],
                    },
                    "decode": {
                        "intended_indices": decode["intended_indices"],
                        "effective_indices": decode["effective_indices"],
                        "metadata_count_corrected": bool(decode["metadata_count_corrected"]),
                        "sampled_read_failure": bool(decode["sampled_read_failure"]),
                        "legacy_output_substituted": bool(decode["legacy_output_substituted"]),
                        "decode_usable": bool(decode["decode_usable"]),
                        "loaded_frames_sha256": frames_sha256,
                    },
                    "counters": evidence_counters(
                        allocation_raw,
                        visual_tokens,
                        int(context["expanded_input_tokens"]),
                    ),
                    "context": {
                        "tokenizer_input_length": None,
                        "expanded_visual_tokens": visual_tokens,
                        "expanded_context_length": int(context["expanded_input_tokens"]),
                        "generated_tokens": None,
                        "measurement_source": "processor_replay_model_boundary",
                    },
                    "raw_decoder_calls": [allocation_raw["decoder"]],
                    "post_adaptation_calls": [allocation_raw["post_resolution_adaptation"]],
                    "cache_state": "mixed",
                    "timing_boundary": "video_decode_plus_autogaze_processor_without_nvila_generation",
                    "processor_replay_wall_seconds_for_shared_video": elapsed,
                    "video_observation_key": replay_key,
                    "compatibility_identity": compatibility_identity,
                },
            )
        video_record_path = args.video_records_dir / f"{video_key}.json"
        write_atomic_json(video_record_path, record)
        existing_by_video[video] = record
        resume_state = load_resume_state(args.evidence_records_dir, compatibility_key)
        del inputs, frames
        gc.collect()
        torch.cuda.empty_cache()

    completed = read_video_records(args.video_records_dir)
    completed_by_video = completed_replay_by_video(completed, compatibility_key)
    expected_selected = set(selected_videos)
    if set(completed_by_video) != expected_selected:
        missing = sorted(expected_selected - set(completed_by_video))
        extra = sorted(set(completed_by_video) - expected_selected)
        raise ValueError(f"Replay coverage mismatch: missing={missing}, extra={extra}")
    covered_question_ids = sorted(
        question_id for record in completed for question_id in record["question_ids"]
    )
    expected_question_ids = sorted(
        int(row["question_id"])
        for video in selected_videos
        for row in rows_by_video[video]
    )
    if covered_question_ids != expected_question_ids:
        raise ValueError("Replay question keys do not exactly cover the selected HLVid rows")
    completed_evidence = load_resume_state(args.evidence_records_dir, compatibility_key)["completed"]
    expected_example_keys = {
        stable_example_key(question_id, "test") for question_id in expected_question_ids
    }
    if set(completed_evidence) != expected_example_keys:
        raise ValueError("Durable replay sidecars do not exactly cover the selected HLVid questions")
    merged_raw = merge_raw_allocations(
        (record["allocation_raw"], int(record["question_count"])) for record in completed
    )
    full_coverage = len(selected_videos) == len(rows_by_video)
    validation = validate_reference(
        completed,
        args.validation_reference_summary,
        args.validated_replay_summary,
    )
    if full_coverage and args.r2d_mode == "variable" and validation.get("status") != "pass":
        raise ValueError("Full variable-budget evidence requires an exact VARIABLE replay validation")
    export_video_records(args.video_records_dir, args.output)
    export_evidence_jsonl(
        args.evidence_records_dir,
        args.evidence_output,
        compatibility_key,
        completed_only=False,
    )
    context_values = [
        context["expanded_input_tokens"]
        for record in completed
        for context in record["expanded_context_by_question"]
    ]
    visual_values = [
        int(record["visual_tokens_per_question"])
        for record in completed
        for _ in range(int(record["question_count"]))
    ]
    weighted_statistics = summarize_raw_allocation(merged_raw)
    summary = {
        **compatibility_identity,
        "compatibility_identity": compatibility_identity,
        "compatibility_key": compatibility_key,
        "execution": execution,
        "record_type": "r2d_hlvid_allocation_replay_summary",
        "status": "complete" if full_coverage else "partial_validation",
        "coverage": {
            "unique_videos_observed": len(completed),
            "unique_videos_expected": len(rows_by_video),
            "question_keys_observed": len(covered_question_ids),
            "question_keys_expected": len(rows),
            "complete": full_coverage,
        },
        "validation": validation,
        "weighted_allocation_raw": merged_raw,
        "weighted_allocation_statistics": weighted_statistics,
        "counters": {
            "decoder_spatial_actions": raw_counter(
                observed_count=merged_raw["decoder"]["frame_observations"],
                observed_sum=merged_raw["decoder"]["spatial_actions_sum"],
                minimum=merged_raw["decoder"]["spatial_actions_min"],
                maximum=merged_raw["decoder"]["spatial_actions_max"],
                unit="actions_per_frame_observation",
            ),
            "retained_patches": raw_counter(
                observed_count=merged_raw["post_resolution_adaptation"]["frame_observations"],
                observed_sum=merged_raw["post_resolution_adaptation"]["retained_patches_sum"],
                minimum=merged_raw["post_resolution_adaptation"]["retained_patches_min"],
                maximum=merged_raw["post_resolution_adaptation"]["retained_patches_max"],
                unit="patches_per_frame_observation",
            ),
            "expanded_visual_tokens": raw_counter(
                observed_count=len(visual_values),
                observed_sum=sum(visual_values),
                minimum=min(visual_values) if visual_values else None,
                maximum=max(visual_values) if visual_values else None,
                unit="tokens_per_question",
            ),
            "expanded_context_tokens": raw_counter(
                observed_count=len(context_values),
                observed_sum=sum(context_values),
                minimum=min(context_values) if context_values else None,
                maximum=max(context_values) if context_values else None,
                unit="tokens_per_question",
            ),
        },
        "expanded_context_tokens": {
            "observed_count": len(context_values),
            "observed_sum": sum(context_values),
            "mean": float(np.mean(context_values)),
            "min": min(context_values),
            "max": max(context_values),
        },
        "visual_tokens": {
            "observed_count": len(visual_values),
            "observed_sum": sum(visual_values),
            "mean": float(np.mean(visual_values)),
            "min": min(visual_values),
            "max": max(visual_values),
        },
        "nvila_generation_calls": 0,
        "qa_source_artifact": {
            "path": str(args.qa_results.resolve()),
            "sha256": sha256_file(args.qa_results),
        },
        "results_jsonl": str(args.output.resolve()),
        "results_jsonl_sha256": sha256_file(args.output),
        "video_records_dir": str(args.video_records_dir.resolve()),
        "evidence_records_dir": str(args.evidence_records_dir.resolve()),
        "evidence_record_count": len(load_resume_state(args.evidence_records_dir, compatibility_key)["records"]),
        "evidence_jsonl": str(args.evidence_output.resolve()),
        "evidence_jsonl_sha256": sha256_file(args.evidence_output),
        "protocol_audit": protocol_audit,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
