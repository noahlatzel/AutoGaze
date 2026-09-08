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
import subprocess
import sys
import time
from collections import OrderedDict
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--attempt-log", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--validation-reference-summary", type=Path)
    parser.add_argument("--validated-replay-summary", type=Path)
    parser.add_argument("--expected-code-commit", required=True)
    return parser.parse_args()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_replay_by_video(
    records: list[dict[str, Any]], expected_identity_sha256: str
) -> dict[str, dict[str, Any]]:
    """Validate completed records before resume skipping or aggregation."""

    completed: dict[str, dict[str, Any]] = {}
    for record in records:
        video = record["video_path"]
        if video in completed:
            raise ValueError(f"Duplicate completed replay video: {video}")
        if record.get("identity_sha256") != expected_identity_sha256:
            raise ValueError(f"Resume identity mismatch for completed video: {video}")
        if record.get("attempt_state") != "completed_processor_observation":
            raise ValueError(f"Non-complete record found in completed replay output: {video}")
        completed[video] = record
    return completed


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
        "allocation_scope": "one_question_agnostic_processor_call_per_unique_video_weighted_by_question_count",
        "nvila_generation_calls": 0,
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "base_seed": args.base_seed,
        "training_seed": args.training_seed,
        "mode": args.r2d_mode,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "protocol": protocol_identity,
        "execution": execution,
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing replay output: {args.output}")
    existing = read_jsonl(args.output) if args.resume else []
    existing_by_video = completed_replay_by_video(existing, identity_sha256)

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
        stable_key_payload = {
            "identity_sha256": identity_sha256,
            "video_path": video,
            "video_sha256": video_sha256,
        }
        replay_key = hashlib.sha256(
            json.dumps(stable_key_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        attempt = {
            "schema_version": SCHEMA_VERSION,
            "replay_key": replay_key,
            "identity_sha256": identity_sha256,
            "video_path": video,
            "video_key": f"hlvid:test:video:{Path(video).as_posix()}",
            "question_ids": [int(row["question_id"]) for row in video_rows],
            "question_keys": [f"hlvid:test:{str(row['question_id']).strip()}" for row in video_rows],
            "attempt_state": "started",
            "started_at_unix": time.time(),
        }
        append_jsonl(args.attempt_log, attempt)

        started = time.perf_counter()
        decodable_count, source_indices = decodable_uniform_indices(video_path, args.num_video_frames)
        frames = runner.load_uniform_frames(video_path, args.num_video_frames)
        if len(frames) != args.num_video_frames:
            raise ValueError(f"Legacy loader did not return {args.num_video_frames} frames for {video}")
        frames_sha256 = frame_content_sha256(frames)
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
            **identity,
            "identity_sha256": identity_sha256,
            "record_type": "r2d_hlvid_allocation_replay_video",
            "replay_key": replay_key,
            "attempt_state": "completed_processor_observation",
            "video_index": video_index,
            "video_path": video,
            "video_key": f"hlvid:test:video:{Path(video).as_posix()}",
            "video_sha256": video_sha256,
            "question_ids": [int(row["question_id"]) for row in video_rows],
            "question_keys": [f"hlvid:test:{str(row['question_id']).strip()}" for row in video_rows],
            "question_count": len(video_rows),
            "source_frames": {
                "decodable_frame_count": decodable_count,
                "requested_source_indices": source_indices,
                "legacy_loaded_frame_count": len(frames),
                "legacy_loaded_frames_sha256": frames_sha256,
            },
            "processor_tensors": tensor_structure(inputs),
            "expanded_context_by_question": contexts,
            "visual_tokens_per_question": visual_tokens,
            "allocation_raw": allocation_raw,
            "allocation_summary": allocation_summary,
            "measurement_source": "replay",
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
        append_jsonl(args.output, record)
        append_jsonl(
            args.attempt_log,
            {
                **attempt,
                "attempt_state": "completed_processor_observation",
                "completed_at_unix": time.time(),
                "processor_replay_wall_seconds": elapsed,
            },
        )
        existing_by_video[video] = record
        del inputs, frames
        gc.collect()
        torch.cuda.empty_cache()

    completed = read_jsonl(args.output)
    completed_by_video = completed_replay_by_video(completed, identity_sha256)
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
    summary = {
        **identity,
        "identity_sha256": identity_sha256,
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
        "weighted_allocation_statistics": summarize_raw_allocation(merged_raw),
        "expanded_context_tokens": {
            "observations": len(covered_question_ids),
            "mean": float(
                np.mean(
                    [
                        context["expanded_input_tokens"]
                        for record in completed
                        for context in record["expanded_context_by_question"]
                    ]
                )
            ),
            "min": min(
                context["expanded_input_tokens"]
                for record in completed
                for context in record["expanded_context_by_question"]
            ),
            "max": max(
                context["expanded_input_tokens"]
                for record in completed
                for context in record["expanded_context_by_question"]
            ),
        },
        "visual_tokens": {
            "observations": len(covered_question_ids),
            "mean": float(
                np.average(
                    [record["visual_tokens_per_question"] for record in completed],
                    weights=[record["question_count"] for record in completed],
                )
            ),
            "min": min(record["visual_tokens_per_question"] for record in completed),
            "max": max(record["visual_tokens_per_question"] for record in completed),
        },
        "nvila_generation_calls": 0,
        "results_jsonl": str(args.output.resolve()),
        "results_jsonl_sha256": sha256_file(args.output),
        "attempt_log": str(args.attempt_log.resolve()),
        "attempt_log_sha256": sha256_file(args.attempt_log),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
