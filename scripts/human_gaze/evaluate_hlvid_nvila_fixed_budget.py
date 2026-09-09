#!/usr/bin/env python3
"""Evaluate an exact-budget spatial policy on HLVid with NVILA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.hlvid import install_fixed_budget_forward, load_uniform_frames
from scripts.runners.hlvid_evidence import (
    counter_summary,
    export_evidence_jsonl,
    load_resume_state,
    merge_counter_summaries,
    resume_compatibility_key,
    stable_example_key,
    stable_video_key,
    write_evidence_record,
)

ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
PROTOCOL_ID = "hlvid_nvila_mtv48_uniform128_thumbnail64_v1"
NVILA_SNAPSHOT_REVISION = "7a5670e20da435d98b0efdc49f9a536f73985152"
CENTER16_FINE_CELLS = tuple(center_order(14)[:16].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-budget", type=int, required=True)
    parser.add_argument("--policy-kind", choices=("trained", "pretrained", "center16"), default="trained")
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--continuation-seed", type=int)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--autogaze-model-id", type=Path, required=True)
    parser.add_argument("--decode-audit-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--event-output", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--tile-len", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--fine-action-offset", type=int, default=69)
    parser.add_argument("--actions-per-frame", type=int, default=265)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_letter(text: str) -> str:
    match = ANSWER_RE.search(text.strip())
    return match.group(1).upper() if match else ""


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_rows(dataset_root: Path) -> list[dict[str, Any]]:
    parquet_path = dataset_root / "data" / "test-00000-of-00001.parquet"
    rows = pq.read_table(parquet_path).to_pylist()
    ids = [int(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("HLVid question_id values are not unique")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from error
    return records


def load_completed_evidence(
    path: Path,
    *,
    compatibility_key: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    answers: dict[int, dict[str, Any]] = {}
    telemetry: dict[int, dict[str, Any]] = {}
    state = load_resume_state(path, compatibility_key)
    for example_key, event in state["completed"].items():
        answer = event["answer"]
        question_id = int(answer["question_id"])
        if example_key != stable_example_key(question_id, "test"):
            raise ValueError(f"Stable example key mismatch for question_id={question_id}")
        if question_id in answers:
            raise ValueError(f"Duplicate completed question_id={question_id} in {path}")
        answers[question_id] = answer
        telemetry[question_id] = {
            "schema_version": event["schema_version"],
            "qa_key": example_key,
            "question_id": question_id,
            "protocol_id": event["protocol_id"],
            "run_identity_sha256": compatibility_key,
            "decode": event.get("decode", {}),
            "counters": event.get("counters", {}),
            "context": event.get("context", {}),
            "raw_decoder_calls": event.get("raw_decoder_calls", []),
            "post_adaptation_calls": event.get("post_adaptation_calls", []),
            "timing": event.get("timing", {}),
            "cache_state": event.get("cache_state", "unknown"),
            "process_warmup_state": event.get("process_warmup_state", "unknown"),
            "video_frame_cache_state": event.get("video_frame_cache_state", "unknown"),
        }
    return answers, telemetry


def load_compact_results(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line_no, record in enumerate(read_jsonl(path), start=1):
        question_id = int(record["question_id"])
        if question_id in records:
            raise ValueError(f"Duplicate question_id={question_id} at {path}:{line_no}")
        records[question_id] = record
    return records


def validate_compact_results(
    compact: dict[int, dict[str, Any]],
    completed: dict[int, dict[str, Any]],
) -> None:
    """Accept a stale derived compact file, but never evidence absent from the log."""
    for question_id, record in compact.items():
        if question_id not in completed:
            raise ValueError(
                f"Compact results contain question_id={question_id} without durable evidence"
            )
        if completed[question_id] != record:
            raise ValueError(
                f"Compact result differs from durable evidence for question_id={question_id}"
            )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_results_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[str(record["video_path"])].append(record)
        by_category[str(record["category"])].append(record)

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(row["is_correct"]) for row in rows)
        return {"num_examples": len(rows), "num_correct": correct, "accuracy": correct / len(rows)}

    video_accuracies = {
        video: metrics(rows)["accuracy"] for video, rows in sorted(by_video.items())
    }
    return {
        **metrics(records),
        "macro_video_accuracy": float(np.mean(list(video_accuracies.values()))),
        "num_videos": len(video_accuracies),
        "by_category": {
            category: metrics(rows) for category, rows in sorted(by_category.items())
        },
        "prediction_counts": dict(
            sorted(Counter(str(row["prediction_letter"]) for row in records).items())
        ),
        "invalid_prediction_count": sum(
            not str(row["prediction_letter"]) for row in records
        ),
    }


def numeric_summary(
    values: list[float | int], *, unit: str, source: str = "original_forward"
) -> dict[str, Any]:
    return counter_summary(
        values,
        expected_observations=len(values),
        unit=unit,
        source=source,
    )


def normalize_acquisition(acquisition: dict[str, Any]) -> dict[str, Any]:
    """Translate adapter traces to the shared WP0 evidence vocabulary."""

    raw_calls = []
    post_calls = []
    raw_spatial_counts: list[int] = []
    post_valid_counts: list[int] = []
    post_padded_lengths: list[int] = []
    selector_seconds: list[float] = []
    autogaze_seconds: list[float] = []
    for call in acquisition["calls"]:
        decoder_action_ids = call.get("decoder_action_ids", [])
        raw_spatial_counts.extend(
            len(actions)
            for batch in decoder_action_ids
            for actions in batch
        )
        raw_calls.append(
            {
                key: call[key]
                for key in (
                    "decoder_batch_size",
                    "decoder_frames_per_item",
                    "decoder_padded_lengths_per_frame",
                    "decoder_action_ids",
                    "selector_seconds",
                )
                if key in call
            }
        )
        valid_counts = call["post_recovery_valid_counts_per_item_frame"]
        padded_lengths = call["post_recovery_padded_lengths_per_frame"]
        post_valid_counts.extend(
            int(value) for sample in valid_counts for value in sample
        )
        post_padded_lengths.extend(
            int(value)
            for _sample in valid_counts
            for value in padded_lengths
        )
        post_calls.append(
            {
                "valid_counts_per_sample_frame": valid_counts,
                "padded_slots_per_frame": padded_lengths,
                "autogaze_total_seconds": call["autogaze_total_seconds"],
                "autogaze_nonselector_seconds": call.get("autogaze_nonselector_seconds"),
            }
        )
        selector_seconds.append(float(call.get("selector_seconds", 0.0)))
        autogaze_seconds.append(float(call["autogaze_total_seconds"]))
    return {
        "raw_decoder_calls": raw_calls,
        "post_adaptation_calls": post_calls,
        "counters": {
            "raw_decoder_spatial_actions_per_tile_frame": numeric_summary(
                raw_spatial_counts, unit="actions_per_tile_frame"
            ),
            "post_adaptation_valid_patches_per_tile_frame": numeric_summary(
                post_valid_counts, unit="patches_per_tile_frame"
            ),
            "post_adaptation_padded_slots_per_tile_frame": numeric_summary(
                post_padded_lengths, unit="slots_per_tile_frame"
            ),
            "selector_seconds_per_processor_call": numeric_summary(
                selector_seconds, unit="seconds"
            ),
            "autogaze_seconds_per_processor_call": numeric_summary(
                autogaze_seconds, unit="seconds"
            ),
        },
    }


def summarize_telemetry(records: list[dict[str, Any]]) -> dict[str, Any]:
    context_fields = {
        "expanded_context_length": [],
        "expanded_visual_tokens": [],
        "text_and_special_tokens": [],
        "tokenizer_input_length": [],
        "generated_tokens": [],
    }
    timing_fields = {
        "decode_seconds": [],
        "processor_seconds": [],
        "selector_seconds": [],
        "autogaze_total_seconds": [],
        "nvila_vision_encoder_projector_seconds": [],
        "generation_total_seconds": [],
        "language_prefill_and_decode_seconds": [],
        "cuda_peak_memory_allocated_bytes": [],
    }
    acquisition_counters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for key in context_fields:
            value = record.get("context", {}).get(key)
            if value is not None:
                context_fields[key].append(value)
        for key in timing_fields:
            value = record.get("timing", {}).get(key)
            if value is not None:
                timing_fields[key].append(value)
        for key, value in record.get("counters", {}).items():
            acquisition_counters[key].append(value)
    return {
        "context": {
            key: numeric_summary(values, unit="tokens")
            for key, values in context_fields.items()
        },
        "timing": {
            key: numeric_summary(
                values,
                unit="bytes" if key == "cuda_peak_memory_allocated_bytes" else "seconds",
            )
            for key, values in timing_fields.items()
        },
        "acquisition": {
            key: merge_counter_summaries(values)
            for key, values in sorted(acquisition_counters.items())
        },
    }


def checkpoint_tensor_keys(checkpoint_path: Path) -> set[str]:
    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        return set(handle.keys())


def verify_loaded_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    """Accept only the known exact-zero buffer absent from historical checkpoints."""
    runtime_state = model.state_dict()
    runtime_keys = set(runtime_state)
    saved_keys = checkpoint_tensor_keys(checkpoint_path)
    missing = sorted(runtime_keys - saved_keys)
    unexpected = sorted(saved_keys - runtime_keys)
    bias_name = "gazing_model.gaze_decoder.output_token_logit_bias"
    if missing != [bias_name] or unexpected:
        raise ValueError(
            "Checkpoint/runtime tensor schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    bias = runtime_state[bias_name]
    if torch.count_nonzero(bias).item() != 0:
        raise ValueError(f"Historical-checkpoint compatibility buffer is not zero: {bias_name}")
    return {
        "accepted_runtime_only_keys": missing,
        "runtime_only_key_initialization": "registered exact-zero buffer",
        "max_absolute_value": float(bias.abs().max().item()),
        "saved_tensor_count": len(saved_keys),
        "runtime_tensor_count": len(runtime_keys),
        "unexpected_saved_keys": unexpected,
    }


def file_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str | None]:
    return {
        name: sha256(root / name) if (root / name).is_file() else None
        for name in names
    }


def load_decode_audit(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if args.decode_audit_manifest is None:
        return None
    audit_root = (
        args.decode_audit_manifest
        if args.decode_audit_manifest.is_dir()
        else args.decode_audit_manifest.parent
    )
    runtime_path = audit_root / "protocol_runtime_manifest.json"
    summary_path = audit_root / "summary.json"
    records_path = audit_root / "decode_audit.jsonl"
    supplement_path = audit_root / "live_runtime_supplement.json"
    runtime = json.loads(runtime_path.read_text())
    summary = json.loads(summary_path.read_text())
    supplement = json.loads(supplement_path.read_text())
    if runtime.get("dataset_sha256") != args.expected_dataset_sha256:
        raise ValueError("Decode audit dataset mismatch")
    expected_protocol = {
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "tile_len": 16,
        "nvila_dtype": "bfloat16",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 16,
        "truncation": False,
    }
    if runtime.get("protocol") != expected_protocol:
        raise ValueError("Decode audit protocol mismatch")
    if not runtime.get("loader_source_verified") or not summary.get("loader_source_verified"):
        raise ValueError("Decode audit did not verify the frozen legacy loader")
    if summary.get("reuse_qualification") != "no_current_requested_decode_failures":
        raise ValueError(
            "Decode audit requires affected-subset inspection before control admission"
        )
    records = read_jsonl(records_path)
    expected_videos = sorted({str(row["video_path"]) for row in rows})
    observed_videos = sorted(str(record["manifest_path"]) for record in records)
    if observed_videos != expected_videos:
        raise ValueError("Decode audit video inventory mismatch")
    expected_supplement_hashes = {
        "protocol_runtime_manifest_sha256": sha256(runtime_path),
        "summary_sha256": sha256(summary_path),
        "decode_audit_jsonl_sha256": sha256(records_path),
    }
    if supplement.get("audit_output_hashes") != expected_supplement_hashes:
        raise ValueError("Decode audit live-runtime supplement hash mismatch")
    required_packages = {"opencv", "pillow", "numpy", "pyarrow"}
    packages = supplement.get("packages")
    if (
        not isinstance(packages, dict)
        or not required_packages.issubset(packages)
        or any(not isinstance(packages[name], str) or not packages[name] for name in required_packages)
        or not isinstance(supplement.get("hostname"), str)
        or not supplement["hostname"]
        or not isinstance(supplement.get("git_dirty"), bool)
    ):
        raise ValueError("Decode audit live-runtime supplement is incomplete")
    return {
        "root": str(audit_root.resolve()),
        "protocol_runtime_manifest_sha256": sha256(runtime_path),
        "summary_sha256": sha256(summary_path),
        "decode_audit_jsonl_sha256": sha256(records_path),
        "live_runtime_supplement_sha256": sha256(supplement_path),
        "live_runtime": {
            "hostname": supplement["hostname"],
            "packages": packages,
            "git_commit": supplement.get("git_commit"),
            "git_dirty": supplement["git_dirty"],
        },
        "reuse_qualification": summary["reuse_qualification"],
        "historical_certification": summary["historical_certification"],
        "video_count": summary["video_count"],
        "question_count": summary["question_count"],
        "live_model_files_verified": runtime["live_model_files_verified"],
    }


class NVILATimingTrace:
    """Time the vision encoder/projector inside generation without remote edits."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.seconds = 0.0
        self.calls = 0
        original = model._encode_vision

        def timed_encode(_model, *args, **kwargs):
            device = _model.vision_tower.device
            if torch.cuda.is_available() and torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = original(*args, **kwargs)
            if torch.cuda.is_available() and torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            self.seconds += time.perf_counter() - started
            self.calls += 1
            return result

        model._encode_vision = MethodType(timed_encode, model)

    def start_example(self) -> None:
        self.seconds = 0.0
        self.calls = 0

    def finish_example(self) -> dict[str, Any]:
        return {"seconds": self.seconds, "calls": self.calls}


def validate_args(args: argparse.Namespace) -> None:
    exact_protocol = {
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "tile_len": 16,
        "max_new_tokens": 16,
        "max_batch_size_autogaze": 16,
        "max_batch_size_siglip": 32,
        "torch_dtype": "bfloat16",
        "fine_action_offset": 69,
        "actions_per_frame": 265,
    }
    drift = {
        key: {"observed": getattr(args, key), "expected": expected}
        for key, expected in exact_protocol.items()
        if getattr(args, key) != expected
    }
    if drift:
        raise ValueError(f"Established HLVid/NVILA protocol drift: {drift}")
    if args.exact_budget not in {16, 24, 32, 36}:
        raise ValueError(f"Unsupported fixed budget: {args.exact_budget}")
    if args.policy_kind in {"pretrained", "center16"} and args.exact_budget != 16:
        raise ValueError("Causal controls are frozen at exact K16")
    if args.policy_kind == "trained" and (args.base_seed is None or args.continuation_seed is None):
        raise ValueError("Trained policies require both seed identifiers")
    if not (args.autogaze_model_id / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete AutoGaze checkpoint: {args.autogaze_model_id}")


def build_identity(
    args: argparse.Namespace,
    *,
    dataset_sha256: str,
    decode_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    checkpoint_path = args.autogaze_model_id / "model.safetensors"
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "benchmark": "HLVid",
        "benchmark_split": "official_huggingface_test",
        "dataset_parquet_sha256": dataset_sha256,
        "policy_kind": args.policy_kind,
        "policy_label": args.policy_label,
        "base_seed": args.base_seed,
        "continuation_seed": args.continuation_seed,
        "exact_budget": args.exact_budget,
        "center16_fine_cells": list(CENTER16_FINE_CELLS) if args.policy_kind == "center16" else None,
        "center16_action_ids": [args.fine_action_offset + value for value in CENTER16_FINE_CELLS]
        if args.policy_kind == "center16"
        else None,
        "autogaze_model_id": str(args.autogaze_model_id.resolve()),
        "autogaze_checkpoint_sha256": sha256(checkpoint_path),
        "autogaze_artifact_hashes": file_hashes(
            args.autogaze_model_id, ("config.json", "preprocessor_config.json")
        ),
        "nvila_model_path": str(args.model_path.resolve()),
        "nvila_snapshot_revision": NVILA_SNAPSHOT_REVISION,
        "nvila_artifact_hashes": file_hashes(
            args.model_path,
            (
                "config.json",
                "generation_config.json",
                "modeling_nvila.py",
                "preprocessor_config.json",
                "processing_nvila.py",
                "processor_config.json",
                "pytorch_model.bin.index.json",
                "tokenizer_config.json",
            ),
        ),
        "decode_audit": decode_audit,
        "inference_code_hashes": {
            "evaluator_sha256": sha256(Path(__file__)),
            "fixed_budget_adapter_sha256": sha256(
                Path(__file__).parents[2] / "autogaze" / "human_gaze" / "hlvid.py"
            ),
        },
        "protocol": {
            "temporal_sampling": "round(linspace(0, decodable_frame_count-1, 128))",
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "max_tiles_video": args.max_tiles_video,
            "tile_len": args.tile_len,
            "max_new_tokens": args.max_new_tokens,
            "thumbnails": "64 full uniformly sampled thumbnail frames",
            "image_processor": "saved slow processor (use_fast=false)",
            "generation": "greedy; do_sample=false",
            "torch_dtype": args.torch_dtype,
            "max_batch_size_autogaze": args.max_batch_size_autogaze,
            "max_batch_size_siglip": args.max_batch_size_siglip,
            "prompt_template": "<video>\\n\\nQuestion: {benchmark_question_with_choices_and_answer_instruction}",
            "answer_parser": "first standalone A/B/C/D letter",
            "answer_parser_regex": ANSWER_RE.pattern,
            "scoring": "exact equality after uppercasing benchmark answer and parsed option",
            "context_policy": "upstream long-context recipe; no evaluator truncation",
            "fixed_fine_only": True,
            "allow_eos": False,
        },
    }


def ensure_manifest(path: Path, identity: dict[str, Any], *, resume: bool) -> str:
    identity_hash = resume_compatibility_key(identity)
    manifest = {
        "schema_version": 1,
        "compatibility_key": identity_hash,
        "inference_identity": identity,
        "instrumentation": {
            "code_commit": git_commit(),
            "code_dirty": git_dirty(),
        },
    }
    if path.exists():
        observed = json.loads(path.read_text())
        if (
            observed.get("compatibility_key") != identity_hash
            or observed.get("inference_identity") != identity
        ):
            raise ValueError(f"Resume-incompatible run manifest: {path}")
    else:
        if not resume and path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError(f"Refusing non-resume run in non-empty directory: {path.parent}")
        write_json_atomic(path, manifest)
    return identity_hash


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "numpy", "pyarrow", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "packages": packages,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    parquet_path = args.dataset_root / "data" / "test-00000-of-00001.parquet"
    dataset_sha256 = sha256(parquet_path)
    if dataset_sha256 != args.expected_dataset_sha256:
        raise ValueError(
            f"HLVid parquet checksum mismatch: {dataset_sha256} != {args.expected_dataset_sha256}"
        )
    full_rows = load_rows(args.dataset_root)
    decode_audit = load_decode_audit(args, full_rows)
    rows = full_rows[: args.limit] if args.limit is not None else full_rows

    args.evidence_dir = args.evidence_dir or args.output.with_name("evidence_records")
    args.event_output = args.event_output or args.output.with_name("events.jsonl")
    args.run_manifest = args.run_manifest or args.output.with_name("run_manifest.json")
    identity = build_identity(args, dataset_sha256=dataset_sha256, decode_audit=decode_audit)
    identity_hash = ensure_manifest(args.run_manifest, identity, resume=args.resume)

    completed, telemetry = load_completed_evidence(
        args.evidence_dir, compatibility_key=identity_hash
    ) if args.resume else ({}, {})
    compact = load_compact_results(args.output) if args.resume else {}
    validate_compact_results(compact, completed)
    expected_ids = {int(row["question_id"]) for row in rows}
    unexpected = set(completed) - expected_ids
    if unexpected:
        raise ValueError(f"Resume output has unexpected question IDs: {sorted(unexpected)[:10]}")

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(args.model_path),
        autogaze_model_id=str(args.autogaze_model_id),
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
        gazing_ratio_tile=[0.2] + [0.06] * 15,
        gazing_ratio_thumbnail=1,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=None,
        max_batch_size_autogaze=args.max_batch_size_autogaze,
        trust_remote_code=True,
        use_fast=False,
    )
    if processor._autogaze_model.training:
        raise ValueError("AutoGaze must be in evaluation mode")
    checkpoint_compatibility = verify_loaded_checkpoint(
        processor._autogaze_model,
        args.autogaze_model_id / "model.safetensors",
    )
    static_cells = CENTER16_FINE_CELLS if args.policy_kind == "center16" else None
    stats = install_fixed_budget_forward(
        processor._autogaze_model,
        exact_budget=args.exact_budget,
        fine_action_offset=args.fine_action_offset,
        actions_per_frame=args.actions_per_frame,
        static_fine_cells=static_cells,
    )
    model = AutoModel.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        device_map=args.device_map,
        torch_dtype=dtype_from_name(args.torch_dtype),
        max_batch_size_siglip=args.max_batch_size_siglip,
    )
    model.eval()
    nvila_timing = NVILATimingTrace(model)
    video_token = processor.tokenizer.video_token
    target_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    generated_in_process = 0

    for position, row in enumerate(rows, start=1):
        question_id = int(row["question_id"])
        if question_id in completed:
            continue
        qa_key = stable_example_key(question_id, "test")
        video_key = stable_video_key(str(row["video_path"]), "test")
        attempt_id = str(uuid.uuid4())
        common_record = {
            "schema_version": 1,
            "example_key": qa_key,
            "video_key": video_key,
            "attempt_id": attempt_id,
            "compatibility_key": identity_hash,
            "question_id": question_id,
            "question_index": position - 1,
            "split": "test",
            "policy_label": args.policy_label,
            "protocol_id": PROTOCOL_ID,
        }
        write_evidence_record(
            args.evidence_dir,
            {**common_record, "state": "attempt_started", "recorded_at": utc_now()},
        )
        try:
            if torch.cuda.is_available() and torch.device(target_device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(target_device)

            video_path = args.dataset_root / "videos" / str(row["video_path"])
            started = time.perf_counter()
            frames, decode_trace = load_uniform_frames(
                video_path, args.num_video_frames, strict=True
            )
            decode_seconds = time.perf_counter() - started
            prompt = f"{video_token}\n\nQuestion: {row['question']}"
            stats.start_example()
            started = time.perf_counter()
            inputs = processor(text=prompt, videos=[frames], return_tensors="pt")
            processor_seconds = time.perf_counter() - started
            acquisition = normalize_acquisition(stats.finish_example())
            input_ids = inputs["input_ids"]
            total_context_tokens = int(input_ids.shape[1])
            projected_visual_tokens = int(
                (input_ids == model.config.video_token_id).sum().item()
            )
            input_shapes = {
                key: list(value.shape)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            inputs = {
                key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            nvila_timing.start_example()
            if torch.cuda.is_available() and torch.device(target_device).type == "cuda":
                torch.cuda.synchronize(target_device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            if torch.cuda.is_available() and torch.device(target_device).type == "cuda":
                torch.cuda.synchronize(target_device)
            generation_seconds = time.perf_counter() - started
            vision_timing = nvila_timing.finish_example()
            generated_tokens = int(generated.shape[1] - inputs["input_ids"].shape[1])
            response = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0].strip()
            prediction = extract_letter(response)
            answer = str(row["answer"]).strip().upper()
            answer_record = {
                "schema_version": 3,
                "qa_key": qa_key,
                "question_id": question_id,
                "category": str(row["category"]),
                "video_path": str(row["video_path"]),
                "answer": answer,
                "prediction": response,
                "prediction_letter": prediction,
                "is_correct": prediction == answer,
                "run_identity_sha256": identity_hash,
            }
            trace = decode_trace.as_dict()
            decode_record = {
                "intended_indices": trace["requested_indices"],
                "effective_indices": trace["requested_indices"],
                "sampled_read_failure": bool(trace["failed_indices"]),
                "legacy_output_substituted": not trace["exact"],
                "decode_usable": True,
                "read_results": trace["reads"],
                "source_frame_count": trace["source_frame_count"],
                "reported_index_mismatches": trace["mismatched_indices"],
                "tail_padding_count": trace["padded_output_frames"],
            }
            context = {
                "tokenizer_input_length": None,
                "expanded_visual_tokens": projected_visual_tokens,
                "expanded_context_length": total_context_tokens,
                "text_and_special_tokens": total_context_tokens - projected_visual_tokens,
                "generated_tokens": generated_tokens,
                "measurement_source": "NVILA model-boundary input_ids",
                "tokenizer_model_max_length": int(processor.tokenizer.model_max_length),
                "nvila_nominal_max_position_embeddings": int(
                    model.config.text_config.max_position_embeddings
                ),
                "context_truncated": False,
                "input_tensor_shapes": input_shapes,
            }
            timing = {
                "decode_seconds": decode_seconds,
                "processor_seconds": processor_seconds,
                "selector_seconds": acquisition["counters"][
                    "selector_seconds_per_processor_call"
                ]["observed_sum"],
                "autogaze_total_seconds": acquisition["counters"][
                    "autogaze_seconds_per_processor_call"
                ]["observed_sum"],
                "nvila_vision_encoder_projector_seconds": vision_timing["seconds"],
                "nvila_vision_encoder_projector_calls": vision_timing["calls"],
                "generation_total_seconds": generation_seconds,
                "language_prefill_and_decode_seconds": max(
                    0.0, generation_seconds - vision_timing["seconds"]
                ),
                "cuda_peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(target_device)
                )
                if torch.cuda.is_available() and torch.device(target_device).type == "cuda"
                else None,
            }
            process_warmup_state = "cold" if generated_in_process == 0 else "warm"
            completed_record = {
                **common_record,
                "state": "completed_answer",
                "recorded_at": utc_now(),
                "measurement_origin": "original_forward",
                "answer": answer_record,
                "decode": decode_record,
                "counters": acquisition["counters"],
                "context": context,
                "raw_decoder_calls": acquisition["raw_decoder_calls"],
                "post_adaptation_calls": acquisition["post_adaptation_calls"],
                "timing": timing,
                "cache_state": "unknown",
                "process_warmup_state": process_warmup_state,
                "video_frame_cache_state": "not_applicable_frames_redecoded_each_question",
                "timing_boundary": (
                    "decode, processor/acquisition, NVILA vision encoder/projector, "
                    "and language prefill/decode measured separately"
                ),
            }
            write_evidence_record(args.evidence_dir, completed_record)
            generated_in_process += 1
            example_telemetry = {
                "schema_version": 1,
                "qa_key": qa_key,
                "question_id": question_id,
                "protocol_id": PROTOCOL_ID,
                "run_identity_sha256": identity_hash,
                "decode": decode_record,
                "counters": acquisition["counters"],
                "context": context,
                "raw_decoder_calls": acquisition["raw_decoder_calls"],
                "post_adaptation_calls": acquisition["post_adaptation_calls"],
                "timing": timing,
                "cache_state": "unknown",
                "process_warmup_state": process_warmup_state,
                "video_frame_cache_state": "not_applicable_frames_redecoded_each_question",
            }
        except Exception as error:
            write_evidence_record(
                args.evidence_dir,
                {
                    **common_record,
                    "state": "failed",
                    "recorded_at": utc_now(),
                    "measurement_origin": "original_forward",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
            )
            export_evidence_jsonl(
                args.evidence_dir,
                args.event_output,
                identity_hash,
            )
            raise
        completed[question_id] = answer_record
        telemetry[question_id] = example_telemetry
        export_evidence_jsonl(
            args.evidence_dir,
            args.event_output,
            identity_hash,
        )
        write_results_atomic(
            args.output,
            [completed[int(item["question_id"])] for item in rows if int(item["question_id"]) in completed],
        )
        print(
            f"[{position}/{len(rows)}] qid={question_id} pred={prediction or '-'} "
            f"answer={answer} correct={answer_record['is_correct']} context={total_context_tokens}",
            flush=True,
        )

    ordered = [completed[int(row["question_id"])] for row in rows]
    ordered_telemetry = [telemetry[int(row["question_id"])] for row in rows]
    write_results_atomic(args.output, ordered)
    summary = summarize(ordered)
    summary.update(
        {
            "schema_version": 3,
            "run_identity_sha256": identity_hash,
            "compatibility_key": identity_hash,
            "run_manifest": str(args.run_manifest),
            "run_manifest_sha256": sha256(args.run_manifest),
            "benchmark": "HLVid",
            "benchmark_split": "official_huggingface_test",
            "human_gaze_split_accessed": False,
            "policy_kind": args.policy_kind,
            "policy_label": args.policy_label,
            "base_seed": args.base_seed,
            "continuation_seed": args.continuation_seed,
            "dataset_root": str(args.dataset_root),
            "dataset_parquet_sha256": dataset_sha256,
            "model_path": str(args.model_path),
            "autogaze_model_id": str(args.autogaze_model_id),
            "autogaze_checkpoint_sha256": sha256(args.autogaze_model_id / "model.safetensors"),
            "checkpoint_compatibility": checkpoint_compatibility,
            "fixed_budget_adapter": {
                "exact_decoder_actions_per_autogaze_frame": args.exact_budget,
                "fine_action_offset": args.fine_action_offset,
                "actions_per_frame": args.actions_per_frame,
                "allowed_action_ids": [args.fine_action_offset, args.actions_per_frame - 1],
                "allow_eos": False,
                "center16_fine_cells": list(CENTER16_FINE_CELLS)
                if args.policy_kind == "center16"
                else None,
                "center16_injection_boundary": "tile-local generation before resolution recovery"
                if args.policy_kind == "center16"
                else None,
                "static_skips_learned_generation_and_probability_rescoring": args.policy_kind
                == "center16",
                "legacy_process_accumulator_diagnostic_only": stats.as_dict(),
            },
            "per_example_telemetry": summarize_telemetry(ordered_telemetry),
            "evidence_records": str(args.evidence_dir),
            "evidence_record_count": len(
                load_resume_state(args.evidence_dir, identity_hash)["records"]
            ),
            "event_output": str(args.event_output),
            "event_output_sha256": sha256(args.event_output),
            "results_jsonl": str(args.output),
            "results_jsonl_sha256": sha256(args.output),
            "code_commit": git_commit(),
            "code_dirty": git_dirty(),
            "environment": environment_manifest(),
        }
    )
    write_json_atomic(args.summary_output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
