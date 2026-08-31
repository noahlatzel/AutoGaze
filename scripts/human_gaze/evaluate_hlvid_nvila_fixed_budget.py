#!/usr/bin/env python3
"""Evaluate one fixed-budget human-gaze AutoGaze checkpoint on HLVid/NVILA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from autogaze.human_gaze.hlvid import install_fixed_budget_forward

ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-budget", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--continuation-seed", type=int, required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--autogaze-model-id", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--tile-len", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
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


def extract_letter(text: str) -> str:
    match = ANSWER_RE.search(text.strip())
    return match.group(1).upper() if match else ""


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_rows(dataset_root: Path) -> list[dict[str, Any]]:
    parquet_path = dataset_root / "data" / "test-00000-of-00001.parquet"
    rows = pq.read_table(parquet_path).to_pylist()
    ids = [int(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("HLVid question_id values are not unique")
    return rows


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = int(record["question_id"])
            if question_id in records:
                raise ValueError(f"Duplicate question_id={question_id} at {path}:{line_no}")
            records[question_id] = record
    return records


def load_uniform_frames(video_path: Path, num_frames: int) -> tuple[list[Image.Image], list[int], int]:
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
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Video has no decodable frames: {video_path}")

    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(np.int64)
    frames_by_index: dict[int, Image.Image] = {}
    for index in indices:
        value = int(index)
        if value in frames_by_index:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, value)
        ok, frame = capture.read()
        if ok:
            frames_by_index[value] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

    decoded = [frames_by_index[int(index)] for index in indices if int(index) in frames_by_index]
    if not decoded:
        raise ValueError(f"Could not extract frames from video: {video_path}")
    if len(decoded) < num_frames:
        decoded.extend([decoded[-1]] * (num_frames - len(decoded)))
    return decoded, [int(index) for index in indices], frame_count


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
        "by_category": {category: metrics(rows) for category, rows in sorted(by_category.items())},
        "prediction_counts": dict(sorted(Counter(str(row["prediction_letter"]) for row in records).items())),
        "invalid_prediction_count": sum(not str(row["prediction_letter"]) for row in records),
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


def write_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    stats: Any,
    checkpoint_compatibility: dict[str, Any],
) -> None:
    parquet_path = args.dataset_root / "data" / "test-00000-of-00001.parquet"
    checkpoint_path = args.autogaze_model_id / "model.safetensors"
    summary = summarize(records)
    summary.update(
        {
            "schema_version": 2,
            "benchmark": "HLVid",
            "benchmark_split": "official_huggingface_test",
            "human_gaze_split_accessed": False,
            "dataset_root": str(args.dataset_root),
            "dataset_parquet_sha256": sha256(parquet_path),
            "model_path": str(args.model_path),
            "autogaze_model_id": str(args.autogaze_model_id),
            "autogaze_checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_compatibility": checkpoint_compatibility,
            "base_seed": args.base_seed,
            "continuation_seed": args.continuation_seed,
            "policy_label": args.policy_label,
            "protocol": {
                "temporal_sampling": "round(linspace(0, decodable_frame_count-1, 128))",
                "num_video_frames": args.num_video_frames,
                "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
                "max_tiles_video": args.max_tiles_video,
                "tile_len": args.tile_len,
                "max_new_tokens": args.max_new_tokens,
                "thumbnails": "full",
                "image_processor": "saved slow processor (use_fast=false)",
                "generation": "greedy",
                "torch_dtype": args.torch_dtype,
                "max_batch_size_autogaze": args.max_batch_size_autogaze,
                "max_batch_size_siglip": args.max_batch_size_siglip,
                "prompt_template": "<video>\n\nQuestion: {benchmark_question_with_choices_and_answer_instruction}",
                "answer_parser": "first standalone A/B/C/D letter",
            },
            "fixed_budget_adapter": {
                "exact_decoder_actions_per_autogaze_frame": args.exact_budget,
                "fine_action_offset": args.fine_action_offset,
                "actions_per_frame": args.actions_per_frame,
                "allowed_action_ids": [args.fine_action_offset, args.actions_per_frame - 1],
                "allow_eos": False,
                "post_resolution_adaptation_current_process": stats.as_dict(),
            },
            "code_commit": git_commit(),
            "results_jsonl": str(args.output),
            "results_jsonl_sha256": sha256(args.output),
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
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
    if not (args.autogaze_model_id / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete AutoGaze checkpoint: {args.autogaze_model_id}")

    parquet_path = args.dataset_root / "data" / "test-00000-of-00001.parquet"
    observed_dataset_sha256 = sha256(parquet_path)
    if observed_dataset_sha256 != args.expected_dataset_sha256:
        raise ValueError(
            f"HLVid parquet checksum mismatch: {observed_dataset_sha256} != "
            f"{args.expected_dataset_sha256}"
        )
    rows = load_rows(args.dataset_root)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing = load_existing(args.output) if args.resume else {}
    expected_ids = {int(row["question_id"]) for row in rows}
    unexpected = set(existing) - expected_ids
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
    checkpoint_compatibility = verify_loaded_checkpoint(
        processor._autogaze_model,
        args.autogaze_model_id / "model.safetensors",
    )
    stats = install_fixed_budget_forward(
        processor._autogaze_model,
        exact_budget=args.exact_budget,
        fine_action_offset=args.fine_action_offset,
        actions_per_frame=args.actions_per_frame,
    )
    model = AutoModel.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        device_map=args.device_map,
        torch_dtype=dtype_from_name(args.torch_dtype),
        max_batch_size_siglip=args.max_batch_size_siglip,
    )
    model.eval()
    video_token = processor.tokenizer.video_token

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode) as output:
        for position, row in enumerate(rows, start=1):
            question_id = int(row["question_id"])
            if question_id in existing:
                continue
            video_path = args.dataset_root / "videos" / str(row["video_path"])
            frames, sampled_indices, source_frame_count = load_uniform_frames(
                video_path, args.num_video_frames
            )
            prompt = f"{video_token}\n\nQuestion: {row['question']}"
            inputs = processor(text=prompt, videos=[frames], return_tensors="pt")
            target_device = model.device if hasattr(model, "device") else next(model.parameters()).device
            inputs = {
                key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            response = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0].strip()
            prediction = extract_letter(response)
            answer = str(row["answer"]).strip().upper()
            record = {
                "question_id": question_id,
                "category": str(row["category"]),
                "video_path": str(row["video_path"]),
                "answer": answer,
                "prediction": response,
                "prediction_letter": prediction,
                "is_correct": prediction == answer,
                "source_frame_count": source_frame_count,
                "sampled_source_indices": sampled_indices,
                "num_sampled_frames": len(sampled_indices),
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            existing[question_id] = record
            print(
                f"[{position}/{len(rows)}] qid={question_id} pred={prediction or '-'} "
                f"answer={answer} correct={record['is_correct']}",
                flush=True,
            )

    ordered = [existing[int(row["question_id"])] for row in rows]
    write_summary(args, ordered, stats, checkpoint_compatibility)


if __name__ == "__main__":
    main()
