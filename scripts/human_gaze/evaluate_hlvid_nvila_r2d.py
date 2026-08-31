#!/usr/bin/env python3
"""Evaluate one R2d seed on HLVid under variable EOS or forced K16."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from autogaze.human_gaze.r2d_hlvid import install_r2d_hlvid_forward

EXPECTED_RUNNER_SHA256 = "f7281533b556e508b6c82fc419e66a138d93e151be68932a7a632c8585c6de1e"
EXPECTED_COMMON_SHA256 = "d390fd54b8fca681e2d530c3fb53ee15fdddcb06c379ef86d7720a914949e0b4"
EXPECTED_DATASET_SHA256 = "ed2a1a47603fd19f5d7fae0db4f9792369f5aff6548f35e0225c3b4bd35e112a"

ESTABLISHED_PROTOCOL = {
    "dataset_split": "test",
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


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--r2d-mode", choices=["variable", "forced_k16"], required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--checkpoint-model-sha256", required=True)
    parser.add_argument("--checkpoint-config-sha256", required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--eos-logit-bias", type=float, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--legacy-runner-dir",
        default="/home/stud/latn/AutoGaze/scripts/runners",
    )
    return parser.parse_known_args()


def validate_established_protocol(args: argparse.Namespace, *, preflight: bool) -> None:
    actual = {
        "frame_source": args.frame_source,
        "num_video_frames": args.num_video_frames,
        "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
        "max_tiles_video": args.max_tiles_video,
        "tile_len": args.tile_len,
        "delta_gap": args.delta_gap,
        "max_new_tokens": args.max_new_tokens,
    }
    mismatches = {
        key: {"expected": ESTABLISHED_PROTOCOL[key], "actual": value}
        for key, value in actual.items()
        if value != ESTABLISHED_PROTOCOL[key]
    }
    if mismatches:
        raise ValueError(f"HLVid established-protocol mismatch: {mismatches}")
    if args.start_index != 0:
        raise ValueError("Established full HLVid evaluation must start at index 0")
    expected_limit = 1 if preflight else None
    if args.limit != expected_limit:
        raise ValueError(f"Expected --limit {expected_limit!r} for this run, got {args.limit!r}")
    variants = {
        "delta_policy_checkpoint": args.delta_policy_checkpoint,
        "selected_indices": args.selected_indices,
        "max_source_tiles": args.max_source_tiles,
    }
    active = {key: value for key, value in variants.items() if value is not None}
    if active:
        raise ValueError(f"Established uniform protocol forbids variant selectors: {active}")


def validate_calibration(
    calibration_path: Path,
    *,
    expected_sha256: str,
    checkpoint: Path,
    eos_logit_bias: float,
) -> dict[str, Any]:
    if sha256_file(calibration_path) != expected_sha256:
        raise ValueError("R2d calibration/diagnostic SHA-256 mismatch")
    record = json.loads(calibration_path.read_text())
    calibration = record.get("calibration") or {}
    expected = {
        "split": "train_source_balanced_fixed_subset",
        "clips_per_source": 8,
        "seed": 200826,
        "target_mean_tokens": 16.0,
    }
    mismatches = {
        key: {"expected": value, "actual": calibration.get(key)}
        for key, value in expected.items()
        if calibration.get(key) != value
    }
    if mismatches:
        raise ValueError(f"R2d EOS calibration mapping mismatch: {mismatches}")
    if len(calibration.get("clip_ids") or []) != 48:
        raise ValueError("R2d calibration must contain the fixed 48-clip source-balanced subset")
    if float(calibration.get("selected_bias")) != float(eos_logit_bias):
        raise ValueError("Requested EOS bias does not match the immutable calibration record")
    if Path(str(record.get("checkpoint", ""))).name != checkpoint.name:
        raise ValueError("Calibration record does not name the evaluated checkpoint")
    return {
        "path": str(calibration_path.resolve()),
        "sha256": expected_sha256,
        "split": calibration["split"],
        "clips_per_source": calibration["clips_per_source"],
        "num_clips": len(calibration["clip_ids"]),
        "seed": calibration["seed"],
        "target_mean_tokens": calibration["target_mean_tokens"],
        "selected_bias": calibration["selected_bias"],
        "selected_mean_tokens": calibration["selected_mean_tokens"],
    }


def validate_checkpoint(
    checkpoint: Path, expected_model_sha256: str, expected_config_sha256: str
) -> dict[str, Any]:
    model_path = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete AutoGaze checkpoint: {checkpoint}")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_model_sha256:
        raise ValueError("R2d checkpoint model SHA-256 mismatch")
    config_sha256 = sha256_file(config_path)
    if config_sha256 != expected_config_sha256:
        raise ValueError("R2d checkpoint config SHA-256 mismatch")
    config = json.loads(config_path.read_text())
    if int(config.get("num_vision_tokens_each_frame", -1)) != 265:
        raise ValueError("R2d checkpoint does not use the audited 265-token action space")
    return {
        "path": str(checkpoint.resolve()),
        "model_sha256": actual_sha256,
        "config_sha256": config_sha256,
        "num_vision_tokens_each_frame": 265,
    }


def main() -> None:
    wrapper_args, runner_argv = parse_wrapper_args()
    runner_dir = Path(wrapper_args.legacy_runner_dir).resolve()
    runner_path = runner_dir / "evaluate_hlvid_nvila.py"
    common_path = runner_dir / "hlvid_common.py"
    if not runner_path.is_file() or not common_path.is_file():
        raise FileNotFoundError(f"Missing established HLVid runner under {runner_dir}")
    if sha256_file(runner_path) != EXPECTED_RUNNER_SHA256 or sha256_file(common_path) != EXPECTED_COMMON_SHA256:
        raise ValueError("Established HLVid runner SHA-256 mismatch")
    sys.path.insert(0, str(runner_dir))
    import evaluate_hlvid_nvila as runner

    original_argv = sys.argv
    sys.argv = [str(runner_dir / "evaluate_hlvid_nvila.py"), *runner_argv]
    try:
        runner_args = runner.parse_args()
    finally:
        sys.argv = original_argv

    validate_established_protocol(runner_args, preflight=wrapper_args.preflight)
    checkpoint = Path(runner_args.autogaze_model_id)
    checkpoint_record = validate_checkpoint(
        checkpoint,
        wrapper_args.checkpoint_model_sha256,
        wrapper_args.checkpoint_config_sha256,
    )
    calibration_record = validate_calibration(
        wrapper_args.calibration_json,
        expected_sha256=wrapper_args.calibration_sha256,
        checkpoint=checkpoint,
        eos_logit_bias=wrapper_args.eos_logit_bias,
    )
    dataset_parquet = Path(runner_args.dataset_root) / "data" / "test-00000-of-00001.parquet"
    if not dataset_parquet.is_file():
        raise FileNotFoundError(f"Missing protected HLVid benchmark split: {dataset_parquet}")
    dataset_sha256 = sha256_file(dataset_parquet)
    if dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError("Protected HLVid benchmark split SHA-256 mismatch")

    real_auto_processor = runner.AutoProcessor
    installed_stats = []

    class R2DAutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            processor = real_auto_processor.from_pretrained(*args, **kwargs)
            stats = install_r2d_hlvid_forward(
                processor._autogaze_model,
                mode=wrapper_args.r2d_mode,
                eos_logit_bias=wrapper_args.eos_logit_bias if wrapper_args.r2d_mode == "variable" else None,
            )
            installed_stats.append(stats)
            return processor

    runner.AutoProcessor = R2DAutoProcessor
    runner.parse_args = lambda: runner_args
    runner.main()

    if len(installed_stats) != 1:
        raise RuntimeError(f"Expected one patched NVILA processor, found {len(installed_stats)}")
    summary_path = Path(runner_args.summary_output)
    summary = json.loads(summary_path.read_text())
    expected_examples = 1 if wrapper_args.preflight else ESTABLISHED_PROTOCOL["num_examples"]
    if int(summary.get("num_examples", -1)) != expected_examples:
        raise ValueError(f"HLVid output has {summary.get('num_examples')} examples; expected {expected_examples}")
    summary["r2d_hlvid_adapter"] = {
        "policy_label": wrapper_args.policy_label,
        "mode": wrapper_args.r2d_mode,
        "base_seed": wrapper_args.base_seed,
        "training_seed": wrapper_args.training_seed,
        "action_contract": {
            "fine_action_ids": [69, 264],
            "eos_action_id": 265,
            "variable_min_spatial_actions": 4,
            "variable_max_generation_tokens": 36,
            "forced_spatial_actions": 16,
            "calibrated_eos_logit_bias": wrapper_args.eos_logit_bias,
        },
        "checkpoint": checkpoint_record,
        "eos_calibration": calibration_record,
        "benchmark_protocol": {
            **ESTABLISHED_PROTOCOL,
            "dataset_root": str(Path(runner_args.dataset_root).resolve()),
            "dataset_parquet_sha256": dataset_sha256,
            "model_path": str(Path(runner_args.model_path).resolve()),
            "preflight": wrapper_args.preflight,
        },
        "allocation_statistics": installed_stats[0].as_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["r2d_hlvid_adapter"], indent=2))


if __name__ == "__main__":
    main()
