#!/usr/bin/env python3
"""Bind the preserved processor smoke and live canonical artifacts before HLVid QA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.human_gaze.evaluate_hlvid_nvila_fixed_budget import (
    build_identity,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def build_pretrained_identity(config: dict) -> dict:
    protocol = config["protocol"]
    models = config["models"]
    args = SimpleNamespace(
        policy_kind="pretrained",
        policy_label="pretrained_exact_k16",
        base_seed=None,
        continuation_seed=None,
        exact_budget=16,
        fine_action_offset=69,
        actions_per_frame=265,
        autogaze_model_id=Path(models["pretrained_autogaze"]["path"]),
        model_path=Path(models["nvila"]["path"]),
        num_video_frames=protocol["num_video_frames"],
        num_video_frames_thumbnail=protocol["num_video_frames_thumbnail"],
        max_tiles_video=protocol["max_tiles_video"],
        tile_len=protocol["tile_len"],
        max_new_tokens=protocol["max_new_tokens"],
        torch_dtype=protocol["torch_dtype"],
        max_batch_size_autogaze=16,
        max_batch_size_siglip=32,
    )
    return build_identity(
        args,
        dataset_sha256=config["dataset"]["parquet_sha256"],
        decode_audit=None,
    )


def validate(config: dict, smoke: dict, smoke_path: Path, expected_commit: str) -> dict:
    observed_commit = git_value("rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise ValueError(f"Execution commit mismatch: {observed_commit} != {expected_commit}")
    dirty = bool(git_value("status", "--porcelain", "--untracked-files=no"))
    if dirty:
        raise ValueError("Tracked execution checkout is dirty")

    dataset_path = Path(config["dataset"]["root"]) / config["dataset"]["parquet"]
    if sha256(dataset_path) != config["dataset"]["parquet_sha256"]:
        raise ValueError("Live HLVid parquet differs from the frozen YAML")
    identity = build_pretrained_identity(config)
    models = config["models"]
    expected_nvila = models["nvila"]
    expected_autogaze = models["pretrained_autogaze"]
    if identity["nvila_snapshot_revision"] != expected_nvila["huggingface_revision"]:
        raise ValueError("NVILA snapshot revision differs from the frozen YAML")
    if identity["nvila_artifact_hashes"] != expected_nvila["artifact_hashes"]:
        raise ValueError("Live NVILA model/config/processor hashes differ from the frozen YAML")
    if identity["autogaze_checkpoint_sha256"] != expected_autogaze["model_safetensors_sha256"]:
        raise ValueError("Live pretrained AutoGaze checkpoint differs from the frozen YAML")
    if identity["autogaze_artifact_hashes"] != expected_autogaze["artifact_hashes"]:
        raise ValueError("Live pretrained AutoGaze config/processor hashes differ from the frozen YAML")

    smoke_contract = config["recording"]["processor_smoke"]
    if sha256(smoke_path) != smoke_contract["report_sha256"]:
        raise ValueError("Preserved processor smoke report hash mismatch")
    if smoke.get("status") != "pass":
        raise ValueError("Preserved processor smoke did not pass")
    if smoke.get("model_path") != expected_nvila["path"]:
        raise ValueError("Processor smoke used a different NVILA path")
    if smoke.get("autogaze_model_id") != expected_autogaze["path"]:
        raise ValueError("Processor smoke used a different AutoGaze checkpoint path")
    if (
        smoke.get("reference_processor_output_sha256")
        != smoke.get("instrumented_processor_output_sha256")
        or smoke.get("tensor_shapes_equal") is not True
        or smoke.get("reference_total_context_tokens")
        != smoke.get("instrumented_total_context_tokens")
    ):
        raise ValueError("Processor smoke does not prove actual tensor compatibility")
    for relative, expected_hash in smoke_contract["code_sha256"].items():
        if sha256(Path(relative)) != expected_hash:
            raise ValueError(f"Processor smoke code identity mismatch: {relative}")

    return {
        "schema_version": 1,
        "status": "pass",
        "code_commit": observed_commit,
        "code_dirty": False,
        "dataset_parquet_sha256": identity["dataset_parquet_sha256"],
        "nvila_snapshot_revision": identity["nvila_snapshot_revision"],
        "nvila_artifact_hashes": identity["nvila_artifact_hashes"],
        "autogaze_checkpoint_sha256": identity["autogaze_checkpoint_sha256"],
        "autogaze_artifact_hashes": identity["autogaze_artifact_hashes"],
        "processor_smoke": {
            "path": str(smoke_path.resolve()),
            "sha256": sha256(smoke_path),
            "fixture": smoke_contract["fixture"],
            "tensor_sha256": smoke["instrumented_processor_output_sha256"],
            "context_tokens": smoke["instrumented_total_context_tokens"],
            "code_sha256": smoke_contract["code_sha256"],
            "scope": smoke["compatibility_scope"],
        },
        "full_protocol_gate": "question 0 under 128-frame/64-thumbnail/max_tiles_video=48 allocation",
    }


def write_or_validate(path: Path, report: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != report:
            raise ValueError("Existing admission record differs from the live verified identity")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-admission-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
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


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    smoke = json.loads(args.smoke_report.read_text())
    report = validate(config, smoke, args.smoke_report, args.expected_code_commit)
    write_or_validate(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
