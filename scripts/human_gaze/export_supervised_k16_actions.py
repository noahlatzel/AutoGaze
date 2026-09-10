#!/usr/bin/env python3
"""Export one checkpoint's full-validation greedy exact-K16 actions once."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset
from autogaze.human_gaze.coverage import global_positions_to_fine_cells
from autogaze.human_gaze.supervised_analysis import (
    sha256_file,
    sha256_tree,
)
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-mass", type=Path, required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=Path)
    checkpoint.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--phase-train-step",
        type=int,
        help="Required with --run-dir; resolves an exact saved phase-local train step.",
    )
    parser.add_argument("--method", choices=("supervised", "rl"), required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--cumulative-update", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def source_identity() -> dict:
    dirty = bool(git_output("status", "--porcelain=v1", "--untracked-files=all"))
    if dirty:
        raise ValueError("Action export requires a clean source checkout")
    return {
        "repository": git_output("remote", "get-url", "origin"),
        "commit": git_output("rev-parse", "HEAD"),
        "dirty": False,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_checkpoint(
    checkpoint: Path | None,
    run_dir: Path | None,
    phase_train_step: int | None,
) -> tuple[Path, Path | None]:
    """Resolve a direct model or one unambiguous saved train step."""
    if checkpoint is not None:
        if phase_train_step is not None:
            raise ValueError("--phase-train-step is only valid with --run-dir")
        return checkpoint.resolve(strict=True), None
    if run_dir is None or phase_train_step is None or phase_train_step < 0:
        raise ValueError("--run-dir requires a nonnegative --phase-train-step")
    root = run_dir.resolve(strict=True)
    latest_train = root / "checkpoint_latest_train.pt"
    if latest_train.is_file():
        latest = torch.load(latest_train, map_location="cpu", weights_only=False)
        if int(latest.get("train_step", -1)) == phase_train_step:
            return (root / "checkpoint_latest_gaze").resolve(strict=True), latest_train
    matches = []
    for state_path in root.glob("checkpoint_ep*/checkpoint_train.pt"):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(state.get("train_step", -1)) == phase_train_step:
            matches.append((state_path.parent / "checkpoint_gaze", state_path))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one saved checkpoint at phase train step {phase_train_step}, "
            f"found {len(matches)} in {root}"
        )
    return matches[0][0].resolve(strict=True), matches[0][1].resolve(strict=True)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace action export: {args.output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative")
    expected_training_seed = args.base_seed + 100000
    if args.base_seed not in range(440826, 440832):
        raise ValueError("base-seed is outside the frozen six-seed matrix")
    if args.training_seed != expected_training_seed:
        raise ValueError("training-seed does not match the frozen paired seed")
    if args.method == "supervised" and args.cumulative_update not in (
        2315,
        5000,
        10000,
        15000,
        20000,
    ):
        raise ValueError("Supervised export is not a fixed convergence checkpoint")
    if args.method == "rl" and args.cumulative_update != 20000:
        raise ValueError("Only the frozen RL K16 endpoint requires a new action export")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source = source_identity()
    manifest_path = args.manifest.resolve(strict=True)
    cell_mass_path = args.cell_mass.resolve(strict=True)
    checkpoint, checkpoint_train_state = resolve_checkpoint(
        args.checkpoint, args.run_dir, args.phase_train_step
    )
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError("Checkpoint lacks model.safetensors")
    dataset_root = args.dataset_root.resolve(strict=True)

    processor = AutoGazeImageProcessor.from_pretrained(
        checkpoint, local_files_only=True
    )
    model = AutoGaze.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    dataset = AVGazeStavisDataset(
        root=str(dataset_root),
        manifest_path=str(manifest_path),
        split="val",
        load_rgb=True,
        load_heatmap=False,
        cell_mass_path=str(cell_mass_path),
        image_processor=processor,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True)
    actions_path = args.output_dir / "actions.jsonl"
    started_utc = utc_now()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    written = 0
    allowed = list(range(69, 265))
    with actions_path.open("x", encoding="utf-8") as handle:
        with torch.inference_mode():
            for batch in loader:
                video = batch["video"].to(device, non_blocking=True)
                gaze = model(
                    {"video": video},
                    max_gaze_tokens_each_frame=16,
                    allowed_token_ids=allowed,
                    generate_only=True,
                )
                if gaze["if_padded_gazing"].any():
                    raise ValueError("Exact-K16 export unexpectedly produced padding/EOS")
                expected_counts = torch.full_like(gaze["num_gazing_each_frame"], 16)
                if not torch.equal(gaze["num_gazing_each_frame"], expected_counts):
                    raise ValueError("Checkpoint did not generate exactly 16 actions per frame")
                cells = global_positions_to_fine_cells(
                    gaze["gazing_pos"],
                    num_frames=16,
                    exact_budget=16,
                    actions_per_frame=265,
                    fine_action_offset=69,
                ).cpu()
                for index in range(cells.shape[0]):
                    row = {
                        "clip_id": batch["clip_id"][index],
                        "source": batch["source"][index],
                        "video_id": batch["video_id"][index],
                        "frame_numbers": [int(value) for value in batch["frame_numbers"][index]],
                        "fine_cells": cells[index].tolist(),
                    }
                    handle.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
                    written += 1
    if written != len(dataset):
        raise AssertionError("Action export did not write every validation clip")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    max_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": "supervised_k16_comparison",
        "kind": "full_validation_greedy_exact_action_export",
        "method": args.method,
        "base_seed": args.base_seed,
        "training_seed": args.training_seed,
        "cumulative_update": args.cumulative_update,
        "split": "val",
        "clip_len": 16,
        "exact_k": 16,
        "num_fine_cells": 196,
        "fine_action_offset": 69,
        "allowed_local_action_ids_inclusive": [69, 264],
        "greedy": True,
        "selection_without_replacement": True,
        "num_clips": written,
        "num_frames": written * 16,
        "num_action_rows": written * 16 * 16,
        "source": source,
        "inputs": {
            "dataset_root": str(dataset_root),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "cell_mass": str(cell_mass_path),
            "cell_mass_sha256": sha256_file(cell_mass_path),
            "checkpoint": str(checkpoint),
            "checkpoint_tree_sha256": sha256_tree(checkpoint),
            "model_safetensors_sha256": sha256_file(checkpoint / "model.safetensors"),
            "checkpoint_train_state": (
                str(checkpoint_train_state) if checkpoint_train_state is not None else None
            ),
            "checkpoint_train_state_sha256": (
                sha256_file(checkpoint_train_state)
                if checkpoint_train_state is not None
                else None
            ),
        },
        "actions_sha256": sha256_file(actions_path),
        "runtime": {
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            ),
            "process_maxrss_raw": int(max_rss_raw),
            "process_maxrss_platform_unit": "KiB_on_Linux_bytes_on_macOS",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NODELIST",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    manifest_path_out = args.output_dir / "manifest.json"
    with manifest_path_out.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output_dir),
                "actions_sha256": manifest["actions_sha256"],
                "num_clips": written,
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
