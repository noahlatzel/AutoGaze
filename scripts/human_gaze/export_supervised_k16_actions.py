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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset
from autogaze.human_gaze.coverage import global_positions_to_fine_cells
from autogaze.human_gaze.checkpoint_provenance import (
    load_checkpoint_inventory,
    verify_checkpoint_for_method,
)
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
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("experiments/human_gaze/configs/supervised_k16_analysis.yaml"),
    )
    parser.add_argument("--training-root", type=Path)
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
    analysis_config = args.analysis_config.resolve(strict=True)
    cfg = OmegaConf.to_container(OmegaConf.load(analysis_config), resolve=True)
    if not isinstance(cfg, dict) or cfg.get("experiment_id") != "supervised_k16_comparison":
        raise ValueError("Unexpected supervised K16 analysis config")
    provenance_cfg = cfg.get("checkpoint_provenance", {})
    inventory_path = Path(provenance_cfg.get("inventory", ""))
    if not inventory_path.is_absolute():
        inventory_path = REPOSITORY_ROOT / inventory_path
    inventory = load_checkpoint_inventory(
        inventory_path,
        expected_sha256=provenance_cfg.get("inventory_sha256", ""),
        repository_root=REPOSITORY_ROOT,
    )
    manifest_path = args.manifest.resolve(strict=True)
    cell_mass_path = args.cell_mass.resolve(strict=True)
    if sha256_file(manifest_path) != inventory["data_sha256"]["manifest"]:
        raise ValueError("STAViS manifest differs from the authoritative execution input")
    if sha256_file(cell_mass_path) != inventory["data_sha256"]["cell_mass"]:
        raise ValueError("Human cell-mass cache differs from the authoritative execution input")
    if args.method == "supervised":
        expected_phase_step = {
            2315: 2315,
            5000: 2685,
            10000: 7685,
            15000: 12685,
            20000: 17685,
        }[args.cumulative_update]
        if args.phase_train_step != expected_phase_step:
            raise ValueError("CLI phase step does not match the frozen cumulative checkpoint")
        canonical_training_root = Path(inventory["supervised_training_root"])
        if args.training_root is not None and args.training_root.resolve(strict=True) != canonical_training_root.resolve(strict=True):
            raise ValueError("Supervised training root differs from the canonical admitted execution")
        checkpoint, checkpoint_provenance = verify_checkpoint_for_method(
            "supervised",
            inventory,
            training_root=canonical_training_root,
            base_seed=args.base_seed,
            cumulative_update=args.cumulative_update,
            supplied_checkpoint=args.checkpoint,
            supplied_run_dir=args.run_dir,
        )
        checkpoint_train_state = Path(checkpoint_provenance["checkpoint_train_state"])
    else:
        if args.phase_train_step is not None:
            raise ValueError("RL export requires its frozen direct checkpoint, not a run directory")
        checkpoint, checkpoint_provenance = verify_checkpoint_for_method(
            "rl",
            inventory,
            base_seed=args.base_seed,
            cumulative_update=args.cumulative_update,
            supplied_checkpoint=args.checkpoint,
            supplied_run_dir=args.run_dir,
        )
        checkpoint_train_state = None
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
        "checkpoint_provenance": checkpoint_provenance,
        "inputs": {
            "analysis_config": str(analysis_config),
            "analysis_config_sha256": sha256_file(analysis_config),
            "checkpoint_provenance_inventory": str(inventory_path.resolve(strict=True)),
            "checkpoint_provenance_inventory_sha256": sha256_file(inventory_path),
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
