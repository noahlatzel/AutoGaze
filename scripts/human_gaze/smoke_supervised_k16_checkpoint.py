#!/usr/bin/env python3
"""Disposable actual-Trainer optimizer/save/reload smoke; never production data selection."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb

from autogaze.datasets.av_gaze_stavis import BalancedSourceSampler
from autogaze.datasets.collate import collate_fn
from autogaze.human_gaze.supervised_analysis import sha256_file
from autogaze.models.autogaze import AutoGazeImageProcessor
from autogaze.supervised_checkpoint import supervised_resume_contract, verify_supervised_completion
from scripts.human_gaze.preflight_supervised_k16 import (
    EXPECTED_CELL_MASS_SHA256, EXPECTED_MANIFEST_SHA256, EXPECTED_MODEL_SHA256,
    build_dataset, enforce_resources, git_output, gpu_snapshot, gradient_contract,
    load_stage_config, make_model, parameter_contract, tensor_sha256, utc_now,
    verify_exact_k, write_new_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Disposable smoke output must be a new directory")
    if git_output("rev-parse", "HEAD") != args.expected_commit or git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Smoke requires the exact clean versioned repair source")
    if gpu_snapshot()["compute_processes"]:
        raise RuntimeError("The VM GPU lane is occupied; refusing smoke")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_ceiling, rss_ceiling = 10 * 1024**3, 8 * 1024**3
    torch.cuda.set_per_process_memory_fraction(gpu_ceiling / torch.cuda.get_device_properties(device).total_memory, device)
    torch.cuda.reset_peak_memory_stats(device)
    manifest = Path("/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/d0_stavis_fold1_validated/clips.jsonl")
    mass_path = manifest.parent / "cell_mass.npy"
    pretrained = Path("/storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze")
    for path, expected in ((manifest, EXPECTED_MANIFEST_SHA256), (mass_path, EXPECTED_CELL_MASS_SHA256), (pretrained / "model.safetensors", EXPECTED_MODEL_SHA256)):
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen smoke input hash mismatch: {path}")
    cfg = load_stage_config(Path("/storage/user/zverev/datasets/av-gaze-stavis"), manifest, mass_path)
    processor_cfg = OmegaConf.to_container(cfg.model.preprocessing, resolve=True)
    processor_cfg["size"] = {"shortest_edge": 224}
    processor = AutoGazeImageProcessor(**processor_cfg)
    dataset = build_dataset(cfg, processor)
    sampler = BalancedSourceSampler(dataset, num_samples=1852, seed=int(cfg.trainer.seed), num_replicas=1, rank=0)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, drop_last=True, num_workers=0, collate_fn=collate_fn)
    model, loading = make_model(cfg, pretrained)
    for component in (model.gazing_model.vision_model, model.gazing_model.connector):
        for parameter in component.parameters():
            parameter.requires_grad = False
    parameters = parameter_contract(model)
    task = instantiate(cfg.task).to(device)
    algorithm = instantiate(cfg.algorithm)
    torch.distributed.init_process_group("nccl", init_method=f"file://{args.output_dir / 'ddp_init'}", rank=0, world_size=1)
    wandb.init(mode="disabled")
    receipt = {"schema_version": 1, "kind": "disposable_actual_trainer_checkpoint_smoke", "scientific_evaluation": False, "production_checkpoint": False, "source_commit": args.expected_commit, "started_utc": utc_now()}
    try:
        model = model.to(device)
        ddp = DDP(model, find_unused_parameters=False, device_ids=[0], output_device=0)
        optimizer = torch.optim.Adam((p for p in ddp.parameters() if p.requires_grad), lr=float(cfg.trainer.lr))
        (args.output_dir / "checkpoint").mkdir()
        trainer = instantiate(cfg.trainer, gaze_model=ddp, task=task, algorithm=algorithm,
                              train_loader=loader, val_loader=loader, optimizer=optimizer,
                              optimizer_name=cfg.trainer.optimizer, save_dir=str(args.output_dir / "checkpoint"),
                              grad_acc_steps=4, gaze_processor=processor, gaze_weights=None)
        # The production config is retained in serialized state. Only this
        # disposable runtime is stopped after one update and skips validation.
        trainer.max_train_steps = 1
        trainer.validate_at_start = False
        trainer.skip_final_validation = True
        trainer.val_nsteps = trainer.save_nsteps = 1000000
        trainer.trainval()
        memory = enforce_resources(device, gpu_ceiling, rss_ceiling)
        contract = supervised_resume_contract(trainer)
        completion = verify_supervised_completion(trainer.save_dir, expected_contract=contract)
        saved = torch.load(Path(trainer.save_dir) / "checkpoint_latest_train.pt", map_location="cpu", weights_only=True)
        if (saved["train_step"], saved["epoch"], saved["iteration"]) != (1, 0, 4) or saved["config"]["optimizer"] != "adam":
            raise ValueError("Actual Trainer saved an incorrect update/cursor/optimizer identity")
        model_before = {name: tensor_sha256(p) for name, p in model.named_parameters()}
        adam_before = copy.deepcopy(optimizer.state_dict())
        scheduler_before = copy.deepcopy(trainer.scheduler.state_dict())
        sampler.set_epoch(0)
        indices = list(iter(sampler))[:5]
        next_inputs = collate_fn([dataset[indices[4]]])
        expected_teacher = algorithm.preprocess_inputs(dict(next_inputs))["gt_gazing_info"]["gazing_pos"].clone()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.01)
        trainer.load_checkpoint(resume=True)
        if {name: tensor_sha256(p) for name, p in model.named_parameters()} != model_before:
            raise ValueError("Reload did not restore exact model parameters")
        if trainer.scheduler.state_dict() != scheduler_before:
            raise ValueError("Reload did not restore scheduler")
        for key, state in adam_before["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0)
        resumed_inputs = algorithm.preprocess_inputs(dict(next_inputs))
        if not torch.equal(resumed_inputs["gt_gazing_info"]["gazing_pos"], expected_teacher):
            raise ValueError("Reload did not restore the next teacher sample")
        resumed_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in resumed_inputs.items()}
        # gt_gazing_info is nested; use the ordinary production mover.
        from autogaze.utils import move_inputs_to_cuda
        resumed_inputs = move_inputs_to_cuda(resumed_inputs)
        optimizer.zero_grad()
        _, _, outputs = trainer._one_step_ntp(resumed_inputs)
        outputs["loss"].mean().backward()
        gradients = gradient_contract(model)
        optimizer.zero_grad()
        with torch.no_grad():
            model.eval()
            info = model({"video": resumed_inputs["video"]}, temperature=0.0, **task.gaze_model_kwargs)
        cells = verify_exact_k(info)
        memory = enforce_resources(device, gpu_ceiling, rss_ceiling)
        if time.monotonic() - started >= 1800:
            raise TimeoutError("Disposable smoke exceeded 30 minutes")
        receipt.update({"status": "pass", "finished_utc": utc_now(), "elapsed_seconds": time.monotonic() - started,
                        "hardware": gpu_snapshot(), "resources": memory, "parameters": parameters,
                        "loading": loading, "completion": completion, "resume_contract": contract,
                        "checkpoint_marker_sha256": sha256_file(Path(trainer.save_dir) / "checkpoint_latest_complete.json"),
                        "optimizer_updates": 1, "training_microbatches": 4, "resumed_backward_microbatches": 1,
                        "greedy_exact_k": True, "greedy_cells_sha256": tensor_sha256(cells), "resumed_gradients": gradients,
                        "train_clip_ids": [dataset.records[index]["clip_id"] for index in indices],
                        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda},
                        "input_sha256": {"manifest": EXPECTED_MANIFEST_SHA256, "cell_mass": EXPECTED_CELL_MASS_SHA256, "pretrained": EXPECTED_MODEL_SHA256}})
    except Exception as error:
        receipt.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        write_new_json(args.output_dir / "receipt.json", receipt)
        torch.distributed.destroy_process_group()
        wandb.finish()
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output_dir / "receipt.json"), "resources": receipt["resources"]}))


if __name__ == "__main__":
    main()
