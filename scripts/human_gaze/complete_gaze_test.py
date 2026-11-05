#!/usr/bin/env python3
"""Frozen K16 test endpoint inference and paired CPU curation; never training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset, read_manifest
from autogaze.human_gaze.checkpoint_provenance import load_checkpoint_inventory, verify_rl_checkpoint
from autogaze.human_gaze.coverage import center_order, global_positions_to_fine_cells
from autogaze.human_gaze.supervised_analysis import (
    aggregate_frame_values, cell_mass_for_records, load_action_export,
    selected_mass, seed_summary, sha256_file, sha256_tree, source_prior_cells,
    source_stratified_video_bootstrap_delta,
)
from scripts.human_gaze.export_supervised_k16_actions import source_identity


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def inputs(config):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if cfg["experiment_id"] != "gaze_completion_20261105" or cfg["training_authorized"] or cfg["hlvid_authorized"]:
        raise ValueError("Unexpected inference-only contract")
    for name in ("manifest", "cell_mass"):
        if sha256_file(Path(cfg["data"][name])) != cfg["data"][name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    all_records = read_manifest(cfg["data"]["manifest"])
    for split, count in cfg["data"]["expected_clips"].items():
        if sum(r["split"] == split for r in all_records) != count:
            raise ValueError(f"Unexpected {split} population")
    split_videos = {split: {(r["source"], r["video_id"]) for r in all_records if r["split"] == split}
                    for split in ("train", "val", "test")}
    if any(split_videos[a] & split_videos[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Train/validation/test video overlap")
    inventory = load_checkpoint_inventory(ROOT / cfg["checkpoint_inventory"],
        expected_sha256=cfg["checkpoint_inventory_sha256"], repository_root=ROOT)
    return cfg, all_records, inventory


def checkpoint_for(model, cfg, inventory):
    if model not in cfg["test"]["models"]:
        raise ValueError("Model not in frozen matrix")
    if model.startswith("rl"):
        return verify_rl_checkpoint(inventory, base_seed=int(model[2:]), cumulative_update=20000)
    checkpoint = Path(cfg["pretrained"]["path"])
    hashes = {name: sha256_file(checkpoint / name) for name in
              ("config.json", "model.safetensors", "preprocessor_config.json")}
    if hashes["model.safetensors"] != cfg["pretrained"]["model_safetensors_sha256"]:
        raise ValueError("Pretrained model changed")
    return checkpoint, {"authority": "frozen_pretrained_initialization", "checkpoint": {
        "path": str(checkpoint), "files_sha256": hashes, "tree_sha256": sha256_tree(checkpoint)}}


def export(args, cfg, records, inventory):
    from torch.utils.data import DataLoader
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

    output = args.output
    if output.exists():
        raise FileExistsError(output)
    source = source_identity()
    checkpoint, provenance = checkpoint_for(args.model, cfg, inventory)
    processor = AutoGazeImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoGaze.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    dataset = AVGazeStavisDataset(root=cfg["data"]["root"], manifest_path=cfg["data"]["manifest"],
        split="test", load_rgb=True, load_heatmap=False, cell_mass_path=cfg["data"]["cell_mass"], image_processor=processor)
    loader = DataLoader(dataset, batch_size=cfg["resources"]["batch_size"],
        num_workers=cfg["resources"]["num_workers"], shuffle=False, pin_memory=True)
    output.mkdir(parents=True)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    written = 0
    with (output / "actions.jsonl").open("x") as handle, torch.inference_mode():
        for batch in loader:
            gaze = model({"video": batch["video"].cuda(non_blocking=True)},
                max_gaze_tokens_each_frame=16, allowed_token_ids=list(range(69, 265)), generate_only=True)
            if gaze["if_padded_gazing"].any() or not torch.all(gaze["num_gazing_each_frame"] == 16):
                raise ValueError("Exact fine K16 contract violated")
            cells = global_positions_to_fine_cells(gaze["gazing_pos"], num_frames=16,
                exact_budget=16, actions_per_frame=265, fine_action_offset=69).cpu().numpy()
            for i in range(len(cells)):
                row = {key: batch[key][i] for key in ("clip_id", "source", "video_id")}
                row.update(frame_numbers=[int(v) for v in batch["frame_numbers"][i]], fine_cells=cells[i].tolist())
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
    if written != 663:
        raise ValueError("Incomplete test inference")
    torch.cuda.synchronize()
    manifest = dict(schema_version=1, status="complete", experiment_id=cfg["experiment_id"],
        run_id=cfg["run_id"],
        method=args.model, base_seed=int(args.model[2:]) if args.model.startswith("rl") else None,
        cumulative_update=20000 if args.model.startswith("rl") else 0,
        split="test", clip_len=16, exact_k=16, num_fine_cells=196, fine_action_offset=69,
        allowed_local_action_ids_inclusive=[69, 264], greedy=True, selection_without_replacement=True,
        num_clips=written, source=source, checkpoint_provenance=provenance,
        actions_sha256=sha256_file(output / "actions.jsonl"),
        inputs=dict(manifest_sha256=cfg["data"]["manifest_sha256"], cell_mass_sha256=cfg["data"]["cell_mass_sha256"],
            checkpoint=str(checkpoint), checkpoint_tree_sha256=sha256_tree(checkpoint),
            model_safetensors_sha256=sha256_file(checkpoint / "model.safetensors"), config_sha256=sha256_file(args.config)),
        runtime=dict(started_unix=started, elapsed_seconds=time.time()-started, gpu=torch.cuda.get_device_name(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), python=platform.python_version(), torch=torch.__version__),
        slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        prior_exposure=cfg["test"]["prior_exposure"], logical_nvila_rollouts=0)
    OmegaConf.save(OmegaConf.create(cfg), output / "config.yaml")
    manifest["inputs"]["resolved_config_sha256"] = sha256_file(output / "config.yaml")
    write(output / "manifest.json", manifest)
    print(json.dumps({"model": args.model, "clips": written, "elapsed_seconds": manifest["runtime"]["elapsed_seconds"]}))


def curate(args, cfg, all_records, inventory):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = source_identity()
    cache = np.load(cfg["data"]["cell_mass"], mmap_mode="r")
    records = [r for r in all_records if r["split"] == "test"]
    mass = cell_mass_for_records(records, cache)
    prior = source_prior_cells(all_records, cell_mass_for_records(all_records, cache))
    controls = {
        "train_source_prior": np.asarray([[prior[r["source"]]] * 16 for r in records]),
        "center16": np.broadcast_to(center_order(14)[:16].numpy(), (len(records), 16, 16)),
    }
    reports = {name: aggregate_frame_values(records, selected_mass(mass, actions)) for name, actions in controls.items()}
    exports = []
    for name in cfg["test"]["models"]:
        _, provenance = checkpoint_for(name, cfg, inventory)
        manifest, actions = load_action_export(args.actions / name, records,
            expected_manifest_sha256=cfg["data"]["manifest_sha256"], expected_cell_mass_sha256=cfg["data"]["cell_mass_sha256"],
            expected_method=name, expected_split="test", expected_checkpoint_provenance=provenance)
        reports[name] = aggregate_frame_values(records, selected_mass(mass, actions))
        exports.append({"path": str(args.actions / name / "manifest.json"),
                        "sha256": sha256_file(args.actions / name / "manifest.json"), "runtime": manifest["runtime"]})
    baseline = json.loads((ROOT / "experiments/human_gaze/results/b0_coverage_baselines_fold1/metrics.json").read_text())
    # Confirm the source-trained spatial prior has not changed across studies.
    if any(prior[s] != baseline["prior_cell_order"][s][:16] for s in prior):
        raise ValueError("Training-source prior differs from original baseline")
    rl = [reports[f"rl{s}"] for s in cfg["test"]["seeds"]]
    contrasts = {name: source_stratified_video_bootstrap_delta(rl, [reports[name]] * 6,
        iterations=cfg["test"]["bootstrap_iterations"], seed=cfg["test"]["bootstrap_seed"], expected_seed_count=6)
        for name in ("train_source_prior", "center16", "pretrained")}
    primary = contrasts["train_source_prior"]
    decision = "supports_advantage" if primary["ci90_low"] > 0 else "negative" if primary["ci90_high"] < 0 else "inconclusive"
    metrics = dict(schema_version=1, status="complete", run_id=cfg["run_id"], primary=primary, decision=decision,
        contrasts=contrasts, per_model=reports,
        rl_seed_summary=seed_summary([r["macro_source_mean"] for r in rl], expected_count=6),
        prior_exposure=cfg["test"]["prior_exposure"])
    args.output.mkdir(parents=True)
    write(args.output / "metrics.json", metrics)
    OmegaConf.save(OmegaConf.create(cfg), args.output / "config.yaml")
    (args.output / "README.md").write_text("# Fixed K16 endpoint confirmation\n\n" +
        f"Decision: {decision}. RL minus training-source prior: {primary['difference']:+.6f}, " +
        f"paired video 90% CI [{primary['ci90_low']:+.6f}, {primary['ci90_high']:+.6f}].\n\n" +
        cfg["test"]["prior_exposure"] + "\n\nThe interval averages the six fixed seeds; seed variation is reported separately. No QA inference or training.\n")
    write(args.output / "manifest.json", dict(schema_version=1, status="complete", source=source,
        run_id=cfg["run_id"],
        data=cfg["data"], config_sha256=sha256_file(args.config), action_exports=exports,
        resolved_config_sha256=sha256_file(args.output / "config.yaml"),
        logical_nvila_rollouts=0, heavy_artifacts=str(args.actions)))
    print(json.dumps({"status": "complete", "decision": decision, "output": str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "export", "curate"])
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/human_gaze/configs/gaze_completion_20261105.yaml")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions", type=Path)
    args = parser.parse_args()
    cfg, records, inventory = inputs(args.config)
    artifact_root = args.output.parent.parent if args.command == "export" else args.output.parent
    cfg["run_id"] = os.environ.get("GAZE_COMPLETION_RUN_ID", artifact_root.name)
    if args.command == "preflight":
        models = {name: checkpoint_for(name, cfg, inventory)[1] for name in cfg["test"]["models"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write(args.output, dict(status="pass", models=models, counts=cfg["data"]["expected_clips"],
            config_sha256=sha256_file(args.config), prior_exposure=cfg["test"]["prior_exposure"]))
    elif args.command == "export":
        export(args, cfg, records, inventory)
    else:
        if args.actions is None:
            parser.error("curate requires --actions")
        curate(args, cfg, records, inventory)


if __name__ == "__main__":
    main()
