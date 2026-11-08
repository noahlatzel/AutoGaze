#!/usr/bin/env python3
"""Gate SB1: TASED-Net (DHF1K) saliency export (GPU) and K16 coverage curation (CPU).

Inference only; never training, NVILA or HLVid.  Commands:

* ``export``  — TASED-Net maps for the 663 test clips x 16 frames; saves the
  predicted fine-cell mass, the top-16 cells per frame and the uint8 maps.
* ``check``   — CPU implementation check (GT1 trained mean, Center16, source prior).
* ``curate``  — frozen SB1 analysis and result bundle.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES, read_manifest
from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.saliency_baseline import (
    INPUT_HW, LEN_TEMPORAL, best_constant_cells, boundary_tie_count, load_native_frame,
    min_video_frames, native_window, png_to_cell_mass, postprocess_prediction, to_model_input,
    top_k_cells,
)
from autogaze.human_gaze.supervised_analysis import (
    action_intersection, aggregate_frame_values, cell_mass_for_records, seed_summary, selected_mass,
    sha256_file, source_prior_cells, source_stratified_video_bootstrap_delta,
)

DEFAULT_CONFIG = ROOT / "experiments/human_gaze/configs/sb1_saliency_baseline_v1.yaml"
K = 16


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def source_identity(allow_dirty=False):
    dirty = bool(git("status", "--porcelain=v1", "--untracked-files=no"))
    if dirty and not allow_dirty:
        raise ValueError("Evidential SB1 runs require a clean tracked checkout")
    return {"repository": git("remote", "get-url", "origin"), "commit": git("rev-parse", "HEAD"), "dirty": dirty}


def load_inputs(config):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if cfg["experiment_id"] != "sb1_saliency_baseline" or cfg["authorization"]["training_authorized"]:
        raise ValueError("Unexpected inference-only contract")
    for name in ("manifest", "cell_mass"):
        if sha256_file(Path(cfg["data"][name])) != cfg["data"][name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    all_records = read_manifest(cfg["data"]["manifest"])
    records = [r for r in all_records if r["split"] == "test"]
    if len(records) != cfg["data"]["expected_clips"]:
        raise ValueError("Unexpected test population")
    if len({(r["source"], r["video_id"]) for r in records}) != cfg["data"]["expected_videos"]:
        raise ValueError("Unexpected test video count")
    if sum(r["split"] == "train" for r in all_records) != cfg["data"]["expected_train_clips"]:
        raise ValueError("Unexpected train population")
    split_videos = {s: {(r["source"], r["video_id"]) for r in all_records if r["split"] == s} for s in ("train", "test")}
    if split_videos["train"] & split_videos["test"]:
        raise ValueError("Train/test video overlap")
    vendored = ROOT / "autogaze/human_gaze/third_party/tased_net/model.py"
    if sha256_file(vendored) != cfg["tased_net"]["model_py_sha256"]:
        raise ValueError("Vendored TASED-Net model.py differs from upstream")
    return cfg, all_records, records


class ClipWindows:
    """Torch dataset: unique native frames of one clip (uint8, 224x384) and the 16 window index rows."""

    def __init__(self, root, records):
        self.root, self.records = Path(root), records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        record = self.records[index]
        windows = [native_window(f) for f in record["frame_numbers"]]
        unique = sorted({f for w in windows for f in w})
        if unique[-1] > record["native_frame_count"] or record["native_frame_count"] < min_video_frames():
            raise ValueError(f"Window outside the annotated frame range for {record['clip_id']}")
        position = {f: i for i, f in enumerate(unique)}
        frame_dir = self.root / "video_frames" / record["source"] / record["video_id"]
        frames = np.stack([load_native_frame(frame_dir / f"img_{f:05d}.jpg") for f in unique])
        rows = np.asarray([[position[f] for f in w] for w in windows], dtype=np.int64)
        return {"frames": torch.from_numpy(frames), "rows": torch.from_numpy(rows), "index": index}


def export(args, cfg, records):
    import torch
    from torch.utils.data import DataLoader, Subset
    from autogaze.human_gaze.saliency_baseline import build_tased

    if args.output.exists():
        raise FileExistsError(args.output)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke = args.limit is not None
    source = source_identity(allow_dirty=smoke)
    tcfg = cfg["tased_net"]
    weights = Path(tcfg["weights_path"])
    if sha256_file(weights) != tcfg["weights_sha256"]:
        raise ValueError("TASED-Net weights changed")
    model = build_tased(weights).cuda().eval()
    subset = records[:args.limit] if smoke else records
    batch_size = args.batch_size or cfg["resources"]["batch_size"]
    if not smoke and batch_size != cfg["resources"]["batch_size"]:
        raise ValueError("Evidential exports must use the frozen batch size")
    loader = DataLoader(ClipWindows(cfg["data"]["root"], subset), batch_size=None, shuffle=False,
                        num_workers=cfg["resources"]["num_workers"], pin_memory=False)
    args.output.mkdir(parents=True)
    n = len(subset)
    maps = np.lib.format.open_memmap(args.output / "maps_uint8.npy", mode="w+", dtype=np.uint8,
                                     shape=(n, K, *INPUT_HW))
    predicted = np.zeros((n, K, 196), dtype=np.float64)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    reversed_frames = 0
    with (args.output / "actions.jsonl").open("x") as handle, torch.inference_mode():
        for item in loader:
            i = int(item["index"])
            record = subset[i]
            frames, rows = item["frames"], item["rows"]
            raw = []
            for start in range(0, K, batch_size):
                clip = to_model_input(frames[rows[start:start + batch_size]])  # [b, 3, 32, 224, 384]
                out = model(clip.cuda(non_blocking=True))
                if tuple(out.shape[1:]) != INPUT_HW:
                    raise ValueError(f"Unexpected TASED-Net output shape {tuple(out.shape)}")
                raw.append(out.float().cpu().numpy())
            raw = np.concatenate(raw)
            for t in range(K):
                png = postprocess_prediction(raw[t])
                maps[i, t] = png
                predicted[i, t] = png_to_cell_mass(png)
            reversed_frames += sum(f - 1 < LEN_TEMPORAL - 1 for f in record["frame_numbers"])
            cells = top_k_cells(predicted[i], K)
            row = {k: record[k] for k in ("clip_id", "source", "video_id", "frame_numbers")}
            row["fine_cells"] = cells.tolist()
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    maps.flush()
    del maps
    torch.cuda.synchronize()
    np.save(args.output / "predicted_cell_mass.npy", predicted)
    manifest = dict(
        schema_version=1, status="smoke" if smoke else "complete", experiment_id=cfg["experiment_id"],
        run_id=args.run_id, model="tased_net_dhf1k", split="test", clip_len=K, k=K, num_clips=n,
        batch_size=batch_size, len_temporal=LEN_TEMPORAL, frames_with_reversed_window=int(reversed_frames),
        boundary_ties_at_k=boundary_tie_count(predicted, K),
        numerics=dict(dtype="float32", cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                      matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_deterministic=torch.backends.cudnn.deterministic),
        source=source,
        weights=dict(path=str(weights), sha256=tcfg["weights_sha256"], url=tcfg["weights_url"],
                     loading="torch.load(weights_only=True), strict"),
        files_sha256={name: sha256_file(args.output / name)
                      for name in ("actions.jsonl", "predicted_cell_mass.npy", "maps_uint8.npy")},
        inputs=dict(manifest_sha256=cfg["data"]["manifest_sha256"], cell_mass_sha256=cfg["data"]["cell_mass_sha256"],
                    config_sha256=sha256_file(args.config)),
        runtime=dict(started_unix=started, elapsed_seconds=time.time() - started, gpu=torch.cuda.get_device_name(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(), python=platform.python_version(),
                     torch=torch.__version__, host=platform.node()),
        slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        logical_nvila_rollouts=0, training_updates_run=0)
    OmegaConf.save(OmegaConf.create(cfg), args.output / "config.yaml")
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"clips": n, "boundary_ties_at_k": manifest["boundary_ties_at_k"],
                      "elapsed_seconds": manifest["runtime"]["elapsed_seconds"]}))


def load_saliency(directory, records, cfg):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["inputs"]["manifest_sha256"] != cfg["data"]["manifest_sha256"]:
        raise ValueError("Saliency export is not a complete export of the frozen manifest")
    for name, digest in manifest["files_sha256"].items():
        if name != "maps_uint8.npy" and sha256_file(directory / name) != digest:
            raise ValueError(f"Saliency {name} checksum mismatch")
    rows = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != len(records):
        raise ValueError("Saliency export does not cover the test split")
    for row, record in zip(rows, records):
        if any(row[k] != record[k] for k in ("clip_id", "source", "video_id", "frame_numbers")):
            raise ValueError(f"Saliency row differs from manifest for {record['clip_id']}")
    actions = np.asarray([r["fine_cells"] for r in rows], dtype=np.int64)
    predicted = np.load(directory / "predicted_cell_mass.npy")
    if not np.array_equal(top_k_cells(predicted, K), actions):
        raise ValueError("Saved top-16 cells differ from the saved predicted cell mass")
    return manifest, actions


def load_reference(cfg, records, all_records, cache):
    ref = cfg["arms"]["reference"]
    root = Path(ref["gt1_actions"])
    actions = {}
    for name, digest in ref["actions_sha256"].items():
        if sha256_file(root / name / "actions.jsonl") != digest:
            raise ValueError(f"GT1 {name} actions changed")
        rows = [json.loads(line) for line in (root / name / "actions.jsonl").read_text().splitlines() if line.strip()]
        if [r["clip_id"] for r in rows] != [r["clip_id"] for r in records]:
            raise ValueError(f"GT1 {name} clip order differs")
        if any(r["frame_numbers"] != rec["frame_numbers"] for r, rec in zip(rows, records)):
            raise ValueError(f"GT1 {name} frame geometry differs")
        actions[name] = np.asarray([r["fine_cells"] for r in rows], dtype=np.int64)
    actions["center16"] = np.ascontiguousarray(np.broadcast_to(center_order(14)[:K].numpy(), (len(records), K, K)))
    prior = source_prior_cells(all_records, cell_mass_for_records(all_records, cache))
    actions["train_source_prior"] = np.asarray([[prior[r["source"]]] * K for r in records], dtype=np.int64)
    train = [r for r in all_records if r["split"] == "train"]
    best_cells, _ = best_constant_cells(train, cell_mass_for_records(train, cache), k=K)
    actions["best_constant"] = np.asarray([[best_cells] * K for _ in records], dtype=np.int64)
    return actions, {"train_source_prior": prior, "best_constant": best_cells}


def implementation_check(cfg, reports):
    ref = cfg["arms"]["reference"]
    path = ROOT / ref["gt1_metrics"]
    if sha256_file(path) != ref["gt1_metrics_sha256"]:
        raise ValueError("GT1 metrics changed")
    gt1 = json.loads(path.read_text())
    observed = {"rl_six_seed_mean": float(np.mean([reports[f"rl{s}"]["macro_source_mean"] for s in ref["seeds"]])),
                "center16": reports["center16"]["macro_source_mean"],
                "train_source_prior": reports["train_source_prior"]["macro_source_mean"]}
    gt1_values = {"rl_six_seed_mean": gt1["rl_seed_summary"]["mean"],
                  "center16": gt1["per_model"]["center16"]["macro_source_mean"],
                  "train_source_prior": gt1["per_model"]["train_source_prior"]["macro_source_mean"]}
    targets = cfg["analysis"]["implementation_check"]["targets"]
    tol = cfg["analysis"]["implementation_check"]["tolerance"]
    check = {k: {"observed": observed[k], "gt1": gt1_values[k], "frozen_target": targets[k],
                 "abs_error": abs(observed[k] - gt1_values[k])} for k in observed}
    passed = all(v["abs_error"] <= tol and abs(v["gt1"] - v["frozen_target"]) <= 1e-12 for v in check.values())
    return check, passed, gt1


def score(records, mass, actions):
    return {name: aggregate_frame_values(records, selected_mass(mass, a)) for name, a in actions.items()}


def check_only(args, cfg, all_records, records):
    cache = np.load(cfg["data"]["cell_mass"], mmap_mode="r")
    mass = cell_mass_for_records(records, cache)
    actions, fitted = load_reference(cfg, records, all_records, cache)
    actions.pop("best_constant")  # test coverage of the new constant is a result; not inspected here
    check, passed, _ = implementation_check(cfg, score(records, mass, actions))
    print(json.dumps({"pass": passed, "values": check, "best_constant_cells_train": fitted["best_constant"]}, indent=2))
    if not passed:
        raise AssertionError("Implementation check failed")


def curate(args, cfg, all_records, records):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = source_identity()
    started = time.time()
    cache = np.load(cfg["data"]["cell_mass"], mmap_mode="r")
    mass = cell_mass_for_records(records, cache)
    actions, fitted = load_reference(cfg, records, all_records, cache)
    export_manifest, actions["saliency_top16"] = load_saliency(args.saliency, records, cfg)
    reports = score(records, mass, actions)
    check, check_pass, gt1 = implementation_check(cfg, reports)
    tol = cfg["analysis"]["implementation_check"]["tolerance"]
    if not check_pass:
        write_json(args.output.parent / f"{args.output.name}.implementation_check_failed.json", check)
        raise AssertionError(f"Implementation check failed: {check}")

    seeds = cfg["arms"]["reference"]["seeds"]
    rl = [reports[f"rl{s}"] for s in seeds]
    sal = reports["saliency_top16"]
    a = cfg["analysis"]
    boot = dict(iterations=a["bootstrap_iterations"], seed=a["bootstrap_seed"])
    primary = source_stratified_video_bootstrap_delta(rl, [sal] * 6, expected_seed_count=6, **boot)
    decision = ("policy_better" if primary["ci90_low"] > 0 else
                "saliency_better" if primary["ci90_high"] < 0 else "inconclusive")
    secondary = {
        "saliency_minus_center16": source_stratified_video_bootstrap_delta(
            [sal], [reports["center16"]], expected_seed_count=1, **boot),
        "saliency_minus_train_source_prior": source_stratified_video_bootstrap_delta(
            [sal], [reports["train_source_prior"]], expected_seed_count=1, **boot),
        "trained_minus_best_constant": source_stratified_video_bootstrap_delta(
            rl, [reports["best_constant"]] * 6, expected_seed_count=6, **boot),
    }

    overlap = {}
    sal_actions = actions["saliency_top16"]
    seed_overlaps = [aggregate_frame_values(records, action_intersection(sal_actions, actions[f"rl{s}"]) / K)
                     for s in seeds]
    overlap["saliency_vs_trained"] = {
        "macro_source_mean": float(np.mean([r["macro_source_mean"] for r in seed_overlaps])),
        "pooled_frame_mean": float(np.mean([r["pooled_frame_mean"] for r in seed_overlaps])),
        "per_seed_macro": {f"rl{s}": r["macro_source_mean"] for s, r in zip(seeds, seed_overlaps)},
        "per_source": {src: float(np.mean([r["per_source"][src]["mean_video_value"] for r in seed_overlaps]))
                       for src in STAVIS_SOURCES}}
    centre = aggregate_frame_values(records, action_intersection(sal_actions, actions["center16"]) / K)
    overlap["saliency_vs_center16"] = {
        "macro_source_mean": centre["macro_source_mean"], "pooled_frame_mean": centre["pooled_frame_mean"],
        "per_source": {src: centre["per_source"][src]["mean_video_value"] for src in STAVIS_SOURCES}}
    trained_centre = [aggregate_frame_values(records, action_intersection(actions[f"rl{s}"], actions["center16"]) / K)
                      for s in seeds]
    overlap["trained_vs_center16_reference"] = {
        "macro_source_mean": float(np.mean([r["macro_source_mean"] for r in trained_centre]))}

    rl_mean = float(np.mean([r["macro_source_mean"] for r in rl]))
    per_arm = {name: {"coverage_macro": rep["macro_source_mean"], "coverage_pooled_frame": rep["pooled_frame_mean"],
                      "per_source": {s: rep["per_source"][s]["mean_video_value"] for s in STAVIS_SOURCES}}
               for name, rep in reports.items()}
    per_arm["rl_six_seed_mean"] = {
        "coverage_macro": rl_mean,
        "per_source": {s: float(np.mean([r["per_source"][s]["mean_video_value"] for r in rl])) for s in STAVIS_SOURCES}}

    metrics = dict(
        schema_version=1, status="complete", run_id=args.run_id, gate="SB1", model="TASED-Net (DHF1K, TASED_updated.pt)",
        decision=decision, primary={"contrast": "trained_rl_k16_six_seed_mean_minus_saliency_top16", **primary},
        secondary=secondary, overlap=overlap,
        implementation_check={"pass": check_pass, "tolerance": tol, "values": check},
        rl_seed_summary=seed_summary([r["macro_source_mean"] for r in rl], expected_count=6),
        gt1_contrasts_reference=gt1["contrasts"],
        constants={"best_constant_cells_train": fitted["best_constant"],
                   "train_source_prior_cells": fitted["train_source_prior"],
                   "center16_cells": center_order(14)[:K].tolist()},
        export={"boundary_ties_at_k": export_manifest["boundary_ties_at_k"],
                "frames_with_reversed_window": export_manifest["frames_with_reversed_window"]},
        per_arm=per_arm, reports=reports, prior_exposure=cfg["data"]["prior_exposure"])

    args.output.mkdir(parents=True)
    write_json(args.output / "metrics.json", metrics)
    OmegaConf.save(OmegaConf.create(cfg), args.output / "config.yaml")
    order = ["saliency_top16", "rl_six_seed_mean", "best_constant", "train_source_prior", "center16",
             *[f"rl{s}" for s in seeds]]
    with (args.output / "coverage_by_source.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "macro", *STAVIS_SOURCES])
        for name in order:
            writer.writerow([name, f"{per_arm[name]['coverage_macro']:.6f}",
                             *[f"{per_arm[name]['per_source'][s]:.6f}" for s in STAVIS_SOURCES]])

    def fmt(c):
        return f"{c['difference']:+.4f} [{c['ci90_low']:+.4f}, {c['ci90_high']:+.4f}]"
    lines = [
        "# SB1 Saliency model baseline (TASED-Net, DHF1K)", "",
        f"Run `{args.run_id}`. Gate note: `wiki/experiments/SB1 Saliency Model Baseline.md` (wrapper).", "",
        f"Implementation check (GT1 trained six-seed mean, Center16, train source prior within {tol:g}): "
        f"{'pass' if check_pass else 'FAIL'}.", "",
        f"Decision: **{decision}**. Trained RL K16 (six-seed mean) minus Saliency-top16, source-macro coverage: "
        f"{fmt(primary)} (paired source-stratified video bootstrap, 90% interval, {boot['iterations']} draws, "
        f"seed {boot['seed']}).", "",
        "| arm | coverage (source-macro) |", "|---|---|",
        *[f"| {n} | {per_arm[n]['coverage_macro']:.4f} |" for n in order[:5]], "",
        "Secondary contrasts (descriptive):", "",
        *[f"- {k}: {fmt(v)}" for k, v in secondary.items()], "",
        f"Cell overlap |S ∩ X| / 16 (source-macro): saliency vs trained (seed mean) "
        f"{overlap['saliency_vs_trained']['macro_source_mean']:.3f}; saliency vs Center16 "
        f"{overlap['saliency_vs_center16']['macro_source_mean']:.3f}; trained vs Center16 "
        f"{overlap['trained_vs_center16_reference']['macro_source_mean']:.3f}.", "",
        "Saliency-top16: TASED-Net (MichiganCOG/TASED-Net @ 0cf47b7, TASED_updated.pt, trained on DHF1K with a "
        "Kinetics-400 S3D encoder; no STAViS source) with its run_example.py protocol (32-frame window of native "
        "frames ending at each target frame, 224x384 input, 8-bit + Gaussian sigma 7 postprocessing), map resized to "
        "224x224 (PIL bilinear), top-16 of 14x14 cell masses, ties by cell index. Best constant: 16 cells with the "
        "largest training attention mass under the evaluation weights. See config.yaml.", "",
        cfg["data"]["prior_exposure"], "",
        "Files: metrics.json, config.yaml, manifest.json, coverage_by_source.csv.", ""]
    (args.output / "README.md").write_text("\n".join(lines))
    write_json(args.output / "manifest.json", dict(
        schema_version=1, status=metrics["status"], run_id=args.run_id, gate="SB1", source=source,
        data={k: cfg["data"][k] for k in ("root", "manifest", "manifest_sha256", "cell_mass", "cell_mass_sha256")},
        model={k: cfg["tased_net"][k] for k in ("repository", "commit", "weights_url", "weights_sha256", "model_py_sha256")},
        saliency_export={"manifest": str(args.saliency / "manifest.json"),
                         "manifest_sha256": sha256_file(args.saliency / "manifest.json"),
                         "files_sha256": export_manifest["files_sha256"], "source": export_manifest["source"],
                         "runtime": export_manifest["runtime"],
                         "slurm_job_id": export_manifest["slurm"].get("SLURM_JOB_ID")},
        reference_actions=cfg["arms"]["reference"]["actions_sha256"],
        config_sha256=sha256_file(args.config), resolved_config_sha256=sha256_file(args.output / "config.yaml"),
        curation_runtime=dict(elapsed_seconds=time.time() - started, host=platform.node(),
                              slurm_job_id=os.environ.get("SLURM_JOB_ID")),
        heavy_artifacts=str(args.saliency), logical_nvila_rollouts=0, training_updates_run=0))
    print(json.dumps({"status": metrics["status"], "decision": decision, "check_pass": check_pass,
                      "output": str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "check", "curate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--saliency", type=Path, help="artifact directory of the complete saliency export")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int, help="smoke only: first N test clips, status=smoke")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    cfg, all_records, records = load_inputs(args.config)
    if args.command == "check":
        check_only(args, cfg, all_records, records)
        return
    if args.output is None or args.run_id is None:
        parser.error("export and curate require --output and --run-id")
    if args.command == "export":
        export(args, cfg, records)
    else:
        if args.saliency is None:
            parser.error("curate requires --saliency")
        curate(args, cfg, all_records, records)


if __name__ == "__main__":
    main()
