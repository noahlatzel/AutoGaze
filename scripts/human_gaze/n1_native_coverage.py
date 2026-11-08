#!/usr/bin/env python3
"""Gate N1: native multi-scale AutoGaze action export (GPU) and pixel-union coverage curation (CPU).

Inference only; never training, NVILA or HLVid.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES, load_aligned_clip, read_manifest
from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.native_coverage import (
    LAYOUT, IMAGE_SIZE, _centre_ring_masses, centre_area_coverage, fine_cells_to_actions,
    frame_coverage, native_actions_from_generation, normalise_heatmap, scale_counts,
    validate_frame_actions,
)
from autogaze.human_gaze.supervised_analysis import (
    aggregate_frame_values, cell_mass_for_records, selected_mass, seed_summary,
    sha256_file, sha256_tree, source_stratified_video_bootstrap_delta,
)

DEFAULT_CONFIG = ROOT / "experiments/human_gaze/configs/n1_native_autogaze_coverage_v1.yaml"
ARMS = ("native16", "native_default")


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
        raise ValueError("Evidential N1 runs require a clean tracked checkout")
    return {"repository": git("remote", "get-url", "origin"), "commit": git("rev-parse", "HEAD"), "dirty": dirty}


def load_inputs(config):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if cfg["experiment_id"] != "n1_native_autogaze_coverage" or cfg["authorization"]["training_authorized"]:
        raise ValueError("Unexpected inference-only contract")
    for name in ("manifest", "cell_mass"):
        if sha256_file(Path(cfg["data"][name])) != cfg["data"][name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    records = read_manifest(cfg["data"]["manifest"], split="test")
    if len(records) != cfg["data"]["expected_clips"]:
        raise ValueError("Unexpected test population")
    if len({(r["source"], r["video_id"]) for r in records}) != cfg["data"]["expected_videos"]:
        raise ValueError("Unexpected test video count")
    geometry = cfg["geometry"]
    for block, (name, spec) in zip(LAYOUT, geometry["action_blocks"].items()):
        if [block["offset"], block["offset"] + block["count"] - 1] != spec["ids"] or block["side_px"] != spec["patch_side_px"]:
            raise ValueError(f"Code geometry differs from frozen config for {name}")
    return cfg, records


def checkpoint_hashes(cfg):
    path = Path(cfg["pretrained"]["path"])
    hashes = {"model.safetensors": sha256_file(path / "model.safetensors"),
              "config.json": sha256_file(path / "config.json"),
              "preprocessor_config.json": sha256_file(path / "preprocessor_config.json")}
    expected = {"model.safetensors": cfg["pretrained"]["model_safetensors_sha256"],
                "config.json": cfg["pretrained"]["config_sha256"],
                "preprocessor_config.json": cfg["pretrained"]["preprocessor_config_sha256"]}
    if hashes != expected:
        raise ValueError("Pretrained AutoGaze checkpoint changed")
    return path, {"path": str(path), "files_sha256": hashes, "tree_sha256": sha256_tree(path)}


def arm_kwargs(arm):
    if arm == "native16":
        return {"max_gaze_tokens_each_frame": 16}, 16
    if arm == "native_default":
        return {"gazing_ratio": 0.75, "task_loss_requirement": 0.7}, int(0.75 * 265)
    raise ValueError(arm)


def export(args, cfg, records):
    import torch
    from torch.utils.data import DataLoader, Subset
    from autogaze.datasets.av_gaze_stavis import AVGazeStavisDataset
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

    if args.output.exists():
        raise FileExistsError(args.output)
    # Full FP32 (the checkpoint's sdpa/fp32 mode): cuDNN TF32 convolutions made greedy
    # decoding depend on batch composition at logit near-ties in the pre-submission smoke.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    smoke = args.limit is not None
    source = source_identity(allow_dirty=smoke)
    checkpoint, provenance = checkpoint_hashes(cfg)
    kwargs, cap = arm_kwargs(args.arm)
    processor = AutoGazeImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoGaze.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    if model.scales != cfg["geometry"]["scales"] or model.num_vision_tokens_each_scale_each_frame != [b["count"] for b in LAYOUT]:
        raise ValueError("Loaded model geometry differs from the frozen layout")
    dataset = AVGazeStavisDataset(root=cfg["data"]["root"], manifest_path=cfg["data"]["manifest"], split="test",
                                  load_rgb=True, load_heatmap=False, cell_mass_path=cfg["data"]["cell_mass"],
                                  image_processor=processor)
    if smoke:
        dataset = Subset(dataset, list(range(args.limit)))
    batch_size = args.batch_size or cfg["resources"]["batch_size"]
    if not smoke and batch_size != cfg["resources"]["batch_size"]:
        raise ValueError("Evidential exports must use the frozen batch size")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=cfg["resources"]["num_workers"],
                        shuffle=False, pin_memory=True)
    args.output.mkdir(parents=True)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    written, frames_with_kept_after_stop, frames_at_cap = 0, 0, 0
    counts = []
    with (args.output / "actions.jsonl").open("x") as handle, torch.inference_mode():
        for batch in loader:
            gaze = model({"video": batch["video"].cuda(non_blocking=True)}, generate_only=True, **kwargs)
            padded = gaze["if_padded_gazing"].cpu().numpy()
            clips = native_actions_from_generation(gaze["gazing_pos"].cpu().numpy(), padded,
                                                   gaze["num_gazing_each_frame"].cpu().numpy())
            offset = 0
            for frame_count in gaze["num_gazing_each_frame"].tolist():
                block = padded[:, offset:offset + frame_count]
                # A kept action after a dropped (stop-padded) slot in the same frame.
                kept_after = (np.maximum.accumulate(block, axis=1)[:, :-1] & ~block[:, 1:]).any(axis=1)
                frames_with_kept_after_stop += int(kept_after.sum())
                offset += frame_count
            for i, frames in enumerate(clips):
                if len(frames) != 16:
                    raise ValueError("Clip does not have 16 frames")
                for frame in frames:
                    validate_frame_actions(frame, max_actions=cap)
                    counts.append(len(frame))
                    frames_at_cap += len(frame) == cap
                if args.arm == "native16" and any(len(f) != 16 for f in frames):
                    raise ValueError("Native-16 exact budget violated")
                row = {key: batch[key][i] for key in ("clip_id", "source", "video_id")}
                row.update(frame_numbers=[int(v) for v in batch["frame_numbers"][i]], actions=frames)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
    expected = args.limit if smoke else cfg["data"]["expected_clips"]
    if written != expected:
        raise ValueError("Incomplete test inference")
    torch.cuda.synchronize()
    manifest = dict(
        schema_version=1, status="smoke" if smoke else "complete", experiment_id=cfg["experiment_id"],
        run_id=args.run_id, arm=args.arm, generation_kwargs=kwargs, max_actions_per_frame=cap,
        allow_eos=False, greedy=True, selection_without_replacement=True, split="test", clip_len=16,
        num_clips=written, batch_size=batch_size,
        numerics=dict(dtype="float32", cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                      matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        action_counts=dict(mean=float(np.mean(counts)), min=int(np.min(counts)), max=int(np.max(counts)),
                           frames_at_cap=frames_at_cap, frames_with_kept_action_after_stop=frames_with_kept_after_stop),
        source=source, checkpoint=provenance, actions_sha256=sha256_file(args.output / "actions.jsonl"),
        inputs=dict(manifest_sha256=cfg["data"]["manifest_sha256"], cell_mass_sha256=cfg["data"]["cell_mass_sha256"],
                    config_sha256=sha256_file(args.config)),
        runtime=dict(started_unix=started, elapsed_seconds=time.time() - started, gpu=torch.cuda.get_device_name(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(), python=platform.python_version(),
                     torch=torch.__version__, host=platform.node()),
        slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        logical_nvila_rollouts=0)
    OmegaConf.save(OmegaConf.create(cfg), args.output / "config.yaml")
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"arm": args.arm, "clips": written, **manifest["action_counts"],
                      "elapsed_seconds": manifest["runtime"]["elapsed_seconds"]}))


def load_native(directory, records, arm, cfg):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"{arm} export is not complete")
    if manifest["arm"] != arm or manifest["inputs"]["manifest_sha256"] != cfg["data"]["manifest_sha256"]:
        raise ValueError(f"{arm} export provenance mismatch")
    if sha256_file(directory / "actions.jsonl") != manifest["actions_sha256"]:
        raise ValueError(f"{arm} actions checksum mismatch")
    rows = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != len(records):
        raise ValueError(f"{arm} export covers {len(rows)} clips, expected {len(records)}")
    cap = manifest["max_actions_per_frame"]
    for row, record in zip(rows, records):
        if any(row[k] != record[k] for k in ("clip_id", "source", "video_id", "frame_numbers")):
            raise ValueError(f"{arm} export row differs from manifest for {record['clip_id']}")
        for frame in row["actions"]:
            validate_frame_actions(frame, max_actions=cap)
    return manifest, [row["actions"] for row in rows]


def load_reference(cfg, records):
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
    actions["center16"] = np.broadcast_to(center_order(14)[:16].numpy(), (len(records), 16, 16))
    return actions


def score_clip(args):
    """Per-frame coverage, area, and centre@area for every arm on one clip."""
    root, record, fine_arms, native_arms = args
    _, heat = load_aligned_clip(root, record, load_rgb=False, load_heatmap=True)
    out = {}
    for t in range(heat.shape[0]):
        h = normalise_heatmap(heat[t].numpy())
        ring = _centre_ring_masses(h)
        for name, cells in fine_arms.items():
            cov, area = frame_coverage(h, fine_cells_to_actions(cells[t]))
            out.setdefault(name, []).append((cov, area, np.nan, 16, *[0, 0, 0, 16]))
        for name, frames in native_arms.items():
            cov, area = frame_coverage(h, frames[t])
            centre = centre_area_coverage(h, area, ring)
            out.setdefault(name, []).append((cov, area, centre, len(frames[t]), *scale_counts(frames[t]).tolist()))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def implementation_check(cfg, reports):
    """Frozen N1 check: pixel-union scores reproduce GT1 fine-only source-macro values."""
    ref = cfg["arms"]["reference"]
    path = ROOT / ref["gt1_metrics"]
    if sha256_file(path) != ref["gt1_metrics_sha256"]:
        raise ValueError("GT1 metrics changed")
    gt1 = json.loads(path.read_text())
    observed = {"rl_six_seed_mean": float(np.mean([reports[f"rl{s}"]["macro_source_mean"] for s in ref["seeds"]])),
                "center16": reports["center16"]["macro_source_mean"],
                "pretrained": reports["pretrained"]["macro_source_mean"]}
    gt1_values = {"rl_six_seed_mean": gt1["rl_seed_summary"]["mean"],
                  "center16": gt1["per_model"]["center16"]["macro_source_mean"],
                  "pretrained": gt1["per_model"]["pretrained"]["macro_source_mean"]}
    targets = cfg["analysis"]["implementation_check"]["targets"]
    tol = cfg["analysis"]["implementation_check"]["tolerance"]
    check = {k: {"observed": observed[k], "gt1": gt1_values[k], "frozen_target": targets[k],
                 "abs_error": abs(observed[k] - gt1_values[k])} for k in observed}
    passed = all(v["abs_error"] <= tol and abs(v["gt1"] - v["frozen_target"]) <= 1e-12 for v in check.values())
    return check, passed, gt1


def score_records(cfg, records, reference, native, workers):
    jobs = [(cfg["data"]["root"], rec, {k: v[i] for k, v in reference.items()}, {k: v[i] for k, v in native.items()})
            for i, rec in enumerate(records)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_clip = list(pool.map(score_clip, jobs))
    return {name: np.stack([clip[name] for clip in per_clip]) for name in per_clip[0]}  # [clips, 16, 8]


def check_only(args, cfg, records):
    """CPU pre-submission check over all test clips using only the saved fine-only actions."""
    reference = load_reference(cfg, records)
    arrays = score_records(cfg, records, reference, {}, args.workers)
    reports = {name: aggregate_frame_values(records, arrays[name][..., 0]) for name in arrays}
    check, passed, _ = implementation_check(cfg, reports)
    mass = cell_mass_for_records(records, np.load(cfg["data"]["cell_mass"], mmap_mode="r"))
    max_abs = {name: float(np.abs(arrays[name][..., 0] - selected_mass(mass, np.ascontiguousarray(reference[name]))).max())
               for name in reference}
    print(json.dumps({"pass": passed, "values": check, "pixel_vs_cell_mass_max_abs_frame_error": max_abs}, indent=2))
    if not passed:
        raise AssertionError("Implementation check failed")


def curate(args, cfg, records):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = source_identity()
    started = time.time()
    reference = load_reference(cfg, records)
    native_manifests, native = {}, {}
    for arm in ARMS:
        native_manifests[arm], native[arm] = load_native(args.actions / arm, records, arm, cfg)
    arrays = score_records(cfg, records, reference, native, args.workers)
    total_px = IMAGE_SIZE * IMAGE_SIZE

    # Cross-check the pixel scorer against the 14x14 cell-mass cache for fine-only arms.
    cache = np.load(cfg["data"]["cell_mass"], mmap_mode="r")
    mass = cell_mass_for_records(records, cache)
    fine_max_abs = {name: float(np.abs(arrays[name][..., 0] - selected_mass(mass, np.ascontiguousarray(reference[name]))).max())
                    for name in reference}

    reports = {name: aggregate_frame_values(records, arrays[name][..., 0]) for name in arrays}
    area_reports = {name: aggregate_frame_values(records, arrays[name][..., 1] / total_px) for name in arrays}
    count_reports = {name: aggregate_frame_values(records, arrays[name][..., 3]) for name in arrays}
    for arm in ARMS:
        reports[f"centre_at_area_{arm}"] = aggregate_frame_values(records, arrays[arm][..., 2])

    seeds = cfg["arms"]["reference"]["seeds"]
    rl = [reports[f"rl{s}"] for s in seeds]
    rl_mean = float(np.mean([r["macro_source_mean"] for r in rl]))
    check, check_pass, gt1 = implementation_check(cfg, reports)
    tol = cfg["analysis"]["implementation_check"]["tolerance"]
    if not check_pass:
        write_json(args.output.parent / f"{args.output.name}.implementation_check_failed.json", check)
        raise AssertionError(f"Implementation check failed: {check}")

    a = cfg["analysis"]
    boot = dict(iterations=a["bootstrap_iterations"], seed=a["bootstrap_seed"])
    primary = source_stratified_video_bootstrap_delta(rl, [reports["native16"]] * 6, expected_seed_count=6, **boot)
    decision = ("trained_better" if primary["ci90_low"] > 0 else
                "native_better" if primary["ci90_high"] < 0 else "inconclusive")
    secondary = {
        "trained_minus_native_default": source_stratified_video_bootstrap_delta(
            rl, [reports["native_default"]] * 6, expected_seed_count=6, **boot),
        "trained_minus_center16": source_stratified_video_bootstrap_delta(
            rl, [reports["center16"]] * 6, expected_seed_count=6, **boot),
        "trained_minus_pretrained_fine16": source_stratified_video_bootstrap_delta(
            rl, [reports["pretrained"]] * 6, expected_seed_count=6, **boot),
    }
    for arm in ARMS:
        secondary[f"{arm}_minus_centre_at_area"] = source_stratified_video_bootstrap_delta(
            [reports[arm]], [reports[f"centre_at_area_{arm}"]], expected_seed_count=1, **boot)

    per_arm = {}
    for name in arrays:
        cov = reports[name]["macro_source_mean"]
        area = area_reports[name]["macro_source_mean"]
        entry = {"coverage_macro": cov, "coverage_pooled_frame": reports[name]["pooled_frame_mean"],
                 "area_fraction_macro": area, "coverage_per_unit_area": cov / area,
                 "actions_per_frame_macro": count_reports[name]["macro_source_mean"],
                 "actions_per_frame_pooled": float(arrays[name][..., 3].mean()),
                 "actions_per_frame_min": int(arrays[name][..., 3].min()),
                 "actions_per_frame_max": int(arrays[name][..., 3].max()),
                 "scale_counts_pooled_mean": dict(zip(("grid2x2", "grid4x4", "grid7x7", "grid14x14"),
                                                      arrays[name][..., 4:].mean(axis=(0, 1)).tolist())),
                 "per_source": {s: reports[name]["per_source"][s]["mean_video_value"] for s in STAVIS_SOURCES}}
        if name in ARMS:
            entry["centre_at_area_macro"] = reports[f"centre_at_area_{name}"]["macro_source_mean"]
            entry["centre_at_area_per_source"] = {
                s: reports[f"centre_at_area_{name}"]["per_source"][s]["mean_video_value"] for s in STAVIS_SOURCES}
            entry["frames_at_cap"] = int((arrays[name][..., 3] == native_manifests[name]["max_actions_per_frame"]).sum())
        per_arm[name] = entry
    rl_entries = [per_arm[f"rl{s}"] for s in seeds]
    per_arm["rl_six_seed_mean"] = {
        "coverage_macro": rl_mean,
        "area_fraction_macro": float(np.mean([e["area_fraction_macro"] for e in rl_entries])),
        "coverage_per_unit_area": rl_mean / float(np.mean([e["area_fraction_macro"] for e in rl_entries])),
        "actions_per_frame_macro": 16.0,
        "per_source": {s: float(np.mean([e["per_source"][s] for e in rl_entries])) for s in STAVIS_SOURCES}}

    metrics = dict(
        schema_version=1, status="complete", run_id=args.run_id,
        gate="N1", decision=decision, primary={"contrast": "trained_rl_k16_six_seed_mean_minus_native16", **primary},
        secondary=secondary, gt1_trained_minus_center16_reference=gt1["contrasts"]["center16"],
        implementation_check={"pass": check_pass, "tolerance": tol, "values": check,
                              "pixel_vs_cell_mass_max_abs_frame_error": fine_max_abs},
        rl_seed_summary=seed_summary([r["macro_source_mean"] for r in rl], expected_count=6),
        per_arm=per_arm, reports=reports, area_reports=area_reports,
        native_export_action_counts={arm: native_manifests[arm]["action_counts"] for arm in ARMS},
        prior_exposure=cfg["data"]["prior_exposure"])

    args.output.mkdir(parents=True)
    write_json(args.output / "metrics.json", metrics)
    OmegaConf.save(OmegaConf.create(cfg), args.output / "config.yaml")
    order = ["native16", "native_default", "rl_six_seed_mean", "center16", "pretrained"] + [f"rl{s}" for s in seeds]
    with (args.output / "coverage_by_source.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "macro", *STAVIS_SOURCES])
        for name in order:
            writer.writerow([name, f"{per_arm[name]['coverage_macro']:.6f}",
                             *[f"{per_arm[name]['per_source'][s]:.6f}" for s in STAVIS_SOURCES]])
        for arm in ARMS:
            writer.writerow([f"centre_at_area_{arm}", f"{per_arm[arm]['centre_at_area_macro']:.6f}",
                             *[f"{per_arm[arm]['centre_at_area_per_source'][s]:.6f}" for s in STAVIS_SOURCES]])
    with (args.output / "arm_summary.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        fields = ["coverage_macro", "area_fraction_macro", "coverage_per_unit_area", "actions_per_frame_macro"]
        writer.writerow(["arm", *fields])
        for name in order:
            writer.writerow([name, *[f"{per_arm[name][f]:.6f}" for f in fields]])

    def fmt(c):
        return f"{c['difference']:+.4f} [{c['ci90_low']:+.4f}, {c['ci90_high']:+.4f}]"
    lines = [
        "# N1 Native AutoGaze coverage", "",
        f"Run `{args.run_id}`. Gate note: `wiki/experiments/N1 Native AutoGaze Coverage.md` (wrapper).", "",
        f"Implementation check (pixel-union scorer reproduces GT1 fine-only values within {tol:g}): "
        f"{'pass' if check_pass else 'FAIL'}.", "",
        f"Decision: **{decision}**. Trained RL K16 (six-seed mean) minus Native-16, source-macro coverage: "
        f"{fmt(primary)} (paired source-stratified video bootstrap, 90% interval, {boot['iterations']} draws, seed {boot['seed']}).", "",
        "| arm | coverage | area fraction | coverage / area | actions per frame |", "|---|---|---|---|---|",
        *[f"| {n} | {per_arm[n]['coverage_macro']:.4f} | {per_arm[n]['area_fraction_macro']:.4f} | "
          f"{per_arm[n]['coverage_per_unit_area']:.3f} | {per_arm[n]['actions_per_frame_macro']:.2f} |"
          for n in ["native16", "native_default", "rl_six_seed_mean", "center16", "pretrained"]], "",
        "Secondary contrasts (descriptive):", "",
        *[f"- {k}: {fmt(v)}" for k, v in secondary.items()], "",
        "Coverage is the normalised attention mass inside the pixel union of the selected patches "
        "(2x2/4x4/7x7/14x14 patches are 112/56/32/16 px on the 224-px map), aggregated frames -> clips -> videos -> "
        "equal-weight sources. Centre@area: exact-area centred square with fractional boundary ring (see config.yaml).", "",
        cfg["data"]["prior_exposure"], "",
        "Files: metrics.json, config.yaml, manifest.json, coverage_by_source.csv, arm_summary.csv.", ""]
    (args.output / "README.md").write_text("\n".join(lines))
    write_json(args.output / "manifest.json", dict(
        schema_version=1, status=metrics["status"], run_id=args.run_id, gate="N1", source=source,
        data={k: cfg["data"][k] for k in ("root", "manifest", "manifest_sha256", "cell_mass", "cell_mass_sha256")},
        model={"path": cfg["pretrained"]["path"], "model_safetensors_sha256": cfg["pretrained"]["model_safetensors_sha256"]},
        native_exports={arm: {"manifest": str(args.actions / arm / "manifest.json"),
                              "manifest_sha256": sha256_file(args.actions / arm / "manifest.json"),
                              "actions_sha256": native_manifests[arm]["actions_sha256"],
                              "source": native_manifests[arm]["source"],
                              "runtime": native_manifests[arm]["runtime"],
                              "slurm_job_id": native_manifests[arm]["slurm"].get("SLURM_JOB_ID")} for arm in ARMS},
        reference_actions=cfg["arms"]["reference"]["actions_sha256"],
        config_sha256=sha256_file(args.config), resolved_config_sha256=sha256_file(args.output / "config.yaml"),
        curation_runtime=dict(elapsed_seconds=time.time() - started, host=platform.node(),
                              slurm_job_id=os.environ.get("SLURM_JOB_ID")),
        heavy_artifacts=str(args.actions), logical_nvila_rollouts=0, training_updates_run=0))
    print(json.dumps({"status": metrics["status"], "decision": decision, "check_pass": check_pass,
                      "output": str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "check", "curate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--actions", type=Path, help="artifact directory containing native16/ and native_default/")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int, help="smoke only: first N test clips, status=smoke")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg, records = load_inputs(args.config)
    if args.command == "check":
        check_only(args, cfg, records)
        return
    if args.output is None or args.run_id is None:
        parser.error("export and curate require --output and --run-id")
    if args.command == "export":
        if args.arm is None:
            parser.error("export requires --arm")
        export(args, cfg, records)
    else:
        if args.actions is None:
            parser.error("curate requires --actions")
        curate(args, cfg, records)


if __name__ == "__main__":
    main()
