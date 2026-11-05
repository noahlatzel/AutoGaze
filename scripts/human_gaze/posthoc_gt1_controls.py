#!/usr/bin/env python3
"""Post-hoc CPU analysis of saved GT1 fixed-K16 test actions and HLVid answers.

No model loads, inference or training. Refuses an existing output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES, read_manifest
from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.posthoc_analysis import (
    overlap_values, pairwise_disagreement, pooled_prior_cells, report_from_video_values,
    video_cluster_bootstrap_mean_difference, video_pooled_bootstrap_delta, within_pairs,
)
from autogaze.human_gaze.supervised_analysis import (
    aggregate_frame_values, cell_mass_for_records, load_action_export, selected_mass,
    sha256_file, source_prior_cells, source_stratified_video_bootstrap_delta,
    within_source_different_video_indices,
)

INPUTS: dict[str, str] = {}


def track(path: Path) -> Path:
    path = Path(path)
    INPUTS[str(path)] = sha256_file(path)
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def video_pooled(report):
    return float(np.mean([v for s in report["per_source"].values() for v in s["video_values"].values()]))


def constant(cells, n):
    return np.broadcast_to(np.asarray(cells, dtype=np.int64), (n, 16, 16))


def contrast_row(name, split, left, right, cfg, *, left_label="rl_six_seed"):
    b = cfg["bootstrap"]
    macro = source_stratified_video_bootstrap_delta(left, right, iterations=b["iterations"],
        seed=b["seed"], expected_seed_count=len(left))
    pooled = video_pooled_bootstrap_delta(left, right, iterations=b["iterations"], seed=b["seed"])
    return {"split": split, "left": left_label, "right": name,
            "left_macro": float(np.mean([r["macro_source_mean"] for r in left])),
            "right_macro": float(np.mean([r["macro_source_mean"] for r in right])),
            "macro_difference": macro["difference"], "macro_ci90_low": macro["ci90_low"], "macro_ci90_high": macro["ci90_high"],
            "video_pooled_difference": pooled["difference"], "video_pooled_ci90_low": pooled["ci90_low"],
            "video_pooled_ci90_high": pooled["ci90_high"], "num_videos": pooled["num_videos"]}


def gaze_analysis(cfg):
    all_records = read_manifest(track(ROOT / cfg["data"]["manifest"]))
    require(INPUTS[str(ROOT / cfg["data"]["manifest"])] == cfg["data"]["manifest_sha256"], "manifest changed")
    cache = np.load(track(ROOT / cfg["data"]["cell_mass"]), mmap_mode="r")
    require(INPUTS[str(ROOT / cfg["data"]["cell_mass"])] == cfg["data"]["cell_mass_sha256"], "cell mass changed")
    all_mass = cell_mass_for_records(all_records, cache)
    test = [r for r in all_records if r["split"] == "test"]
    val = [r for r in all_records if r["split"] == "val"]
    test_mass = cell_mass_for_records(test, cache)
    val_mass = cell_mass_for_records(val, cache)
    n = len(test)
    source_prior = source_prior_cells(all_records, all_mass)
    pooled_prior = pooled_prior_cells(all_records, all_mass)
    center = center_order(14)[:16].tolist()
    prior_actions = np.asarray([[source_prior[r["source"]]] * 16 for r in test], dtype=np.int64)

    # Load GT1 exports and verify their manifests are the ones GT1 validated.
    gt1 = ROOT / cfg["gt1"]["bundle"]
    gt1_metrics = json.loads(track(gt1 / "metrics.json").read_text())
    expected = {Path(e["path"]).parent.name: e["sha256"]
                for e in json.loads(track(gt1 / "execution_manifest.json").read_text())["action_exports"]}
    actions = {}
    for name in [f"rl{s}" for s in cfg["gt1"]["seeds"]] + ["pretrained"]:
        directory = Path(cfg["gt1"]["actions_root"]) / name
        require(sha256_file(directory / "manifest.json") == expected[name], f"{name} manifest differs from GT1")
        track(directory / "manifest.json"); track(directory / "actions.jsonl")
        _, actions[name] = load_action_export(directory, test, expected_manifest_sha256=cfg["data"]["manifest_sha256"],
            expected_cell_mass_sha256=cfg["data"]["cell_mass_sha256"], expected_method=name, expected_split="test")
    rl_names = [f"rl{s}" for s in cfg["gt1"]["seeds"]]
    report = lambda acts: aggregate_frame_values(test, selected_mass(test_mass, acts))
    rl = [report(actions[k]) for k in rl_names]
    controls = {"train_source_prior16": report(prior_actions),
                "train_pooled_prior16": report(constant(pooled_prior, n)),
                "center16": report(constant(center, n)),
                "pretrained_fine_k16": report(actions["pretrained"])}

    # 0. Exact GT1 reproduction.
    b = cfg["bootstrap"]
    repro = source_stratified_video_bootstrap_delta(rl, [controls["train_source_prior16"]] * 6,
        iterations=b["iterations"], seed=b["seed"], expected_seed_count=6)
    tol = cfg["gt1"]["reproduction_tolerance"]
    for key in ("difference", "ci90_low", "ci90_high"):
        require(abs(repro[key] - gt1_metrics["primary"][key]) <= tol, f"GT1 {key} not reproduced")
    for name, key in (("train_source_prior16", "train_source_prior"), ("center16", "center16"), ("pretrained_fine_k16", "pretrained")):
        require(abs(controls[name]["macro_source_mean"] - gt1_metrics["per_model"][key]["macro_source_mean"]) <= tol, f"{name} not reproduced")

    # 2. Per-seed content controls.
    shuffle_idx = within_source_different_video_indices(test)
    seed_controls = {"same_source_shuffled_video": [], "static_top16_test_derived": [], "static_top16_validation_derived": []}
    val_static_cells, test_static_cells, val_rl = {}, {}, []
    for seed, name in zip(cfg["gt1"]["seeds"], rl_names):
        acts = actions[name]
        seed_controls["same_source_shuffled_video"].append(report(acts[shuffle_idx]))
        counts = np.bincount(acts.reshape(-1), minlength=196)
        test_static_cells[name] = sorted(range(196), key=lambda c: (-counts[c], c))[:16]
        seed_controls["static_top16_test_derived"].append(report(constant(test_static_cells[name], n)))
        diag_path = ROOT / cfg["validation"]["diagnostics_template"].format(
            seed=seed, training_seed=seed + cfg["validation"]["training_seed_offset"])
        diag = json.loads(track(diag_path).read_text())
        require(diag["split"] == "val" and diag["processed_clips"] == len(val), "unexpected diagnostics population")
        val_static_cells[name] = [int(c) for c in diag["diagnostics"]["selection_frequency_top16_cells"]]
        seed_controls["static_top16_validation_derived"].append(report(constant(val_static_cells[name], n)))
        val_rl.append(report_from_video_values({s: diag["coverage"]["actual"]["16"]["per_source"][s]["video_coverage"]
                                                for s in STAVIS_SOURCES}))

    contrasts = [contrast_row(k, "test", rl, [v] * 6, cfg) for k, v in controls.items()]
    contrasts += [contrast_row(k, "test", rl, v, cfg) for k, v in seed_controls.items()]

    # 4. Validation paired-video interval for K16 RL minus source prior.
    val_prior = aggregate_frame_values(val, selected_mass(val_mass, np.asarray(
        [[source_prior[r["source"]]] * 16 for r in val], dtype=np.int64)))
    val_pooled = aggregate_frame_values(val, selected_mass(val_mass, constant(pooled_prior, len(val))))
    ref = json.loads(track(ROOT / cfg["validation"]["reference_metrics"]).read_text())
    vt = cfg["validation"]["reference_tolerance"]
    require(abs(val_prior["macro_source_mean"] - ref["baselines"]["16"]["prior"]) <= vt, "validation prior not reproduced")
    for value, expected_value in zip([r["macro_source_mean"] for r in val_rl], ref["fixed_budget_endpoints"]["16"]["values"]):
        require(abs(value - expected_value) <= vt, "validation RL endpoint not reproduced")
    for report_ in val_rl:
        for s in STAVIS_SOURCES:
            require(set(report_["per_source"][s]["video_values"]) == set(val_prior["per_source"][s]["video_values"]),
                    "validation video sets differ")
    contrasts.append(contrast_row("train_source_prior16", "val", val_rl, [val_prior] * 6, cfg))
    contrasts.append(contrast_row("train_pooled_prior16", "val", val_rl, [val_pooled] * 6, cfg))

    # 3. Overlap on test.
    overlap_rows = []
    references = {"train_source_prior16": prior_actions, "train_pooled_prior16": constant(pooled_prior, n),
                  "center16": constant(center, n), "pretrained_fine_k16": actions["pretrained"]}
    for ref_name, ref_actions in references.items():
        per_seed = [{m: aggregate_frame_values(test, v)["macro_source_mean"]
                     for m, v in overlap_values(actions[k], ref_actions).items()} for k in rl_names]
        overlap_rows.append({"left": "rl_each_seed", "right": ref_name, "num_pairs": 6,
            **{f"{m}_{stat}": float(fn([p[m] for p in per_seed]))
               for m in ("intersection_over_k", "set_jaccard") for stat, fn in (("mean", np.mean), ("min", np.min), ("max", np.max))}})
    pair_values = [{m: aggregate_frame_values(test, v)["macro_source_mean"]
                    for m, v in overlap_values(actions[a], actions[c]).items()} for a, c in within_pairs(rl_names)]
    overlap_rows.append({"left": "rl_seed", "right": "rl_other_seed", "num_pairs": len(pair_values),
        **{f"{m}_{stat}": float(fn([p[m] for p in pair_values]))
           for m in ("intersection_over_k", "set_jaccard") for stat, fn in (("mean", np.mean), ("min", np.min), ("max", np.max))}})
    for ref_name in ("train_pooled_prior16", "center16"):
        v = overlap_values(prior_actions, references[ref_name])
        overlap_rows.append({"left": "train_source_prior16", "right": ref_name, "num_pairs": 1,
            **{f"{m}_{stat}": aggregate_frame_values(test, v[m])["macro_source_mean"]
               for m in ("intersection_over_k", "set_jaccard") for stat in ("mean", "min", "max")}})

    coverage_rows = []
    for name, r in [*((k, v) for k, v in zip(rl_names, rl)), *controls.items()]:
        coverage_rows.append({"split": "test", "method": name, "macro_source_mean": r["macro_source_mean"],
            "video_pooled_mean": video_pooled(r), "frame_pooled_mean": r["pooled_frame_mean"],
            **{f"source_{s}": r["per_source"][s]["mean_video_value"] for s in STAVIS_SOURCES}})
    for name, reps in seed_controls.items():
        for seed_name, r in zip(rl_names, reps):
            coverage_rows.append({"split": "test", "method": f"{name}:{seed_name}", "macro_source_mean": r["macro_source_mean"],
                "video_pooled_mean": video_pooled(r), "frame_pooled_mean": r["pooled_frame_mean"],
                **{f"source_{s}": r["per_source"][s]["mean_video_value"] for s in STAVIS_SOURCES}})
    for name, r in [*((f"rl{s}", v) for s, v in zip(cfg["gt1"]["seeds"], val_rl)),
                    ("train_source_prior16", val_prior), ("train_pooled_prior16", val_pooled)]:
        coverage_rows.append({"split": "val", "method": name, "macro_source_mean": r["macro_source_mean"],
            "video_pooled_mean": video_pooled(r), "frame_pooled_mean": r.get("pooled_frame_mean", "unavailable_from_saved_val_diagnostics"),
            **{f"source_{s}": r["per_source"][s]["mean_video_value"] for s in STAVIS_SOURCES}})

    # 7. Test-video count reconciliation.
    summary = json.loads(track(ROOT / cfg["data"]["d0_summary"]).read_text())
    recon = []
    for source in STAVIS_SOURCES:
        for video, info in sorted(summary["sources"][source]["videos"].items()):
            if info["split"] == "test":
                recon.append({"source": source, "video_id": video, "target_grid_frames": info["target_grid_frames"],
                              "valid_clips": info["valid_clips"], "excluded_clips": info["excluded_clips"],
                              "in_gt1": (source, video) in {(r["source"], r["video_id"]) for r in test}})
    names = [r["video_id"] for r in recon if r["in_gt1"]]
    reconciliation = {"d0_listed_test_videos": len(recon), "gt1_source_video_pairs": len(names),
        "unique_video_names_in_gt1": len(set(names)),
        "names_shared_across_sources": sorted({v for v in names if names.count(v) > 1}),
        "listed_without_valid_clips": [f"{r['source']}/{r['video_id']} ({r['target_grid_frames']} target frames < 16)"
                                       for r in recon if not r["in_gt1"]]}
    require(all(r["valid_clips"] > 0 for r in recon if r["in_gt1"]), "inconsistent reconciliation")

    return dict(source_prior=source_prior, pooled_prior=pooled_prior, center=center,
                test_static_cells=test_static_cells, val_static_cells=val_static_cells,
                reproduction=repro, contrasts=contrasts, overlap=overlap_rows, coverage=coverage_rows,
                reconciliation=reconciliation, recon_rows=recon, shuffle_pairs=shuffle_pairs(test, shuffle_idx))


def shuffle_pairs(records, idx):
    pairs = {}
    for r, j in zip(records, idx):
        pairs.setdefault(f"{r['source']}/{r['video_id']}", f"{records[j]['source']}/{records[j]['video_id']}")
    return pairs


def hlvid_analysis(cfg, out):
    bundle = ROOT / cfg["hlvid"]["curation_bundle"]
    manifest = json.loads(track(bundle / "manifest.json").read_text())
    shutil.copyfile(track(bundle / "contrast_summary.csv"), out / "hlvid_contrast_summary_reused.csv")
    arms = {}
    for entry in manifest["inputs"]:
        name = f"trained_{entry['base_seed']}" if entry["method"] == "trained_k16" else entry["method"]
        path = track(Path(entry["raw_directory"]) / "results.jsonl")
        require(INPUTS[str(path)] == entry["results_sha256"], f"{name} results hash mismatch")
        rows = sorted((json.loads(l) for l in path.open() if l.strip()), key=lambda r: int(r["question_id"]))
        arms[name] = rows
    trained = [f"trained_{s}" for s in cfg["hlvid"]["trained_seeds"]]
    require(set(arms) == set(trained) | set(cfg["hlvid"]["controls"]), "unexpected HLVid arms")
    ref = arms[trained[0]]
    for rows in arms.values():
        require([(r["question_id"], r["video_path"], r["answer"]) for r in rows] ==
                [(r["question_id"], r["video_path"], r["answer"]) for r in ref], "HLVid question alignment")
    videos = [r["video_path"] for r in ref]
    letters = {k: [r["prediction_letter"] for r in v] for k, v in arms.items()}
    correct = {k: [bool(r["is_correct"]) for r in v] for k, v in arms.items()}
    groups = {"within_trained_seed_pairs": within_pairs(trained),
              "trained_vs_stavis_center16": [(t, "stavis_center16") for t in trained],
              "trained_vs_pretrained_exact_k16": [(t, "pretrained_exact_k16") for t in trained],
              "pretrained_vs_stavis_center16": [("pretrained_exact_k16", "stavis_center16")]}
    b = cfg["bootstrap"]
    rows, per_question = [], {}
    for field, data in (("answer_letter", letters), ("correctness", correct)):
        for group, pairs in groups.items():
            d = pairwise_disagreement(data, pairs)
            per_question[(field, group)] = d.mean(axis=0)
            rows.append({"field": field, "group": group, "num_pairs": len(pairs), "num_questions": d.shape[1],
                         "mean_disagreement_rate": float(d.mean()), "min_pair_rate": float(d.mean(axis=1).min()),
                         "max_pair_rate": float(d.mean(axis=1).max())})
    excess = []
    for field in ("answer_letter", "correctness"):
        for group in ("trained_vs_stavis_center16", "trained_vs_pretrained_exact_k16"):
            r = video_cluster_bootstrap_mean_difference(per_question[(field, group)],
                per_question[(field, "within_trained_seed_pairs")], videos, iterations=b["iterations"], seed=b["seed"])
            excess.append({"field": field, "across_group": group, "minus": "within_trained_seed_pairs",
                           "difference": r["difference"], "ci90_low": r["ci90_low"], "ci90_high": r["ci90_high"],
                           "num_video_clusters": r["num_clusters"]})
    unanimous = float(np.mean([len({letters[t][i] for t in trained}) == 1 for i in range(len(ref))]))
    with (bundle / "contrast_summary.csv").open() as handle:
        reused = list(csv.DictReader(handle))
    return dict(disagreement=rows, excess=excess, trained_unanimous_letter_fraction=unanimous,
                reused_contrasts=reused, num_questions=len(ref), num_videos=len(set(videos)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/human_gaze/configs/gaze_posthoc_gt1_controls_20261105.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readme", type=Path, help="hand-written interpretation copied into the bundle")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = OmegaConf.to_container(OmegaConf.load(track(args.config)), resolve=True)
    gaze = gaze_analysis(cfg)
    args.output.mkdir(parents=True)
    hl = hlvid_analysis(cfg, args.output)
    write_csv(args.output / "coverage_by_method.csv", gaze["coverage"])
    write_csv(args.output / "coverage_contrasts.csv", gaze["contrasts"])
    write_csv(args.output / "action_overlap_test.csv", gaze["overlap"])
    write_csv(args.output / "test_video_reconciliation.csv", gaze["recon_rows"])
    write_csv(args.output / "hlvid_answer_disagreement.csv", hl["disagreement"])
    write_csv(args.output / "hlvid_disagreement_excess.csv", hl["excess"])
    run_id = args.output.name
    metrics = {"schema_version": 1, "run_id": run_id, "evidential_status": "post_hoc",
               "test_exposure": cfg["test_exposure"],
               "gt1_reproduction": gaze["reproduction"],
               "prior_cells": {"train_source_prior16": gaze["source_prior"], "train_pooled_prior16": gaze["pooled_prior"],
                               "center16": gaze["center"],
                               "pooled_shared_with_source_prior": {s: len(set(c) & set(gaze["pooled_prior"]))
                                                                   for s, c in gaze["source_prior"].items()}},
               "static_top16_cells": {"test_derived": gaze["test_static_cells"], "validation_derived": gaze["val_static_cells"]},
               "same_source_shuffle_video_pairs": gaze["shuffle_pairs"],
               "coverage_contrasts": gaze["contrasts"], "action_overlap_test": gaze["overlap"],
               "test_video_reconciliation": gaze["reconciliation"],
               "hlvid": {k: v for k, v in hl.items()}}
    with (args.output / "metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    OmegaConf.save(OmegaConf.create(cfg), args.output / "config.yaml")
    if args.readme is not None:
        shutil.copyfile(args.readme, args.output / "README.md")
    script = Path(__file__).resolve()
    module = ROOT / "autogaze/human_gaze/posthoc_analysis.py"
    with (args.output / "manifest.json").open("x") as handle:
        json.dump({"schema_version": 1, "run_id": run_id, "evidential_status": "post_hoc",
                   "compute": cfg["compute"], "logical_nvila_rollouts": 0, "training_updates_run": 0,
                   "script": {"path": str(script.relative_to(ROOT)), "sha256": sha256_file(script)},
                   "module": {"path": str(module.relative_to(ROOT)), "sha256": sha256_file(module)},
                   "inputs_sha256": dict(sorted(INPUTS.items())),
                   "heavy_artifacts": {"gt1_test_actions": cfg["gt1"]["actions_root"],
                                       "hlvid_raw_results": "see inputs_sha256 results.jsonl paths"},
                   "outputs_sha256": {p.name: sha256_file(p) for p in sorted(args.output.iterdir()) if p.name != "manifest.json"}},
                  handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
