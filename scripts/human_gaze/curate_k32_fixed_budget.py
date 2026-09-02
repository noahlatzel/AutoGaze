# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curate metrics and figures for the exact-K32 reproduction of R2c K36."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.human_gaze.verify_k32_fixed_budget import SOURCES, read_json, read_jsonl, run_names


def load_yaml(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def summarize(values: list[float], seeds: list[int]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "seeds": seeds,
        "values": values,
        "mean": float(array.mean()),
        "population_sd": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def endpoint_from_run(run_dir: Path, metric: str, train_step: int = 17685) -> float:
    records = read_jsonl(run_dir / "validation_metrics.jsonl")
    matches = [row for row in records if int(row["train_step"]) == train_step]
    if len(matches) != 1:
        raise AssertionError(f"Expected one endpoint at {train_step}: {run_dir}")
    return float(matches[0][metric])


def stitched_history(cfg: dict, run_root: Path, base_seed: int) -> list[dict]:
    records = []
    for stage_index, name in enumerate(run_names(base_seed)):
        for row in read_jsonl(run_root / name / "validation_metrics.jsonl"):
            step = int(row["train_step"])
            if stage_index == 0:
                cumulative = step
            elif stage_index == 1:
                if step == 0:
                    continue
                cumulative = int(cfg["stage1_updates"]) + step
            else:
                if step <= int(cfg["stage2_updates"]):
                    continue
                cumulative = int(cfg["stage1_updates"]) + step
            records.append({**row, "cumulative_step": cumulative})
    return sorted(records, key=lambda row: int(row["cumulative_step"]))


def reference_endpoints(cfg: dict, reference_root: Path) -> dict[int, dict]:
    result = {}
    for budget in (16, 24, 36):
        ref = cfg["reference_budgets"][f"k{budget}"]
        seeds = [int(value) for value in ref["seeds"]]
        values = []
        for seed in seeds:
            run = ref["run_template"].format(
                base_seed=seed, continuation_seed=seed + 100000
            )
            values.append(
                endpoint_from_run(reference_root / "grpo" / run, ref["metric"])
            )
        result[budget] = summarize(values, seeds)
    return result


def pretrained_macro(budget: int, cfg: dict, reference_root: Path) -> float:
    if budget == 16:
        report = read_json(Path("experiments/human_gaze/results/m0_pretrained_exact16_fold1/metrics.json"))
    elif budget == 32:
        report = read_json(Path(cfg["pretrained"]))
    else:
        report = read_json(reference_root / "baselines" / f"pretrained_fixedk{budget}_val.json")
    return float(report["metrics"]["val"][str(budget)]["macro_source_mean"])


def load_diagnostics(cfg: dict, root: Path) -> dict[int, dict[int, dict]]:
    result = {}
    for seed in [int(value) for value in cfg["base_seeds"]]:
        names = run_names(seed)
        result[seed] = {}
        for step in [int(value) for value in cfg["diagnostics"]["checkpoints_cumulative"]]:
            name = names[1] if step == 10000 else names[2]
            result[seed][step] = read_json(
                root / name / f"center_collapse_val_step{step}.json"
            )
    return result


def control_metrics(diagnostics: dict[int, dict[int, dict]]) -> dict:
    methods = ("actual", "center", "selection_frequency_top32", "shuffled")
    by_step = {}
    for step in (10000, 15000, 20000):
        by_step[str(step)] = {}
        for method in methods:
            values = [
                float(by_step_report[step]["coverage"][method]["32"]["macro_source_mean"])
                for by_step_report in diagnostics.values()
            ]
            by_step[str(step)][method] = summarize(values, sorted(diagnostics))
        for metric in (
            "mean_center_overlap_fraction",
            "normalized_selection_entropy",
            "mean_actual_shuffled_overlap_fraction",
        ):
            values = [
                float(by_step_report[step]["diagnostics"][metric])
                for by_step_report in diagnostics.values()
            ]
            by_step[str(step)][metric] = summarize(values, sorted(diagnostics))
    return by_step


def plot_learning(histories: dict[int, list[dict]], baselines: dict, pretrained: float, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    metric = "coverage_k32_macro_source"
    for seed, history in histories.items():
        axis.plot([row["cumulative_step"] for row in history], [row[metric] for row in history], alpha=0.35, label=f"Seed {seed}")
    common_steps = sorted(set.intersection(*[{int(row["cumulative_step"]) for row in history} for history in histories.values()]))
    matrix = np.asarray([[next(float(row[metric]) for row in history if int(row["cumulative_step"]) == step) for step in common_steps] for history in histories.values()])
    axis.plot(common_steps, matrix.mean(axis=0), color="black", linewidth=2.4, label="Three-seed mean")
    axis.fill_between(common_steps, matrix.mean(axis=0) - matrix.std(axis=0), matrix.mean(axis=0) + matrix.std(axis=0), color="black", alpha=0.12)
    val = baselines["splits"]["val"]
    for value, label, style in (
        (pretrained, "Pretrained", "--"),
        (val["center"]["32"]["macro_source_mean"], "Center-32", ":"),
        (val["prior"]["32"]["macro_source_mean"], "Prior-32", "-."),
    ):
        axis.axhline(value, color="#555555", linestyle=style, linewidth=1.1, label=label)
    axis.axvline(2315, color="#999999", linestyle="--", linewidth=0.8)
    axis.axvline(10000, color="#999999", linestyle="--", linewidth=0.8)
    axis.set(xlabel="Cumulative optimizer updates", ylabel="Validation macro-source coverage", title="Exact-K32 decoder-only direct-coverage GRPO", xlim=(0, 20000))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_sources(histories: dict[int, list[dict]], baselines: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.7), sharex=True, constrained_layout=True)
    val = baselines["splits"]["val"]
    for axis, source in zip(axes.flat, SOURCES):
        metric = f"coverage_k32_source_{source}"
        for history in histories.values():
            axis.plot([row["cumulative_step"] for row in history], [row[metric] for row in history], alpha=0.25)
        common_steps = sorted(set.intersection(*[{int(row["cumulative_step"]) for row in history} for history in histories.values()]))
        matrix = np.asarray([[next(float(row[metric]) for row in history if int(row["cumulative_step"]) == step) for step in common_steps] for history in histories.values()])
        axis.plot(common_steps, matrix.mean(axis=0), color="black", linewidth=2.0)
        axis.fill_between(common_steps, matrix.mean(axis=0) - matrix.std(axis=0), matrix.mean(axis=0) + matrix.std(axis=0), color="black", alpha=0.12)
        axis.axhline(val["prior"]["32"]["per_source"][source]["mean_video_coverage"], color="#d95f02", linestyle="-.", label="Prior-32")
        axis.axhline(val["center"]["32"]["per_source"][source]["mean_video_coverage"], color="#666666", linestyle=":", label="Center-32")
        axis.set_title(source)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)
    for axis in axes[:, 0]: axis.set_ylabel("Mean-video coverage")
    for axis in axes[-1, :]: axis.set_xlabel("Cumulative updates")
    figure.suptitle("Exact-K32 validation trajectories by source")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_controls(controls: dict, output: Path) -> None:
    steps = np.asarray([10000, 15000, 20000])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    for method, label, color, style in (
        ("actual", "Dynamic policy", "#1677b8", "-"),
        ("center", "Center-32", "#666666", ":"),
        ("selection_frequency_top32", "Static learned Top-32", "#d95f02", "-."),
        ("shuffled", "Same-source shuffled video", "#7b3294", "--"),
    ):
        means = np.asarray([controls[str(step)][method]["mean"] for step in steps])
        stds = np.asarray([controls[str(step)][method]["population_sd"] for step in steps])
        axes[0].plot(steps, means, marker="o", color=color, linestyle=style, label=label)
        axes[0].fill_between(steps, means - stds, means + stds, color=color, alpha=0.1)
    for metric, label, color, style in (
        ("mean_center_overlap_fraction", "Center overlap", "#d95f02", "-"),
        ("normalized_selection_entropy", "Normalized entropy", "#1b9e77", "--"),
        ("mean_actual_shuffled_overlap_fraction", "Actual/shuffled overlap", "#7b3294", ":"),
    ):
        means = [controls[str(step)][metric]["mean"] for step in steps]
        axes[1].plot(steps, means, marker="o", color=color, linestyle=style, label=label)
    axes[0].set(title="Dynamic and static controls", ylabel="Validation macro-source coverage")
    axes[1].set(title="Concentration and content dependence", ylabel="Fraction / normalized entropy")
    for axis in axes:
        axis.set_xlabel("Cumulative optimizer updates")
        axis.set_xticks(steps)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_scaling(endpoints: dict[int, dict], baselines: dict, pretrained: dict[int, float], output: Path) -> None:
    budgets = np.asarray([16, 24, 32, 36])
    val = baselines["splits"]["val"]
    figure, axis = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    axis.errorbar(budgets, [endpoints[k]["mean"] for k in budgets], yerr=[endpoints[k]["population_sd"] for k in budgets], marker="o", linewidth=2.2, capsize=4, label="Learned at 20k")
    for method, label, style in (("prior", "Source Prior-K", "-."), ("center", "Center-K", ":"), ("oracle", "Per-frame Oracle-K", "--")):
        axis.plot(budgets, [val[method][str(k)]["macro_source_mean"] for k in budgets], marker="o", linestyle=style, label=label)
    axis.plot(budgets, [pretrained[k] for k in budgets], marker="o", linestyle="--", label="Pretrained AutoGaze")
    axis.set(xlabel="Fixed fine tokens per frame", ylabel="Validation macro-source coverage", title="Matched fixed-budget coverage scaling")
    axis.set_xticks(budgets)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--training-job-id", type=int, required=True)
    parser.add_argument("--diagnostics-job-id", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    verification = read_json(args.verification)
    assert verification["status"] == "verified" and verification["test_split_opened"] is False
    baselines = read_json(Path(cfg["static_baselines"]))
    histories = {seed: stitched_history(cfg, args.run_root, seed) for seed in [int(v) for v in cfg["base_seeds"]]}
    endpoints = reference_endpoints(cfg, args.reference_root)
    k32_values = [endpoint_from_run(args.run_root / run_names(seed)[2], "coverage_k32_macro_source") for seed in sorted(histories)]
    endpoints[32] = summarize(k32_values, sorted(histories))
    diagnostics = load_diagnostics(cfg, args.diagnostics_root)
    controls = control_metrics(diagnostics)
    val = baselines["splits"]["val"]
    prior32 = float(val["prior"]["32"]["macro_source_mean"])
    coverage_pass = endpoints[32]["mean"] > prior32
    endpoint_reports = [diagnostics[seed][20000] for seed in sorted(diagnostics)]
    content_pass = all(
        float(report["coverage"]["actual"]["32"]["macro_source_mean"])
        > float(report["coverage"][method]["32"]["macro_source_mean"])
        for report in endpoint_reports
        for method in ("selection_frequency_top32", "shuffled")
    )
    per_source = {}
    for source in SOURCES:
        values = [float(verification["endpoints"][str(seed)][f"coverage_k32_source_{source}"]) for seed in sorted(histories)]
        per_source[source] = summarize(values, sorted(histories))
    pretrained = {budget: pretrained_macro(budget, cfg, args.reference_root) for budget in (16, 24, 32, 36)}

    metrics = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "primary_metric": "validation_macro_source_coverage_at_fixed_20000_update_endpoint",
        "fixed_budget_endpoints": {str(key): value for key, value in sorted(endpoints.items())},
        "k32": {
            "source_prior": prior32,
            "learned_minus_prior": endpoints[32]["mean"] - prior32,
            "coverage_decision": "pass" if coverage_pass else "fail",
            "content_dependence_decision": "pass" if content_pass else "fail",
            "per_source": per_source,
            "longitudinal_controls": controls,
        },
        "baselines": {str(k): {method: float(val[method][str(k)]["macro_source_mean"]) for method in ("random", "center", "prior", "oracle")} for k in (16, 24, 32, 36)},
        "pretrained": {str(k): value for k, value in pretrained.items()},
        "test_split_opened": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", metrics)
    shutil.copyfile(args.config, args.output_dir / "config.yaml")
    shutil.copyfile(args.verification, args.output_dir / "verification.json")

    plot_learning(histories, baselines, pretrained[32], args.figure_dir / "r2c_k32_fixed_budget_learning.png")
    plot_sources(histories, baselines, args.figure_dir / "r2c_k32_fixed_budget_by_source.png")
    plot_controls(controls, args.figure_dir / "r2c_k32_fixed_budget_endpoint_controls.png")
    plot_scaling(endpoints, baselines, pretrained, args.figure_dir / "r2c_k16_k24_k32_k36_coverage_scaling.png")

    d0_root = Path("outputs/human_gaze/d0_stavis_fold1_validated")
    elapsed = {}
    for seed in sorted(histories):
        elapsed[str(seed)] = sum(float(read_jsonl(args.run_root / name / "training_metrics.jsonl")[-1]["elapsed_seconds"]) for name in run_names(seed))
    manifest = {
        "schema_version": 1,
        "run_id": f"20260901-r2c_k32_fixed_budget_reproduction_{args.code_commit[:7]}",
        "training_code_commit": args.code_commit,
        "k36_execution_code_commit": "2e726b22674d372c7df16517970aaec6b4abee18",
        "training_job_id": args.training_job_id,
        "diagnostics_job_id": args.diagnostics_job_id,
        "training_gpu_seconds_by_seed": elapsed,
        "successful_training_gpu_hours": sum(elapsed.values()) / 3600.0,
        "dataset_manifest_sha256": sha256(d0_root / "clips.jsonl"),
        "cell_mass_sha256": sha256(d0_root / "cell_mass.npy"),
        "baseline_sha256": sha256(Path(cfg["static_baselines"])),
        "pretrained_k32_sha256": sha256(Path(cfg["pretrained"])),
        "heavy_artifacts": {
            "execution_root": str(Path.cwd()),
            "training_template": "outputs/human_gaze/grpo/r2c_k32_reproduction_stage{1,2,3}_*",
            "diagnostics_template": "outputs/human_gaze/diagnostics/r2c_k32_reproduction_*/center_collapse_val_step*.json",
            "k36_reference_root": str(args.reference_root),
        },
        "test_split_opened": False,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    readme = f"""# Exact-K32 fixed-budget reproduction\n\nThis bundle reproduces the completed R2c exact-K36 protocol at exact K32. The only scientific treatment change is the fixed fine-cell budget (36 to 32); run-identity paths and reporting prefixes change accordingly. All three matched seeds use the fixed 20,000-update endpoint on the unchanged train-derived validation population. The protected test split was not opened.\n\nThe learned K32 endpoint is `{endpoints[32]['mean']:.6f} ± {endpoints[32]['population_sd']:.6f}` (population SD) versus Prior-32 `{prior32:.6f}`, a delta of `{endpoints[32]['mean'] - prior32:+.6f}`. The frozen coverage rule is **{'pass' if coverage_pass else 'fail'}**. The endpoint dynamic/static/shuffled content-dependence rule is **{'pass' if content_pass else 'fail'}**.\n\n`metrics.json` contains per-seed endpoints, K16/K24/K32/K36 comparisons, per-source K32 estimates, and the complete longitudinal control suite. `verification.json` proves exact 32-per-frame / 512-per-clip accounting, all 20,000 updates, fixed validation schedules, matched seeds, and resolved-config parity with K36 except the budget and run identity. `manifest.json` records commits, jobs, hashes, compute, and heavy-artifact paths.\n"""
    (args.output_dir / "README.md").write_text(readme)
    print(json.dumps({"coverage_pass": coverage_pass, "content_dependence_pass": content_pass, "k32_mean": endpoints[32]["mean"], "prior32": prior32}, sort_keys=True))


if __name__ == "__main__":
    main()
