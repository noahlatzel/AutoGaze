# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create presentation figures and tables for HLVid frame selectors."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIR = "outputs/hlvid/external_frame_selector_comparison_128"
EXPECTED_EXAMPLES = 268
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260717

METHODS = [
    ("uniform", "Uniform"),
    ("aks", "AKS-style"),
    ("bolt", "BOLT"),
    ("sealing_mdf", "Sealing MDF"),
    ("sealing_mif", "Sealing MIF"),
    ("infoshot_reimpl", "InfoShot-reimpl"),
    ("hier_pixel_diversity", "Hier. Pixel Diversity"),
    ("hier_dino_semdiv", "Hier. DINOv2 SemDiv"),
    ("hier_siglip_semdiv", "Hier. SigLIP SemDiv"),
    ("hier_autogaze_density", "Hier. AutoGaze Density"),
]
METHOD_COLORS = {
    "uniform": "#666666",
    "aks": "#4C78A8",
    "bolt": "#2A9D8F",
    "sealing_mdf": "#59A14F",
    "sealing_mif": "#E9C46A",
    "infoshot_reimpl": "#E76F51",
    "hier_pixel_diversity": "#8E6C8A",
    "hier_dino_semdiv": "#6F4E7C",
    "hier_siglip_semdiv": "#B279A2",
    "hier_autogaze_density": "#9C755F",
}
SETTING_COLORS = {"mtv1": "#3B6FB6", "mtv48": "#D05A47"}
QUERY_CONDITIONING = {
    "uniform": "agnostic",
    "aks": "aware",
    "bolt": "aware",
    "sealing_mdf": "agnostic",
    "sealing_mif": "aware",
    "infoshot_reimpl": "agnostic",
    "hier_pixel_diversity": "agnostic",
    "hier_dino_semdiv": "agnostic",
    "hier_siglip_semdiv": "agnostic",
    "hier_autogaze_density": "agnostic",
}
METHOD_FAMILY = {
    method: (
        "baseline"
        if method == "uniform"
        else "hierarchical"
        if method.startswith("hier_")
        else "external"
    )
    for method, _ in METHODS
}
EXTERNAL_DIRS = {
    "aks": "aks",
    "bolt": "bolt",
    "sealing_mdf": "sealing_mdf",
    "sealing_mif": "sealing_mif",
    "infoshot_reimpl": "infoshot_reimpl",
}
HIERARCHICAL_DIRS = {
    "hier_pixel_diversity": "pixel_diversity",
    "hier_dino_semdiv": "dino_semdiv",
    "hier_siglip_semdiv": "siglip_semdiv",
    "hier_autogaze_density": "autogaze_gaze_density",
}


@dataclass
class Run:
    setting: str
    method: str
    label: str
    summary_path: Path
    results_path: Path
    summary: dict
    results: dict[str, dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtv1-root", type=Path, default=Path("outputs/hlvid/external_frame_selectors_128_mtv1"))
    parser.add_argument("--mtv48-root", type=Path, default=Path("outputs/hlvid/external_frame_selectors_128_mtv48"))
    parser.add_argument(
        "--hierarchical-mtv1-root",
        type=Path,
        default=Path("outputs/hlvid/hierarchical_temporal_density_128_mtv1"),
    )
    parser.add_argument(
        "--hierarchical-mtv48-root",
        type=Path,
        default=Path("outputs/hlvid/hierarchical_temporal_density_128_mtv48"),
    )
    parser.add_argument("--uniform-mtv1-root", type=Path, default=Path("outputs/hlvid/uniform_128_mtv1"))
    parser.add_argument("--uniform-mtv48-root", type=Path, default=Path("outputs/hlvid/uniform_128_mtv48"))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in rows:
                raise ValueError(f"Duplicate question_id in {path}: {question_id}")
            rows[question_id] = row
    return rows


def load_run(setting: str, method: str, label: str, root: Path) -> Run:
    summary_path = root / "summary.json"
    results_path = root / "results.jsonl"
    if not summary_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Missing completed run artifacts under {root}")
    summary = json.loads(summary_path.read_text())
    if int(summary["num_examples"]) != EXPECTED_EXAMPLES:
        raise ValueError(f"Expected {EXPECTED_EXAMPLES} examples in {summary_path}, got {summary['num_examples']}")
    results = read_jsonl(results_path)
    if len(results) != EXPECTED_EXAMPLES:
        raise ValueError(f"Expected {EXPECTED_EXAMPLES} results in {results_path}, got {len(results)}")
    return Run(setting, method, label, summary_path, results_path, summary, results)


def load_runs(args: argparse.Namespace) -> dict[str, dict[str, Run]]:
    roots = {
        "mtv1": {
            "uniform": args.uniform_mtv1_root,
            **{method: args.mtv1_root / directory for method, directory in EXTERNAL_DIRS.items()},
            **{
                method: args.hierarchical_mtv1_root / directory
                for method, directory in HIERARCHICAL_DIRS.items()
            },
        },
        "mtv48": {
            "uniform": args.uniform_mtv48_root,
            **{method: args.mtv48_root / directory for method, directory in EXTERNAL_DIRS.items()},
            **{
                method: args.hierarchical_mtv48_root / directory
                for method, directory in HIERARCHICAL_DIRS.items()
            },
        },
    }
    labels = dict(METHODS)
    runs = {
        setting: {
            method: load_run(setting, method, labels[method], root)
            for method, root in setting_roots.items()
        }
        for setting, setting_roots in roots.items()
    }
    validate_alignment(runs)
    return runs


def validate_alignment(runs: dict[str, dict[str, Run]]) -> None:
    for setting, setting_runs in runs.items():
        baseline = setting_runs["uniform"]
        baseline_ids = set(baseline.results)
        baseline_categories = {question_id: row["category"] for question_id, row in baseline.results.items()}
        for run in setting_runs.values():
            if set(run.results) != baseline_ids:
                raise ValueError(f"Question IDs are not aligned for {setting}/{run.method}")
            categories = {question_id: row["category"] for question_id, row in run.results.items()}
            if categories != baseline_categories:
                raise ValueError(f"Categories are not aligned for {setting}/{run.method}")
            if float(run.summary["mean_visited_frames"]) != 128.0:
                raise ValueError(f"Expected 128 visited frames for {setting}/{run.method}")


def video_cluster_arrays(run: Run) -> tuple[np.ndarray, np.ndarray, list[str]]:
    by_video: dict[str, list[bool]] = {}
    for row in run.results.values():
        by_video.setdefault(str(row["video_path"]), []).append(bool(row["is_correct"]))
    videos = sorted(by_video)
    correct = np.asarray([sum(by_video[video]) for video in videos], dtype=np.float64)
    total = np.asarray([len(by_video[video]) for video in videos], dtype=np.float64)
    return correct, total, videos


def cluster_bootstrap_accuracy(run: Run, samples: int, seed: int) -> tuple[float, float]:
    correct, total, _ = video_cluster_arrays(run)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(correct), size=(samples, len(correct)))
    estimates = correct[draws].sum(axis=1) / total[draws].sum(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def cluster_bootstrap_delta(
    run: Run,
    baseline: Run,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    videos = sorted({str(row["video_path"]) for row in baseline.results.values()})
    run_by_video = {video: [0, 0] for video in videos}
    base_by_video = {video: [0, 0] for video in videos}
    for question_id, base_row in baseline.results.items():
        video = str(base_row["video_path"])
        base_by_video[video][0] += int(bool(base_row["is_correct"]))
        base_by_video[video][1] += 1
        run_by_video[video][0] += int(bool(run.results[question_id]["is_correct"]))
        run_by_video[video][1] += 1

    run_correct = np.asarray([run_by_video[video][0] for video in videos], dtype=np.float64)
    base_correct = np.asarray([base_by_video[video][0] for video in videos], dtype=np.float64)
    totals = np.asarray([base_by_video[video][1] for video in videos], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(videos), size=(samples, len(videos)))
    estimates = (run_correct[draws].sum(axis=1) - base_correct[draws].sum(axis=1)) / totals[draws].sum(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def paired_flips(run: Run, baseline: Run) -> tuple[int, int]:
    gains = 0
    losses = 0
    for question_id, base_row in baseline.results.items():
        base_correct = bool(base_row["is_correct"])
        run_correct = bool(run.results[question_id]["is_correct"])
        gains += int(run_correct and not base_correct)
        losses += int(base_correct and not run_correct)
    return gains, losses


def build_table_rows(
    runs: dict[str, dict[str, Run]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    rows = []
    for setting_index, setting in enumerate(("mtv1", "mtv48")):
        baseline = runs[setting]["uniform"]
        baseline_accuracy = float(baseline.summary["accuracy"])
        for method_index, (method, label) in enumerate(METHODS):
            run = runs[setting][method]
            accuracy = float(run.summary["accuracy"])
            ci_low, ci_high = cluster_bootstrap_accuracy(
                run,
                bootstrap_samples,
                seed + setting_index * 100 + method_index,
            )
            if method == "uniform":
                delta_low = delta_high = 0.0
                gains = losses = 0
            else:
                delta_low, delta_high = cluster_bootstrap_delta(
                    run,
                    baseline,
                    bootstrap_samples,
                    seed + 1000 + setting_index * 100 + method_index,
                )
                gains, losses = paired_flips(run, baseline)
            rows.append(
                {
                    "setting": setting,
                    "max_tiles_video": 1 if setting == "mtv1" else 48,
                    "method": method,
                    "method_label": label,
                    "method_family": METHOD_FAMILY[method],
                    "query_conditioning": QUERY_CONDITIONING[method],
                    "accuracy": accuracy,
                    "accuracy_ci_low": ci_low,
                    "accuracy_ci_high": ci_high,
                    "delta_vs_uniform": accuracy - baseline_accuracy,
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "av_accuracy": float(run.summary["by_category"]["av"]["accuracy"]),
                    "household_accuracy": float(run.summary["by_category"]["household"]["accuracy"]),
                    "gains_vs_uniform": gains,
                    "losses_vs_uniform": losses,
                    "net_flips_vs_uniform": gains - losses,
                    "num_examples": int(run.summary["num_examples"]),
                    "mean_visited_frames": float(run.summary["mean_visited_frames"]),
                    "mean_visited_tiles": float(run.summary["mean_visited_tiles"]),
                    "summary_path": str(run.summary_path),
                }
            )
    return rows


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def plot_overall_accuracy(rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["setting"], row["method"]): row for row in rows}
    methods = [method for method, _ in METHODS]
    labels = [label for _, label in METHODS]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(11.2, 8.2))

    mtv1 = np.asarray([lookup[("mtv1", method)]["accuracy"] * 100 for method in methods])
    mtv48 = np.asarray([lookup[("mtv48", method)]["accuracy"] * 100 for method in methods])
    ax.hlines(y, np.minimum(mtv1, mtv48), np.maximum(mtv1, mtv48), color="#B7B7B7", linewidth=2.2, zorder=1)
    ax.scatter(mtv1, y, s=95, color=SETTING_COLORS["mtv1"], label="max_tiles_video = 1", zorder=3)
    ax.scatter(mtv48, y, s=95, marker="s", color=SETTING_COLORS["mtv48"], label="max_tiles_video = 48", zorder=3)

    for index, value in enumerate(mtv1):
        ax.annotate(f"{value:.1f}", (value, index), xytext=(-7, -13), textcoords="offset points", ha="right", color=SETTING_COLORS["mtv1"], fontsize=9)
    for index, value in enumerate(mtv48):
        ax.annotate(f"{value:.1f}", (value, index), xytext=(7, 5), textcoords="offset points", ha="left", color=SETTING_COLORS["mtv48"], fontsize=9)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("HLVid accuracy (%)")
    ax.axhline(5.5, color="#D0D0D0", linewidth=1.0)
    ax.text(
        0.995,
        0.39,
        "Hierarchical temporal density",
        transform=ax.transAxes,
        ha="right",
        color="#6F4E7C",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_title("Frame Selectors at a Fixed 128-Frame Budget", loc="left", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(min(mtv1.min(), mtv48.min()) - 2.5, max(mtv1.max(), mtv48.max()) + 3.0)
    save_figure(fig, output_dir, "overall_accuracy_mtv_comparison")


def plot_delta_vs_uniform(rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["setting"], row["method"]): row for row in rows}
    methods = [method for method, _ in METHODS if method != "uniform"]
    labels = [label for method, label in METHODS if method != "uniform"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.8), sharey=True)

    for ax, setting, title in zip(
        axes,
        ("mtv1", "mtv48"),
        ("max_tiles_video = 1", "max_tiles_video = 48"),
    ):
        values = np.asarray([lookup[(setting, method)]["delta_vs_uniform"] * 100 for method in methods])
        y = np.arange(len(methods))
        colors = [METHOD_COLORS[method] for method in methods]
        ax.barh(y, values, color=colors, height=0.64)
        ax.axvline(0.0, color="#222222", linewidth=1.2)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Accuracy change vs uniform (percentage points)")
        limit = max(6.0, float(np.abs(values).max()) + 1.0)
        ax.set_xlim(-limit, limit)
        for index, value in enumerate(values):
            ha = "left" if value >= 0 else "right"
            offset = 0.14 if value >= 0 else -0.14
            ax.text(value + offset, index, f"{value:+.1f}", va="center", ha=ha, fontsize=10, fontweight="bold")

    fig.suptitle("Sampling Methods Interact with the Spatial-Token Budget", x=0.03, ha="left", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "accuracy_delta_vs_uniform")


def plot_accuracy_heatmap(rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["setting"], row["method"]): row for row in rows}
    labels = [label for _, label in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.8), sharey=True)
    image = None
    for ax, setting, title in zip(
        axes,
        ("mtv1", "mtv48"),
        ("max_tiles_video = 1", "max_tiles_video = 48"),
    ):
        values = np.asarray(
            [
                [
                    lookup[(setting, method)]["accuracy"] * 100,
                    lookup[(setting, method)]["av_accuracy"] * 100,
                    lookup[(setting, method)]["household_accuracy"] * 100,
                ]
                for method, _ in METHODS
            ]
        )
        image = ax.imshow(values, cmap="RdYlGn", vmin=40, vmax=56, aspect="auto")
        ax.set_xticks(np.arange(3), ["Overall", "AV", "Household"])
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_title(title, fontweight="bold")
        ax.grid(False)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                color = "white" if value < 42 or value > 54 else "#202020"
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                )
    assert image is not None
    fig.suptitle("Accuracy by HLVid Category", x=0.03, ha="left", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "category_accuracy_heatmap")


def plot_paired_flips(rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["setting"], row["method"]): row for row in rows}
    methods = [method for method, _ in METHODS if method != "uniform"]
    labels = [label for method, label in METHODS if method != "uniform"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.8), sharey=True)

    for ax, setting, title in zip(
        axes,
        ("mtv1", "mtv48"),
        ("max_tiles_video = 1", "max_tiles_video = 48"),
    ):
        gains = np.asarray([lookup[(setting, method)]["gains_vs_uniform"] for method in methods])
        losses = np.asarray([lookup[(setting, method)]["losses_vs_uniform"] for method in methods])
        y = np.arange(len(methods))
        ax.barh(y, gains, color="#2A9D8F", height=0.62, label="Selector correct / uniform wrong")
        ax.barh(y, -losses, color="#E76F51", height=0.62, label="Uniform correct / selector wrong")
        ax.axvline(0.0, color="#222222", linewidth=1.2)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Questions flipped relative to uniform")
        limit = max(int(gains.max()), int(losses.max())) + 3
        ax.set_xlim(-limit, limit)
        for index, (gain, loss) in enumerate(zip(gains, losses)):
            ax.text(gain + 0.4, index, str(gain), va="center", ha="left", fontsize=9)
            ax.text(-loss - 0.4, index, str(loss), va="center", ha="right", fontsize=9)
        ax.text(0.03, 1.02, "Uniform only", transform=ax.transAxes, color="#E76F51", fontsize=9, fontweight="bold")
        ax.text(0.97, 1.02, "Selector only", transform=ax.transAxes, color="#2A9D8F", fontsize=9, fontweight="bold", ha="right")

    fig.suptitle("Where Each Selector Changes the Uniform Baseline", x=0.03, ha="left", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "paired_question_flips")


def format_accuracy(value: float, low: float, high: float) -> str:
    return f"{value * 100:.2f}% [{low * 100:.2f}, {high * 100:.2f}]"


def format_delta(value: float, low: float, high: float, baseline: bool) -> str:
    if baseline:
        return "reference"
    return f"{value * 100:+.2f} pp [{low * 100:+.2f}, {high * 100:+.2f}]"


def write_csv(rows: list[dict], output_dir: Path) -> None:
    path = output_dir / "results_tidy.csv"
    fieldnames = list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: list[dict], output_dir: Path) -> None:
    by_setting = {
        setting: [row for row in rows if row["setting"] == setting]
        for setting in ("mtv1", "mtv48")
    }
    lines = [
        "# Frame Selector Results",
        "",
        "Fixed budget: 128 visited frames, eight temporal tiles, 268 HLVid questions.",
        "Intervals are 95% video-cluster bootstrap intervals over 77 videos.",
        "",
    ]
    for setting, title in (("mtv1", "`max_tiles_video=1`"), ("mtv48", "`max_tiles_video=48`")):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Family | Method | Overall accuracy [95% CI] | Delta vs uniform [95% CI] | AV | Household | Gains / losses |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        setting_rows = by_setting[setting]
        best_accuracy = max(row["accuracy"] for row in setting_rows)
        for row in setting_rows:
            overall = format_accuracy(row["accuracy"], row["accuracy_ci_low"], row["accuracy_ci_high"])
            if row["accuracy"] == best_accuracy:
                overall = f"**{overall}**"
            delta = format_delta(
                row["delta_vs_uniform"],
                row["delta_ci_low"],
                row["delta_ci_high"],
                row["method"] == "uniform",
            )
            flips = "—" if row["method"] == "uniform" else f"{row['gains_vs_uniform']} / {row['losses_vs_uniform']}"
            lines.append(
                f"| {row['method_family'].title()} | {row['method_label']} | {overall} | {delta} | "
                f"{row['av_accuracy'] * 100:.2f}% | {row['household_accuracy'] * 100:.2f}% | {flips} |"
            )
        lines.append("")
    (output_dir / "results_table.md").write_text("\n".join(lines))


def write_latex_table(rows: list[dict], output_dir: Path) -> None:
    escaped_labels = {
        "Uniform": "Uniform",
        "AKS-style": "AKS-style",
        "BOLT": "BOLT",
        "Sealing MDF": "Sealing MDF",
        "Sealing MIF": "Sealing MIF",
        "InfoShot-reimpl": r"InfoShot-reimpl",
        "Hier. Pixel Diversity": "Hier. Pixel Diversity",
        "Hier. DINOv2 SemDiv": "Hier. DINOv2 SemDiv",
        "Hier. SigLIP SemDiv": "Hier. SigLIP SemDiv",
        "Hier. AutoGaze Density": "Hier. AutoGaze Density",
    }
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"$\mathtt{max\_tiles}$ & Family & Method & Overall & $\Delta$ uniform & AV & Household \\",
        r"\midrule",
    ]
    for setting in ("mtv1", "mtv48"):
        setting_rows = [row for row in rows if row["setting"] == setting]
        best_accuracy = max(row["accuracy"] for row in setting_rows)
        for row in setting_rows:
            accuracy = f"{row['accuracy'] * 100:.2f}"
            if row["accuracy"] == best_accuracy:
                accuracy = rf"\textbf{{{accuracy}}}"
            delta = "--" if row["method"] == "uniform" else f"{row['delta_vs_uniform'] * 100:+.2f}"
            lines.append(
                f"{row['max_tiles_video']} & {row['method_family'].title()} & "
                f"{escaped_labels[row['method_label']]} & "
                f"{accuracy} & {delta} & {row['av_accuracy'] * 100:.2f} & "
                f"{row['household_accuracy'] * 100:.2f} \\\\"
            )
        if setting == "mtv1":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output_dir / "results_table.tex").write_text("\n".join(lines))


def write_key_findings(rows: list[dict], output_dir: Path) -> None:
    lookup = {(row["setting"], row["method"]): row for row in rows}
    mtv1_best = max((row for row in rows if row["setting"] == "mtv1"), key=lambda row: row["accuracy"])
    mtv48_best = max((row for row in rows if row["setting"] == "mtv48"), key=lambda row: row["accuracy"])
    hierarchical_mtv1_best = max(
        (
            row
            for row in rows
            if row["setting"] == "mtv1" and row["method_family"] == "hierarchical"
        ),
        key=lambda row: row["accuracy"],
    )
    hierarchical_mtv48_best = max(
        (
            row
            for row in rows
            if row["setting"] == "mtv48" and row["method_family"] == "hierarchical"
        ),
        key=lambda row: row["accuracy"],
    )
    info_mtv1 = lookup[("mtv1", "infoshot_reimpl")]
    info_mtv48 = lookup[("mtv48", "infoshot_reimpl")]
    lines = [
        "# Presentation Notes",
        "",
        "## Main findings",
        "",
        f"- At `max_tiles_video=1`, **{mtv1_best['method_label']}** is best at "
        f"**{mtv1_best['accuracy'] * 100:.2f}%**, "
        f"{mtv1_best['delta_vs_uniform'] * 100:+.2f} percentage points versus uniform.",
        f"- At `max_tiles_video=48`, **{mtv48_best['method_label']}** is best at "
        f"**{mtv48_best['accuracy'] * 100:.2f}%**, "
        f"{mtv48_best['delta_vs_uniform'] * 100:+.2f} percentage points versus uniform.",
        f"- The strongest hierarchical method at `mtv1` is **{hierarchical_mtv1_best['method_label']}** "
        f"at **{hierarchical_mtv1_best['accuracy'] * 100:.2f}%** "
        f"({hierarchical_mtv1_best['delta_vs_uniform'] * 100:+.2f} points versus uniform).",
        f"- The strongest hierarchical method at `mtv48` is **{hierarchical_mtv48_best['method_label']}** "
        f"at **{hierarchical_mtv48_best['accuracy'] * 100:.2f}%** "
        f"({hierarchical_mtv48_best['delta_vs_uniform'] * 100:+.2f} points versus uniform).",
        f"- InfoShot-reimpl changes from {info_mtv1['accuracy'] * 100:.2f}% at `mtv1` "
        f"to {info_mtv48['accuracy'] * 100:.2f}% at `mtv48`, a "
        f"{(info_mtv48['accuracy'] - info_mtv1['accuracy']) * 100:+.2f} point interaction.",
        "- Both external and hierarchical selector rankings depend strongly on the downstream spatial-token budget; "
        "frame selection and AutoGaze capacity should not be evaluated independently.",
        "- Every run uses the same 128 visited frames and eight temporal tiles, so these are quality comparisons, "
        "not frame-budget comparisons.",
        "",
        "## Statistical caveat",
        "",
        "HLVid contains 268 questions over 77 videos. Questions from the same video are correlated, "
        "so confidence intervals resample videos rather than treating all questions as independent. "
        "The point estimates are appropriate for presentation, but small differences should not be described "
        "as conclusive unless the paired interval excludes zero.",
        "",
        "## Generated artifacts",
        "",
        "- `overall_accuracy_mtv_comparison`: direct mtv1/mtv48 interaction across external and hierarchical methods.",
        "- `accuracy_delta_vs_uniform`: selector improvement relative to the matching baseline.",
        "- `category_accuracy_heatmap`: overall, AV, and household accuracy.",
        "- `paired_question_flips`: gains and losses on identical questions.",
        "- `results_table.md`, `results_table.tex`, and `results_tidy.csv`: reusable tables and source data.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args)
    rows = build_table_rows(runs, args.bootstrap_samples, args.seed)

    configure_plotting()
    plot_overall_accuracy(rows, args.output_dir)
    plot_delta_vs_uniform(rows, args.output_dir)
    plot_accuracy_heatmap(rows, args.output_dir)
    plot_paired_flips(rows, args.output_dir)
    write_csv(rows, args.output_dir)
    write_markdown_table(rows, args.output_dir)
    write_latex_table(rows, args.output_dir)
    write_key_findings(rows, args.output_dir)
    print(f"Wrote presentation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
