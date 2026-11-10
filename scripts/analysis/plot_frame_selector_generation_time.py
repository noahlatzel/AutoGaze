# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot SLURM wall time for generating HLVid frame-selector indices."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIR = Path("outputs/hlvid/external_frame_selector_comparison_128")

INDEX_JOBS = {
    "aks": ("AKS-style", ("1616353",)),
    "bolt": ("BOLT", ("1616355", "1616957")),
    "sealing_mdf": ("Sealing MDF", ("1616354",)),
    "sealing_mif": ("Sealing MIF", ("1616356",)),
    "infoshot_reimpl": ("InfoShot-reimpl", ("1626645",)),
}

HIERARCHICAL_EVALUATION_JOBS = {
    "mtv1": {
        "Hier. Pixel Diversity": ("1615395",),
        "Hier. DINOv2 SemDiv": ("1615394",),
        "Hier. SigLIP SemDiv": ("1615396",),
        "Hier. AutoGaze Density": ("1615393",),
    },
    "mtv48": {
        "Hier. Pixel Diversity": ("1615401",),
        "Hier. DINOv2 SemDiv": ("1615400", "1616017"),
        "Hier. SigLIP SemDiv": ("1615399",),
        "Hier. AutoGaze Density": ("1615398",),
    },
}

SUMMARY_PATHS = {
    "mtv1": {
        "uniform": Path("outputs/hlvid/uniform_128_mtv1/summary.json"),
        **{
            method: Path("outputs/hlvid/external_frame_selectors_128_mtv1") / method / "summary.json"
            for method in INDEX_JOBS
        },
    },
    "mtv48": {
        "uniform": Path("outputs/hlvid/uniform_128_mtv48/summary.json"),
        **{
            method: Path("outputs/hlvid/external_frame_selectors_128_mtv48") / method / "summary.json"
            for method in INDEX_JOBS
        },
    },
}

METHOD_COLORS = {
    "aks": "#4C78A8",
    "bolt": "#2A9D8F",
    "sealing_mdf": "#59A14F",
    "sealing_mif": "#E9C46A",
    "infoshot_reimpl": "#E76F51",
}


@dataclass(frozen=True)
class AccountingRecord:
    job_id: str
    job_name: str
    state: str
    node: str
    elapsed_seconds: int
    gpu: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-time", default="2026-07-01")
    return parser.parse_args()


def query_accounting(job_ids: list[str], start_time: str) -> dict[str, AccountingRecord]:
    command = [
        "sacct",
        "-X",
        "-j",
        ",".join(job_ids),
        "--starttime",
        start_time,
        "--format=JobIDRaw,JobName,State,NodeList,ElapsedRaw,AllocTRES",
        "-P",
        "-n",
    ]
    output = subprocess.check_output(command, text=True)
    records = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, job_name, state, node, elapsed_raw, tres = line.split("|")[:6]
        gpu = next(
            (
                item.split("=", 1)[0].split(":", 1)[1]
                for item in tres.split(",")
                if item.startswith("gres/gpu:") and item.count("=") == 1
            ),
            "unknown",
        )
        records[job_id] = AccountingRecord(
            job_id=job_id,
            job_name=job_name,
            state=state,
            node=node,
            elapsed_seconds=int(elapsed_raw),
            gpu=gpu,
        )
    missing = sorted(set(job_ids) - set(records))
    if missing:
        raise RuntimeError(f"SLURM accounting records not found: {', '.join(missing)}")
    return records


def format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def build_index_rows(records: dict[str, AccountingRecord]) -> list[dict]:
    rows = []
    for method, (label, job_ids) in INDEX_JOBS.items():
        method_records = [records[job_id] for job_id in job_ids]
        elapsed_seconds = sum(record.elapsed_seconds for record in method_records)
        rows.append(
            {
                "method": method,
                "method_label": label,
                "job_ids": ",".join(job_ids),
                "stages": len(job_ids),
                "elapsed_seconds": elapsed_seconds,
                "elapsed": format_duration(elapsed_seconds),
                "gpu_allocation_hours": elapsed_seconds / 3600,
                "nodes": ",".join(record.node for record in method_records),
                "gpus": ",".join(record.gpu for record in method_records),
                "states": ",".join(record.state for record in method_records),
            }
        )
    return rows


def plot_generation_time(rows: list[dict], output_dir: Path) -> None:
    labels = [row["method_label"] for row in rows]
    hours = np.asarray([row["gpu_allocation_hours"] for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    ax.barh(y, hours, color=[METHOD_COLORS[row["method"]] for row in rows], height=0.64)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("GPU allocation wall time (hours)")
    ax.set_title("Cost to Generate 128-Frame HLVid Selections", loc="left", fontweight="bold")
    ax.set_xlim(0, max(hours) + 2.0)
    for index, (value, row) in enumerate(zip(hours, rows)):
        ax.text(
            value + 0.18,
            index,
            f"{value:.2f} h  |  job {row['job_ids']}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    save_figure(fig, output_dir, "index_generation_time")


def load_accuracy(path: Path) -> float:
    return float(json.loads(path.read_text())["accuracy"]) * 100


def plot_accuracy_vs_time(rows: list[dict], output_dir: Path) -> None:
    hours = {row["method"]: row["gpu_allocation_hours"] for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharey=True)
    for ax, setting, title in zip(
        axes,
        ("mtv1", "mtv48"),
        ("max_tiles_video = 1", "max_tiles_video = 48"),
    ):
        uniform = load_accuracy(SUMMARY_PATHS[setting]["uniform"])
        ax.scatter(0, uniform, color="#666666", s=95, marker="D", zorder=3)
        ax.annotate("Uniform", (0, uniform), xytext=(7, 5), textcoords="offset points", fontsize=9)
        for method, (label, _) in INDEX_JOBS.items():
            accuracy = load_accuracy(SUMMARY_PATHS[setting][method])
            ax.scatter(hours[method], accuracy, color=METHOD_COLORS[method], s=105, zorder=3)
            ax.annotate(label, (hours[method], accuracy), xytext=(6, 5), textcoords="offset points", fontsize=8)
        ax.axhline(uniform, color="#999999", linestyle="--", linewidth=1.0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("One-time index-generation GPU hours")
        ax.set_xlim(-0.5, max(hours.values()) + 2.2)
    axes[0].set_ylabel("HLVid accuracy (%)")
    fig.suptitle("Accuracy Versus Measured Selector Preprocessing Cost", x=0.04, ha="left", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "accuracy_vs_index_generation_time")


def write_csv(rows: list[dict], output_dir: Path) -> None:
    with (output_dir / "index_generation_time.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict],
    records: dict[str, AccountingRecord],
    output_dir: Path,
) -> None:
    lines = [
        "# Frame-Selection Timing",
        "",
        "## Dedicated index-generation jobs",
        "",
        "| Method | SLURM job ID(s) | Active allocation time | GPU/node |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | `{row['job_ids']}` | {row['elapsed']} "
            f"({row['gpu_allocation_hours']:.2f} h) | {row['gpus']} / {row['nodes']} |"
        )
    lines.extend(
        [
            "",
            "Times are active SLURM allocation wall times, excluding queue delay. BOLT is a two-stage pipeline, "
            "so its reported time sums feature extraction (`1616355`) and index selection (`1616957`). "
            "Uniform sampling has no dedicated GPU preprocessing job.",
            "",
            "AKS-style and Sealing MIF are question-conditioned and produce a selection for every question. "
            "InfoShot-reimpl computes one query-agnostic selection per unique video. These totals therefore measure "
            "the implementations that were benchmarked, not an architecture-independent cost.",
            "",
            "## Hierarchical temporal density",
            "",
            "Hierarchical indices were generated online inside the NVILA evaluator and cached per video only for the "
            "lifetime of that process. The runner did not record selection time separately, so its SLURM durations "
            "include both frame selection and NVILA inference and must not be placed on the dedicated preprocessing "
            "cost axis.",
            "",
            "| Setting | Method | SLURM job ID(s) | Combined allocation time |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for setting in ("mtv1", "mtv48"):
        for label, job_ids in HIERARCHICAL_EVALUATION_JOBS[setting].items():
            elapsed = sum(records[job_id].elapsed_seconds for job_id in job_ids)
            lines.append(f"| `{setting}` | {label} | `{','.join(job_ids)}` | {format_duration(elapsed)} |")
    lines.extend(
        [
            "",
            "The DINOv2 `mtv48` total includes the timed-out allocation `1615400` and resumed allocation `1616017`.",
            "These combined values are provenance records, not isolated selector runtimes.",
            "",
        ]
    )
    (output_dir / "index_generation_time.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    job_ids = sorted(
        {
            job_id
            for _, job_ids in INDEX_JOBS.values()
            for job_id in job_ids
        }
        | {
            job_id
            for setting_jobs in HIERARCHICAL_EVALUATION_JOBS.values()
            for job_ids in setting_jobs.values()
            for job_id in job_ids
        }
    )
    records = query_accounting(job_ids, args.start_time)
    rows = build_index_rows(records)
    plot_generation_time(rows, args.output_dir)
    plot_accuracy_vs_time(rows, args.output_dir)
    write_csv(rows, args.output_dir)
    write_markdown(rows, records, args.output_dir)
    print(f"Wrote frame-selection timing artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
