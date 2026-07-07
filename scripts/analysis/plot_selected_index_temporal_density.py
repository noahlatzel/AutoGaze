# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot temporal density for selected-frame-index JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", required=True, help="Selected-index JSONL.")
    parser.add_argument("--dataset-root", default="/storage/slurm/latn/data/AutoGaze/HLVid")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--title-prefix", default="Selected-index sampling")
    parser.add_argument(
        "--group-by-video",
        action="store_true",
        help="Aggregate multiple question records for the same video before plotting.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_category_by_question(dataset_root: Path) -> dict[str, str]:
    parquet_path = dataset_root / "data" / "test-00000-of-00001.parquet"
    rows = pq.read_table(parquet_path).to_pylist()
    return {str(row["question_id"]): str(row.get("category", "")) for row in rows}


def aggregate_by_video(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["video_path"])].append(record)

    aggregated = []
    for video_path, rows in sorted(grouped.items()):
        frame_count = int(rows[0]["frame_count"])
        indices = sorted({int(index) for row in rows for index in row["selected_source_indices"]})
        aggregated.append(
            {
                "question_id": ",".join(str(row["question_id"]) for row in rows),
                "video_path": video_path,
                "method": rows[0].get("method"),
                "num_frames": len(indices),
                "selected_source_indices": indices,
                "frame_count": frame_count,
                "metadata": {"num_questions": len(rows)},
            }
        )
    return aggregated


def normalized_bin_counts(indices: list[int], frame_count: int, bins: int) -> np.ndarray:
    edges = np.linspace(0, frame_count, bins + 1)
    counts, _ = np.histogram(indices, bins=edges)
    return counts


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    records = load_jsonl(Path(args.indices))
    if not records:
        raise SystemExit(f"No records found in {args.indices}")
    category_by_question = load_category_by_question(Path(args.dataset_root))
    if args.group_by_video:
        records = aggregate_by_video(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = []
    labels = []
    categories = []
    selected_ratios = []
    for row in records:
        frame_count = int(row["frame_count"])
        indices = [int(index) for index in row["selected_source_indices"]]
        matrix.append(normalized_bin_counts(indices, frame_count, args.bins))
        label = str(row["video_path"]).replace(".mp4", "").replace("clip_", "")
        if not args.group_by_video:
            label = f"{row['question_id']}:{label}"
        labels.append(label)
        categories.append(category_by_question.get(str(row["question_id"]), "mixed" if args.group_by_video else ""))
        selected_ratios.append(len(set(indices)) / max(frame_count, 1))
    matrix_arr = np.asarray(matrix)

    fig_h = max(4.0, min(42.0, 0.18 * len(records) + 2.0))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    im = ax.imshow(matrix_arr, aspect="auto", cmap="viridis")
    ax.set_title(f"{args.title_prefix}: selected frames per normalized time bin")
    ax.set_xlabel("Normalized video time bin")
    ax.set_ylabel("Video" if args.group_by_video else "Question/video")
    ax.set_xticks(np.linspace(0, args.bins - 1, 9))
    ax.set_xticklabels([f"{value:.2f}" for value in np.linspace(0, 1, 9)])
    if len(labels) <= 80:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Selected frames")
    fig.tight_layout()
    heatmap_path = output_dir / "temporal_density_heatmap.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, fig_h))
    for y, row in enumerate(records):
        frame_count = int(row["frame_count"])
        xs = [int(index) / max(frame_count - 1, 1) for index in row["selected_source_indices"]]
        ax.vlines(xs, y - 0.36, y + 0.36, color="black", linewidth=0.22, alpha=0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(records))
    ax.invert_yaxis()
    ax.set_title(f"{args.title_prefix}: selected-frame positions over normalized video time")
    ax.set_xlabel("Normalized video time")
    ax.set_ylabel("Video" if args.group_by_video else "Question/video")
    if len(labels) <= 80:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_yticks([])
    fig.tight_layout()
    timeline_path = output_dir / "selected_frame_timeline.png"
    fig.savefig(timeline_path, dpi=180)
    plt.close(fig)

    per_video = []
    for row, counts, ratio, category in zip(records, matrix_arr, selected_ratios, categories):
        indices = [int(index) for index in row["selected_source_indices"]]
        diffs = np.diff(sorted(set(indices)))
        per_video.append(
            {
                "question_id": str(row["question_id"]),
                "video_path": row["video_path"],
                "category": category,
                "num_selected_frames": len(indices),
                "num_unique_selected_frames": len(set(indices)),
                "selected_frame_ratio": float(ratio),
                "nonzero_bins": int((counts > 0).sum()),
                "max_bin_count": int(counts.max()),
                "bin_counts": counts.tolist(),
                "min_gap": int(diffs.min()) if len(diffs) else None,
                "median_gap": float(np.median(diffs)) if len(diffs) else None,
                "max_gap": int(diffs.max()) if len(diffs) else None,
            }
        )

    summary = {
        "indices": args.indices,
        "num_records": len(records),
        "group_by_video": args.group_by_video,
        "bins": args.bins,
        "heatmap": str(heatmap_path),
        "timeline": str(timeline_path),
        "method_counts": dict(Counter(str(row.get("method")) for row in records)),
        "category_counts": dict(Counter(categories)),
        "mean_unique_selected_frames": float(np.mean([row["num_unique_selected_frames"] for row in per_video])),
        "mean_selected_frame_ratio": float(np.mean(selected_ratios)),
        "mean_nonzero_bins": float(np.mean([row["nonzero_bins"] for row in per_video])),
        "mean_max_bin_count": float(np.mean([row["max_bin_count"] for row in per_video])),
        "per_video": per_video,
    }
    summary_path = output_dir / "temporal_density_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
