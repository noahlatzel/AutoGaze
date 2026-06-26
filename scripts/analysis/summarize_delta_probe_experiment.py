# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize and plot a supervised delta-probe experiment config."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIMARY_METRICS = (
    "accuracy",
    "top2_accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mean_abs_delta_error",
    "mean_abs_seconds_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root from config.")
    return parser.parse_args()


def metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in PRIMARY_METRICS if key in metrics}


def run_short_label(run: dict[str, Any]) -> str:
    command = run.get("command", {})
    feature_key = command.get("feature_key", "features")
    probe = command.get("probe", "probe")
    return f"{feature_key.replace('_', ' ')}\n{probe}"


def load_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in config["runs"]:
        metrics_path = Path(run["output"])
        data = json.loads(metrics_path.read_text())
        row = {
            "id": run["id"],
            "status": "completed",
            "metrics_path": str(metrics_path),
            "feature_key": run.get("command", {}).get("feature_key") or data.get("metadata", {}).get("feature_key"),
            "probe": run.get("command", {}).get("probe") or data.get("metadata", {}).get("probe"),
            "input_mode": data.get("metadata", {}).get("input_mode"),
            "architecture": data.get("metadata", {}).get("architecture"),
            "loss": data.get("metadata", {}).get("loss"),
            "ordinal_alpha": data.get("metadata", {}).get("ordinal_alpha"),
            "feature_probe": metric_subset(data["feature_probe"]),
            "metadata_probe": metric_subset(data["metadata_probe"]),
            "baselines": {key: metric_subset(value) for key, value in data.get("baselines", {}).items()},
            "label_counts": data["feature_probe"].get("label_counts", {}),
            "prediction_counts": data["feature_probe"].get("prediction_counts", {}),
        }
        rows.append({"run": run, "data": data, "summary": row})
    return rows


def write_results_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row["summary"], sort_keys=True) + "\n")


def plot_accuracy(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [run_short_label(row["run"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.24
    metric_names = ("accuracy", "top2_accuracy", "balanced_accuracy")

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    for offset, metric in enumerate(metric_names):
        values = [row["data"]["feature_probe"][metric] for row in rows]
        ax.bar(x + (offset - 1) * width, values, width, label=metric.replace("_", " "))

    metadata = rows[0]["data"]["metadata_probe"]
    ax.axhline(metadata["accuracy"], color="tab:blue", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(metadata["balanced_accuracy"], color="tab:green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score")
    ax.set_title("Delta Probe Validation Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_delta_error(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [run_short_label(row["run"]) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=180, sharex=True)
    for ax, metric, ylabel in (
        (axes[0], "mean_abs_delta_error", "Delta classes"),
        (axes[1], "mean_abs_seconds_error", "Seconds"),
    ):
        values = [row["data"]["feature_probe"][metric] for row in rows]
        ax.bar(x, values, color="tab:purple", alpha=0.8)
        ax.axhline(rows[0]["data"]["metadata_probe"][metric], color="black", linestyle="--", linewidth=1.5, alpha=0.65)
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(best: dict[str, Any], output_path: Path) -> None:
    data = best["data"]
    deltas = [str(delta) for delta in data["metadata"]["deltas"]]
    confusion = np.asarray(data["feature_probe"]["confusion"], dtype=float)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    image = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(deltas)))
    ax.set_yticks(np.arange(len(deltas)))
    ax.set_xticklabels(deltas)
    ax.set_yticklabels(deltas)
    ax.set_xlabel("Predicted delta")
    ax.set_ylabel("Label delta")
    ax.set_title(f"Best Probe Confusion: {best['run']['id']}")
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized count")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_delta_distribution(best: dict[str, Any], output_path: Path) -> None:
    data = best["data"]
    deltas = [str(delta) for delta in data["metadata"]["deltas"]]
    label_counts = data["feature_probe"]["label_counts"]
    prediction_counts = data["feature_probe"]["prediction_counts"]
    labels = [label_counts.get(str(i), 0) for i in range(len(deltas))]
    predictions = [prediction_counts.get(str(i), 0) for i in range(len(deltas))]

    x = np.arange(len(deltas))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.bar(x - width / 2, labels, width, label="labels")
    ax.bar(x + width / 2, predictions, width, label="predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(deltas)
    ax.set_xlabel("Delta")
    ax.set_ylabel("Validation examples")
    ax.set_title(f"Delta Distribution: {best['run']['id']}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(rows: list[dict[str, Any]], best: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)

    for row in rows:
        history = row["data"].get("training_history", {}).get("feature_probe", [])
        if not history:
            continue
        epochs = [item["epoch"] for item in history]
        label = run_short_label(row["run"]).replace("\n", " ")
        linewidth = 2.2 if row["run"]["id"] == best["run"]["id"] else 1.0
        alpha = 1.0 if row["run"]["id"] == best["run"]["id"] else 0.35
        axes[0].plot(epochs, [item["train_loss"] for item in history], label=label, linewidth=linewidth, alpha=alpha)
        axes[1].plot(epochs, [item["val_loss"] for item in history], label=label, linewidth=linewidth, alpha=alpha)
        axes[2].plot(epochs, [item["val_accuracy"] for item in history], label=label, linewidth=linewidth, alpha=alpha)

    for ax, title, ylabel in (
        (axes[0], "Train Loss", "Cross-entropy"),
        (axes[1], "Validation Loss", "Cross-entropy"),
        (axes[2], "Validation Accuracy", "Accuracy"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_summary(config: dict[str, Any], rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    best_accuracy = max(rows, key=lambda row: row["data"]["feature_probe"]["accuracy"])
    best_balanced = max(rows, key=lambda row: row["data"]["feature_probe"]["balanced_accuracy"])
    best_delta_error = min(rows, key=lambda row: row["data"]["feature_probe"]["mean_abs_delta_error"])
    metadata_probe = rows[0]["summary"]["metadata_probe"]
    return {
        "id": config["id"],
        "family": config["family"],
        "status": "completed",
        "config": config.get("_config_path"),
        "output_root": str(output_root),
        "num_runs": len(rows),
        "runs": [row["summary"] for row in rows],
        "metadata_probe": metadata_probe,
        "best_feature_probe_by_accuracy": rowsummary(best_accuracy),
        "best_feature_probe_by_balanced_accuracy": rowsummary(best_balanced),
        "best_feature_probe_by_delta_error": rowsummary(best_delta_error),
    }


def rowsummary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["run"]["id"],
        "feature_key": row["summary"]["feature_key"],
        "probe": row["summary"]["probe"],
        "input_mode": row["summary"].get("input_mode"),
        "architecture": row["summary"].get("architecture"),
        "loss": row["summary"].get("loss"),
        "ordinal_alpha": row["summary"].get("ordinal_alpha"),
        **row["summary"]["feature_probe"],
    }


def write_run_manifest(config: dict[str, Any], rows: list[dict[str, Any]], output_root: Path) -> None:
    manifest = {
        "id": config["id"],
        "family": config["family"],
        "status": "completed",
        "config": config.get("_config_path"),
        "command": str(output_root / "command.sh"),
        "results": str(output_root / "results.jsonl"),
        "summary": str(output_root / "summary.json"),
        "runs": [
            {
                "id": row["run"]["id"],
                "status": "completed",
                "metrics": row["summary"]["metrics_path"],
            }
            for row in rows
        ],
    }
    (output_root / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    config["_config_path"] = str(config_path)
    output_root = Path(args.output_root or config["outputs"]["root"])
    plots_dir = Path(config["outputs"].get("plots_dir", output_root / "plots"))
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runs(config)
    if len(rows) != len(config["runs"]):
        raise RuntimeError(f"Expected {len(config['runs'])} runs, found {len(rows)}.")

    best = max(rows, key=lambda row: row["data"]["feature_probe"]["balanced_accuracy"])
    write_results_jsonl(rows, output_root / "results.jsonl")
    plot_accuracy(rows, plots_dir / "accuracy_by_run.png")
    plot_delta_error(rows, plots_dir / "delta_error_by_run.png")
    plot_confusion(best, plots_dir / "best_confusion_matrix.png")
    plot_delta_distribution(best, plots_dir / "best_delta_distribution.png")
    plot_training_curves(rows, best, plots_dir / "training_curves.png")

    summary = build_summary(config, rows, output_root)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_run_manifest(config, rows, output_root)
    print(json.dumps(summary["best_feature_probe_by_balanced_accuracy"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
