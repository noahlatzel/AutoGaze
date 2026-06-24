# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot supervised delta-probe loss curves from metrics JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Metrics JSON files from train_delta_probe.py.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    return parser.parse_args()


def run_label(path: Path, metadata: dict) -> str:
    feature = metadata.get("feature_key") or "features"
    probe = metadata.get("probe")
    fields = metadata.get("metadata_fields") or []
    suffix = "+runtime" if fields == ["effective_fps", "frame_count", "start"] else ""
    return f"{feature}{suffix}, {probe}"


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.inputs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    for path in paths:
        data = json.loads(path.read_text())
        label = run_label(path, data["metadata"])
        history = data.get("training_history", {}).get("feature_probe", [])
        if not history:
            continue
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=label, linewidth=2)
        axes[1].plot(epochs, [row["val_loss"] for row in history], label=label, linewidth=2)

    axes[0].set_title("Feature Probe Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Feature Probe Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("Pixel MSE t=0.04 Supervised Delta-Probe Loss Curves", fontsize=12)
    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
