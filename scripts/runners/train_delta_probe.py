# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train Experiment 1 delta probes from cached frozen AutoGaze features."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", required=True, help="Feature cache produced by extract_delta_probe_features.py.")
    parser.add_argument("--output", required=True, help="Output JSON metrics path.")
    parser.add_argument(
        "--feature-key",
        default=None,
        help="Representation key when the cache contains features_by_representation.",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="Optional Experiment 0B label JSON used to override labels in the feature cache.",
    )
    parser.add_argument(
        "--heuristic",
        default=None,
        help="Heuristic to read from --labels-json when overriding labels.",
    )
    parser.add_argument("--probe", choices=["linear", "mlp"], default="linear", help="Probe architecture.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="none",
        help="Class weighting for cross-entropy. balanced uses inverse train-set class frequency.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Video-level validation fraction.")
    parser.add_argument(
        "--filter-key",
        default=None,
        help="Optional example metadata key to filter before splitting, e.g. fps_bucket.",
    )
    parser.add_argument(
        "--filter-value",
        default=None,
        help="Optional metadata value for --filter-key.",
    )
    parser.add_argument(
        "--metadata-fields",
        default="dataset_id,fps_bucket,effective_fps,frame_count,start",
        help=(
            "Comma-separated metadata fields for the metadata-only baseline. "
            "Choices: dataset_id, fps_bucket, effective_fps, frame_count, start. "
            "Use runtime-plausible fields such as effective_fps,frame_count,start "
            "to avoid dataset/source-id shortcuts."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser.parse_args()


def stable_fraction(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_indices(examples: Sequence[dict], val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    train_indices = []
    val_indices = []
    for idx, example in enumerate(examples):
        key = example["video_path"]
        if stable_fraction(key) < val_fraction:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    if (not train_indices or not val_indices) and len(examples) > 1:
        train_indices = []
        val_indices = []
        for idx, example in enumerate(examples):
            key = f"{example['video_path']}::{example.get('start', idx)}::{idx}"
            if stable_fraction(key) < val_fraction:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
    return torch.tensor(train_indices, dtype=torch.long), torch.tensor(val_indices, dtype=torch.long)


def build_probe(kind: str, input_dim: int, num_classes: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_dim, num_classes)
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(input_dim, num_classes),
    )


def parse_metadata_fields(value: str) -> List[str]:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    valid = {"dataset_id", "fps_bucket", "effective_fps", "frame_count", "start"}
    unknown = sorted(set(fields) - valid)
    if unknown:
        raise ValueError(f"Unknown metadata fields: {unknown}. Valid fields: {sorted(valid)}")
    if not fields:
        raise ValueError("--metadata-fields must contain at least one field")
    return fields


def metadata_values(examples: Sequence[dict]) -> tuple[List[str], List[str]]:
    datasets = sorted({example.get("label_source_name") or "unknown" for example in examples})
    fps_buckets = sorted({example.get("fps_bucket") or "unknown" for example in examples})
    return datasets, fps_buckets


def build_metadata_features(examples: Sequence[dict], fields: Sequence[str]) -> torch.Tensor:
    datasets, fps_buckets = metadata_values(examples)
    dataset_to_idx = {value: idx for idx, value in enumerate(datasets)}
    bucket_to_idx = {value: idx for idx, value in enumerate(fps_buckets)}

    rows = []
    for example in examples:
        row = []
        if "dataset_id" in fields:
            dataset_onehot = [0.0] * len(datasets)
            dataset_onehot[dataset_to_idx[example.get("label_source_name") or "unknown"]] = 1.0
            row.extend(dataset_onehot)
        if "fps_bucket" in fields:
            bucket_onehot = [0.0] * len(fps_buckets)
            bucket_onehot[bucket_to_idx[example.get("fps_bucket") or "unknown"]] = 1.0
            row.extend(bucket_onehot)
        if "effective_fps" in fields:
            row.append(float(example.get("effective_fps") or 0.0))
        if "frame_count" in fields:
            row.append(float(example.get("frame_count") or 0.0))
        if "start" in fields:
            row.append(float(example.get("start") or 0.0))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def normalize(train_x: torch.Tensor, all_x: torch.Tensor) -> torch.Tensor:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (all_x - mean) / std


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    values = []
    for class_idx in range(num_classes):
        true_pos = ((y_true == class_idx) & (y_pred == class_idx)).sum().item()
        false_pos = ((y_true != class_idx) & (y_pred == class_idx)).sum().item()
        false_neg = ((y_true == class_idx) & (y_pred != class_idx)).sum().item()
        denom = 2 * true_pos + false_pos + false_neg
        values.append(0.0 if denom == 0 else (2 * true_pos) / denom)
    return float(np.mean(values))


def balanced_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    values = []
    for class_idx in range(num_classes):
        mask = y_true == class_idx
        if mask.any():
            values.append((y_pred[mask] == class_idx).float().mean().item())
    return float(np.mean(values)) if values else 0.0


def evaluate_predictions(
    y_true: torch.Tensor,
    logits_or_pred: torch.Tensor,
    deltas: Sequence[int],
    examples: Sequence[dict],
    indices: torch.Tensor,
) -> Dict[str, object]:
    if logits_or_pred.ndim == 2:
        pred = logits_or_pred.argmax(dim=-1)
        top2 = logits_or_pred.topk(k=min(2, logits_or_pred.shape[-1]), dim=-1).indices
        top2_acc = (top2 == y_true[:, None]).any(dim=-1).float().mean().item()
    else:
        pred = logits_or_pred
        top2_acc = None

    delta_tensor = torch.tensor(deltas, dtype=torch.float32)
    true_delta = delta_tensor[y_true]
    pred_delta = delta_tensor[pred]
    seconds_errors = []
    for local_idx, example_idx in enumerate(indices.tolist()):
        fps = examples[example_idx].get("effective_fps")
        if fps:
            seconds_errors.append(abs(float(pred_delta[local_idx] - true_delta[local_idx]) / fps))

    num_classes = len(deltas)
    metrics = {
        "accuracy": float((pred == y_true).float().mean().item()),
        "top2_accuracy": top2_acc,
        "macro_f1": macro_f1(y_true, pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, pred, num_classes),
        "mean_abs_delta_error": float((pred_delta - true_delta).abs().mean().item()),
        "mean_abs_seconds_error": float(np.mean(seconds_errors)) if seconds_errors else None,
        "label_counts": dict(sorted(Counter(y_true.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(pred.tolist()).items())),
        "confusion": confusion_matrix(y_true, pred, num_classes),
    }
    metrics["by_fps_bucket"] = grouped_metrics(y_true, pred, logits_or_pred, deltas, examples, indices, "fps_bucket")
    metrics["by_dataset"] = grouped_metrics(y_true, pred, logits_or_pred, deltas, examples, indices, "label_source_name")
    return metrics


def grouped_metrics(
    y_true: torch.Tensor,
    pred: torch.Tensor,
    logits_or_pred: torch.Tensor,
    deltas: Sequence[int],
    examples: Sequence[dict],
    indices: torch.Tensor,
    key: str,
) -> Dict[str, object]:
    groups = defaultdict(list)
    for local_idx, example_idx in enumerate(indices.tolist()):
        groups[examples[example_idx].get(key) or "unknown"].append(local_idx)

    output = {}
    for group, local_indices in sorted(groups.items()):
        group_indices = torch.tensor(local_indices, dtype=torch.long)
        group_logits_or_pred = logits_or_pred[group_indices] if logits_or_pred.ndim == 2 else pred[group_indices]
        output[group] = evaluate_predictions_shallow(
            y_true[group_indices],
            group_logits_or_pred,
            deltas,
            examples,
            indices[group_indices],
        )
    return output


def evaluate_predictions_shallow(
    y_true: torch.Tensor,
    logits_or_pred: torch.Tensor,
    deltas: Sequence[int],
    examples: Sequence[dict],
    indices: torch.Tensor,
) -> Dict[str, object]:
    if logits_or_pred.ndim == 2:
        pred = logits_or_pred.argmax(dim=-1)
        top2 = logits_or_pred.topk(k=min(2, logits_or_pred.shape[-1]), dim=-1).indices
        top2_acc = (top2 == y_true[:, None]).any(dim=-1).float().mean().item()
    else:
        pred = logits_or_pred
        top2_acc = None

    delta_tensor = torch.tensor(deltas, dtype=torch.float32)
    true_delta = delta_tensor[y_true]
    pred_delta = delta_tensor[pred]
    seconds_errors = []
    for local_idx, example_idx in enumerate(indices.tolist()):
        fps = examples[example_idx].get("effective_fps")
        if fps:
            seconds_errors.append(abs(float(pred_delta[local_idx] - true_delta[local_idx]) / fps))

    num_classes = len(deltas)
    return {
        "num_examples": int(y_true.numel()),
        "accuracy": float((pred == y_true).float().mean().item()),
        "top2_accuracy": top2_acc,
        "macro_f1": macro_f1(y_true, pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, pred, num_classes),
        "mean_abs_delta_error": float((pred_delta - true_delta).abs().mean().item()),
        "mean_abs_seconds_error": float(np.mean(seconds_errors)) if seconds_errors else None,
        "label_counts": dict(sorted(Counter(y_true.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(pred.tolist()).items())),
    }


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> List[List[int]]:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        matrix[true, pred] += 1
    return matrix.tolist()


def majority_predictions(train_y: torch.Tensor, val_y: torch.Tensor) -> torch.Tensor:
    majority = Counter(train_y.tolist()).most_common(1)[0][0]
    return torch.full_like(val_y, majority)


def grouped_majority_predictions(
    train_examples: Sequence[dict],
    train_y: torch.Tensor,
    val_examples: Sequence[dict],
    val_y: torch.Tensor,
    key: str,
) -> torch.Tensor:
    global_majority = Counter(train_y.tolist()).most_common(1)[0][0]
    grouped = defaultdict(list)
    for example, label in zip(train_examples, train_y.tolist()):
        grouped[example.get(key) or "unknown"].append(label)
    lookup = {
        group: Counter(labels).most_common(1)[0][0]
        for group, labels in grouped.items()
    }
    preds = [lookup.get(example.get(key) or "unknown", global_majority) for example in val_examples]
    return torch.tensor(preds, dtype=val_y.dtype)


def fixed_delta_predictions(delta: int, deltas: Sequence[int], val_y: torch.Tensor) -> torch.Tensor:
    closest_idx = min(range(len(deltas)), key=lambda idx: abs(deltas[idx] - delta))
    return torch.full_like(val_y, closest_idx)


def labels_from_json(examples: Sequence[dict], labels_json: str, heuristic: str, deltas: Sequence[int]) -> torch.Tensor:
    data = json.loads(Path(labels_json).read_text())
    delta_to_index = {int(delta): idx for idx, delta in enumerate(deltas)}
    labels_by_key = {}
    for video in data["videos"]:
        for tile in video["tiles"]:
            label = tile["labels"].get(heuristic)
            if label is None:
                continue
            labels_by_key[(video["video_path"], int(tile["start"]))] = delta_to_index[int(label)]

    labels = []
    missing = []
    for idx, example in enumerate(examples):
        key = (example["video_path"], int(example["start"]))
        if key not in labels_by_key:
            missing.append({"index": idx, "video_path": key[0], "start": key[1]})
            continue
        labels.append(labels_by_key[key])
    if missing:
        raise ValueError(f"Missing {len(missing)} labels in {labels_json}; first missing: {missing[0]}")
    return torch.tensor(labels, dtype=torch.long)


def filter_examples(
    features: torch.Tensor,
    labels: torch.Tensor,
    examples: Sequence[dict],
    filter_key: str | None,
    filter_value: str | None,
) -> tuple[torch.Tensor, torch.Tensor, List[dict]]:
    if filter_key is None and filter_value is None:
        return features, labels, list(examples)
    if not filter_key or filter_value is None:
        raise ValueError("--filter-key and --filter-value must be provided together.")

    keep = [idx for idx, example in enumerate(examples) if str(example.get(filter_key)) == filter_value]
    if not keep:
        raise ValueError(f"No examples match {filter_key}={filter_value}")
    keep_tensor = torch.tensor(keep, dtype=torch.long)
    return features[keep_tensor], labels[keep_tensor], [examples[idx] for idx in keep]


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    kind: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    class_weighting: str,
    device: str,
) -> tuple[nn.Module, List[Dict[str, float]]]:
    model = build_probe(kind, x.shape[1], int(y.max().item()) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_x = x[train_idx].to(device)
    train_y = y[train_idx].to(device)
    val_x = x[val_idx].to(device)
    val_y = y[val_idx].to(device)
    class_weights = None
    if class_weighting == "balanced":
        num_classes = int(y.max().item()) + 1
        counts = torch.bincount(train_y.cpu(), minlength=num_classes).float().clamp_min(1)
        class_weights = (counts.sum() / (num_classes * counts)).to(device)
    generator = torch.Generator().manual_seed(0)
    history = []

    for epoch in tqdm(range(epochs), desc=f"train {kind}"):
        perm = torch.randperm(train_x.shape[0], generator=generator)
        train_losses = []
        for start in range(0, train_x.shape[0], batch_size):
            batch_idx = perm[start:start + batch_size]
            logits = model(train_x[batch_idx])
            loss = F.cross_entropy(logits, train_y[batch_idx], weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        with torch.inference_mode():
            val_logits = model(val_x)
            val_loss = F.cross_entropy(val_logits, val_y, weight=class_weights)
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred == val_y).float().mean()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(val_loss.detach().cpu().item()),
                "val_accuracy": float(val_acc.detach().cpu().item()),
            }
        )
    return model, history


@torch.inference_mode()
def model_logits(model: nn.Module, x: torch.Tensor, indices: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    logits = []
    for start in range(0, len(indices), 1024):
        batch_idx = indices[start:start + 1024]
        logits.append(model(x[batch_idx].to(device)).cpu())
    return torch.cat(logits, dim=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    if args.feature_key is not None:
        features = cache["features_by_representation"][args.feature_key].float()
    else:
        features = cache["features"].float()
    examples = cache["examples"]
    deltas = [int(delta) for delta in cache["metadata"]["deltas"]]
    if args.labels_json is not None:
        if args.heuristic is None:
            raise ValueError("--heuristic is required when --labels-json is provided.")
        labels = labels_from_json(examples, args.labels_json, args.heuristic, deltas)
    else:
        labels = cache["labels"].long()

    features, labels, examples = filter_examples(features, labels, examples, args.filter_key, args.filter_value)
    metadata_fields = parse_metadata_fields(args.metadata_fields)

    train_idx, val_idx = split_indices(examples, args.val_fraction)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Empty train or validation split.")

    features = normalize(features[train_idx], features)
    metadata_features = build_metadata_features(examples, metadata_fields)
    metadata_features = normalize(metadata_features[train_idx], metadata_features)

    feature_probe, feature_history = train_model(
        features,
        labels,
        train_idx,
        val_idx,
        args.probe,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.class_weighting,
        args.device,
    )
    metadata_probe, metadata_history = train_model(
        metadata_features,
        labels,
        train_idx,
        val_idx,
        "linear",
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.class_weighting,
        args.device,
    )

    train_examples = [examples[idx] for idx in train_idx.tolist()]
    val_examples = [examples[idx] for idx in val_idx.tolist()]
    val_y = labels[val_idx]
    train_y = labels[train_idx]

    metrics = {
        "metadata": {
            "features": args.features,
            "feature_key": args.feature_key,
            "labels_json": args.labels_json,
            "heuristic": args.heuristic or cache["metadata"].get("heuristic"),
            "filter_key": args.filter_key,
            "filter_value": args.filter_value,
            "num_examples": len(examples),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "deltas": deltas,
            "probe": args.probe,
            "class_weighting": args.class_weighting,
            "metadata_fields": metadata_fields,
        },
        "feature_probe": evaluate_predictions(
            val_y,
            model_logits(feature_probe, features, val_idx, args.device),
            deltas,
            examples,
            val_idx,
        ),
        "metadata_probe": evaluate_predictions(
            val_y,
            model_logits(metadata_probe, metadata_features, val_idx, args.device),
            deltas,
            examples,
            val_idx,
        ),
        "baselines": {
            "global_majority": evaluate_predictions(val_y, majority_predictions(train_y, val_y), deltas, examples, val_idx),
            "dataset_majority": evaluate_predictions(
                val_y,
                grouped_majority_predictions(train_examples, train_y, val_examples, val_y, "label_source_name"),
                deltas,
                examples,
                val_idx,
            ),
            "fps_bucket_majority": evaluate_predictions(
                val_y,
                grouped_majority_predictions(train_examples, train_y, val_examples, val_y, "fps_bucket"),
                deltas,
                examples,
                val_idx,
            ),
            "fixed_delta_1": evaluate_predictions(val_y, fixed_delta_predictions(1, deltas, val_y), deltas, examples, val_idx),
            "fixed_delta_16": evaluate_predictions(val_y, fixed_delta_predictions(16, deltas, val_y), deltas, examples, val_idx),
            "fixed_delta_32": evaluate_predictions(val_y, fixed_delta_predictions(32, deltas, val_y), deltas, examples, val_idx),
        },
        "training_history": {
            "feature_probe": feature_history,
            "metadata_probe": metadata_history,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
