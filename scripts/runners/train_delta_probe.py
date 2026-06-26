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
    parser.add_argument("--features", required=True, help="Train feature cache produced by extract_delta_probe_features.py.")
    parser.add_argument(
        "--val-features",
        default=None,
        help="Optional validation feature cache. When set, training uses all train-cache examples and evaluates on all val-cache examples.",
    )
    parser.add_argument("--output", required=True, help="Output JSON metrics path.")
    parser.add_argument(
        "--feature-key",
        default=None,
        help="Representation key when the cache contains features_by_representation.",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="Optional train label JSON used to override labels in the train feature cache.",
    )
    parser.add_argument(
        "--val-labels-json",
        default=None,
        help="Optional validation label JSON used to override labels in --val-features.",
    )
    parser.add_argument(
        "--label-metric",
        default=None,
        help="Proxy metric label to train on. Inferred from single-metric label JSONs or feature cache metadata when omitted.",
    )
    parser.add_argument(
        "--heuristic",
        default=None,
        help="Deprecated alias for --label-metric.",
    )
    parser.add_argument("--probe", choices=["linear", "mlp"], default="linear", help="Probe architecture.")
    parser.add_argument(
        "--input-mode",
        choices=["features", "metadata", "concat"],
        default="features",
        help="Input representation for the trained primary probe.",
    )
    parser.add_argument(
        "--architecture",
        choices=["classifier", "candidate_scorer"],
        default="classifier",
        help="Delta prediction architecture.",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "ce_ordinal"],
        default="ce",
        help="Training objective. ce_ordinal adds SmoothL1 on expected log2(delta).",
    )
    parser.add_argument("--ordinal-alpha", type=float, default=0.0, help="Weight for the ordinal auxiliary loss.")
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


class CandidateDeltaScorer(nn.Module):
    """Score each candidate delta from shared input plus an explicit delta encoding."""

    def __init__(self, kind: str, input_dim: int, deltas: Sequence[int]) -> None:
        super().__init__()
        delta_values = torch.tensor(deltas, dtype=torch.float32)
        log_delta = torch.log2(delta_values)
        log_delta = (log_delta - log_delta.mean()) / log_delta.std().clamp_min(1e-6)
        self.register_buffer("delta_features", torch.stack([delta_values / delta_values.max(), log_delta], dim=-1))
        scorer_input_dim = input_dim + self.delta_features.shape[1]
        if kind == "linear":
            self.scorer = nn.Linear(scorer_input_dim, 1)
        else:
            hidden_dim = input_dim
            self.scorer = nn.Sequential(
                nn.Linear(scorer_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        num_deltas = self.delta_features.shape[0]
        x_expanded = x[:, None, :].expand(batch_size, num_deltas, x.shape[-1])
        delta_expanded = self.delta_features[None, :, :].expand(batch_size, num_deltas, self.delta_features.shape[-1])
        scores = self.scorer(torch.cat([x_expanded, delta_expanded], dim=-1)).squeeze(-1)
        return scores


def build_delta_model(architecture: str, kind: str, input_dim: int, deltas: Sequence[int]) -> nn.Module:
    if architecture == "classifier":
        return build_probe(kind, input_dim, len(deltas))
    return CandidateDeltaScorer(kind, input_dim, deltas)


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


def build_primary_features(features: torch.Tensor, metadata_features: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "features":
        return features
    if input_mode == "metadata":
        return metadata_features
    if input_mode == "concat":
        return torch.cat([features, metadata_features], dim=-1)
    raise ValueError(f"Unknown input mode: {input_mode}")


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


def infer_label_metric_from_data(label_data: dict, requested: str | None) -> str:
    if requested:
        return requested
    metrics = label_data.get("metadata", {}).get("proxy_metrics")
    if metrics and len(metrics) == 1:
        return str(metrics[0])
    heuristics = label_data.get("metadata", {}).get("heuristics")
    if heuristics and len(heuristics) == 1:
        return str(heuristics[0])
    raise ValueError("--label-metric is required when the label JSON contains multiple or no metrics.")


def infer_label_metric_from_cache(cache: dict, requested: str | None) -> str | None:
    if requested:
        return requested
    metadata = cache.get("metadata", {})
    return metadata.get("label_metric") or metadata.get("heuristic")


def labels_from_json(examples: Sequence[dict], labels_json: str, label_metric: str, deltas: Sequence[int]) -> torch.Tensor:
    data = json.loads(Path(labels_json).read_text())
    label_metric = infer_label_metric_from_data(data, label_metric)
    delta_to_index = {int(delta): idx for idx, delta in enumerate(deltas)}
    labels_by_key = {}
    for video in data["videos"]:
        for tile in video["tiles"]:
            label = tile["labels"].get(label_metric)
            if label is None:
                continue
            label = int(label)
            if label not in delta_to_index:
                raise ValueError(f"Label delta {label} is not in expected deltas={list(deltas)}")
            labels_by_key[(video["video_path"], int(tile["start"]))] = delta_to_index[label]

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
    deltas: Sequence[int],
    architecture: str,
    loss_name: str,
    ordinal_alpha: float,
    device: str,
) -> tuple[nn.Module, List[Dict[str, float]]]:
    num_classes = len(deltas)
    model = build_delta_model(architecture, kind, x.shape[1], deltas).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_x = x[train_idx].to(device)
    train_y = y[train_idx].to(device)
    val_x = x[val_idx].to(device)
    val_y = y[val_idx].to(device)
    class_weights = None
    if class_weighting == "balanced":
        counts = torch.bincount(train_y.cpu(), minlength=num_classes).float().clamp_min(1)
        class_weights = (counts.sum() / (num_classes * counts)).to(device)
    log_deltas = torch.log2(torch.tensor(deltas, dtype=torch.float32, device=device))
    generator = torch.Generator().manual_seed(0)
    history = []

    for epoch in tqdm(range(epochs), desc=f"train {kind}"):
        perm = torch.randperm(train_x.shape[0], generator=generator)
        train_losses = []
        for start in range(0, train_x.shape[0], batch_size):
            batch_idx = perm[start:start + batch_size]
            logits = model(train_x[batch_idx])
            ce_loss = F.cross_entropy(logits, train_y[batch_idx], weight=class_weights)
            ordinal_loss = expected_log_delta_loss(logits, train_y[batch_idx], log_deltas) if loss_name == "ce_ordinal" else logits.new_tensor(0.0)
            loss = ce_loss + ordinal_alpha * ordinal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        with torch.inference_mode():
            val_logits = model(val_x)
            val_ce_loss = F.cross_entropy(val_logits, val_y, weight=class_weights)
            val_ordinal_loss = expected_log_delta_loss(val_logits, val_y, log_deltas) if loss_name == "ce_ordinal" else val_logits.new_tensor(0.0)
            val_loss = val_ce_loss + ordinal_alpha * val_ordinal_loss
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred == val_y).float().mean()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(val_loss.detach().cpu().item()),
                "val_ce_loss": float(val_ce_loss.detach().cpu().item()),
                "val_ordinal_loss": float(val_ordinal_loss.detach().cpu().item()),
                "val_accuracy": float(val_acc.detach().cpu().item()),
            }
        )
    return model, history


def expected_log_delta_loss(logits: torch.Tensor, y: torch.Tensor, log_deltas: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    expected_log_delta = (probs * log_deltas[None, :]).sum(dim=-1)
    target_log_delta = log_deltas[y]
    return F.smooth_l1_loss(expected_log_delta, target_log_delta)


@torch.inference_mode()
def model_logits(model: nn.Module, x: torch.Tensor, indices: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    logits = []
    for start in range(0, len(indices), 1024):
        batch_idx = indices[start:start + 1024]
        logits.append(model(x[batch_idx].to(device)).cpu())
    return torch.cat(logits, dim=0)


def select_features(cache: dict, feature_key: str | None) -> torch.Tensor:
    if feature_key is not None:
        return cache["features_by_representation"][feature_key].float()
    return cache["features"].float()


def load_split(
    features_path: str,
    labels_json: str | None,
    requested_label_metric: str | None,
    feature_key: str | None,
    expected_deltas: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, List[dict], List[int], str | None, dict]:
    cache = torch.load(features_path, map_location="cpu", weights_only=False)
    features = select_features(cache, feature_key)
    examples = list(cache["examples"])
    deltas = [int(delta) for delta in cache["metadata"]["deltas"]]
    if expected_deltas is not None and deltas != list(expected_deltas):
        raise ValueError(f"Delta mismatch for {features_path}: {deltas} != {list(expected_deltas)}")

    label_metric = infer_label_metric_from_cache(cache, requested_label_metric)
    if labels_json is not None:
        label_data = json.loads(Path(labels_json).read_text())
        label_metric = infer_label_metric_from_data(label_data, requested_label_metric or label_metric)
        labels = labels_from_json(examples, labels_json, label_metric, deltas)
    else:
        labels = cache["labels"].long()
        if requested_label_metric and label_metric and requested_label_metric != label_metric:
            raise ValueError(
                f"Feature cache {features_path} contains labels for {label_metric}, "
                f"but --label-metric requested {requested_label_metric}. Pass --labels-json to override."
            )
    return features, labels, examples, deltas, label_metric, cache


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    requested_label_metric = args.label_metric or args.heuristic
    train_features, train_labels, train_examples, deltas, label_metric, train_cache = load_split(
        args.features,
        args.labels_json,
        requested_label_metric,
        args.feature_key,
    )
    train_features, train_labels, train_examples = filter_examples(
        train_features,
        train_labels,
        train_examples,
        args.filter_key,
        args.filter_value,
    )

    if args.val_features is not None:
        val_features, val_labels, val_examples, _, val_label_metric, _ = load_split(
            args.val_features,
            args.val_labels_json,
            requested_label_metric or label_metric,
            args.feature_key,
            expected_deltas=deltas,
        )
        val_features, val_labels, val_examples = filter_examples(
            val_features,
            val_labels,
            val_examples,
            args.filter_key,
            args.filter_value,
        )
        if train_features.shape[1:] != val_features.shape[1:]:
            raise ValueError(f"Feature shape mismatch: train {tuple(train_features.shape)} vs val {tuple(val_features.shape)}")
        features = torch.cat([train_features, val_features], dim=0)
        labels = torch.cat([train_labels, val_labels], dim=0)
        examples = train_examples + val_examples
        train_idx = torch.arange(len(train_examples), dtype=torch.long)
        val_idx = torch.arange(len(train_examples), len(examples), dtype=torch.long)
        split_mode = "explicit"
        label_metric = requested_label_metric or label_metric or val_label_metric
    else:
        features = train_features
        labels = train_labels
        examples = train_examples
        train_idx, val_idx = split_indices(examples, args.val_fraction)
        split_mode = "hash_fraction"
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("Empty train or validation split.")

    metadata_fields = parse_metadata_fields(args.metadata_fields)
    num_classes = len(deltas)
    features = normalize(features[train_idx], features)
    metadata_features = build_metadata_features(examples, metadata_fields)
    metadata_features = normalize(metadata_features[train_idx], metadata_features)

    primary_features = build_primary_features(features, metadata_features, args.input_mode)

    primary_probe, primary_history = train_model(
        primary_features,
        labels,
        train_idx,
        val_idx,
        args.probe,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.class_weighting,
        deltas,
        args.architecture,
        args.loss,
        args.ordinal_alpha,
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
        deltas,
        "classifier",
        "ce",
        0.0,
        args.device,
    )

    train_examples = [examples[idx] for idx in train_idx.tolist()]
    val_examples = [examples[idx] for idx in val_idx.tolist()]
    val_y = labels[val_idx]
    train_y = labels[train_idx]

    metrics = {
        "metadata": {
            "features": args.features,
            "val_features": args.val_features,
            "feature_key": args.feature_key,
            "labels_json": args.labels_json,
            "val_labels_json": args.val_labels_json,
            "label_metric": label_metric,
            "heuristic": label_metric,
            "filter_key": args.filter_key,
            "filter_value": args.filter_value,
            "num_examples": len(examples),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "deltas": deltas,
            "split_mode": split_mode,
            "train_cache_label_metric": infer_label_metric_from_cache(train_cache, None),
            "probe": args.probe,
            "input_mode": args.input_mode,
            "architecture": args.architecture,
            "loss": args.loss,
            "ordinal_alpha": args.ordinal_alpha,
            "class_weighting": args.class_weighting,
            "metadata_fields": metadata_fields,
        },
        "primary_probe": evaluate_predictions(
            val_y,
            model_logits(primary_probe, primary_features, val_idx, args.device),
            deltas,
            examples,
            val_idx,
        ),
        "feature_probe": evaluate_predictions(
            val_y,
            model_logits(primary_probe, primary_features, val_idx, args.device),
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
            "primary_probe": primary_history,
            "feature_probe": primary_history,
            "metadata_probe": metadata_history,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
