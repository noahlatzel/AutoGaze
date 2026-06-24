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

"""Train a frozen-feature delta policy with a contextual-bandit objective."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

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
    parser.add_argument("--features", required=True, help="Feature cache from extract_delta_probe_features.py.")
    parser.add_argument("--labels-json", required=True, help="Delta-label JSON containing all-action scores.")
    parser.add_argument("--heuristic", default="siglip", help="Score/label heuristic to optimize.")
    parser.add_argument("--feature-key", default="mean_vision", help="Feature key in features_by_representation.")
    parser.add_argument(
        "--metadata-fields",
        default="",
        help=(
            "Optional comma-separated runtime metadata fields to append. "
            "Choices: fps_bucket, effective_fps, frame_count, start."
        ),
    )
    parser.add_argument("--policy", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--output", required=True, help="Output JSON metrics path.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--skip-bonus",
        type=float,
        default=0.0025,
        help="Reward bonus multiplied by log(delta). Larger values prefer longer skips.",
    )
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    if not train_indices or not val_indices:
        raise ValueError("Empty train or validation split.")
    return torch.tensor(train_indices), torch.tensor(val_indices)


def normalize(train_x: torch.Tensor, all_x: torch.Tensor) -> torch.Tensor:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (all_x - mean) / std


def parse_metadata_fields(value: str) -> list[str]:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    valid = {"fps_bucket", "effective_fps", "frame_count", "start"}
    unknown = sorted(set(fields) - valid)
    if unknown:
        raise ValueError(f"Unknown metadata fields: {unknown}. Valid fields: {sorted(valid)}")
    return fields


def build_metadata_features(examples: Sequence[dict], fields: Sequence[str]) -> torch.Tensor:
    if not fields:
        return torch.empty((len(examples), 0), dtype=torch.float32)

    fps_buckets = sorted({example.get("fps_bucket") or "unknown" for example in examples})
    bucket_to_idx = {value: idx for idx, value in enumerate(fps_buckets)}
    rows = []
    for example in examples:
        row = []
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


def build_score_lookup(labels_json: str, heuristic: str, deltas: Sequence[int]) -> dict[tuple[str, int], dict]:
    data = json.loads(Path(labels_json).read_text())
    delta_set = {str(delta) for delta in deltas}
    lookup = {}
    for video in data["videos"]:
        for tile in video["tiles"]:
            scores = tile.get("scores", {}).get(heuristic, {})
            if not scores:
                continue
            filtered_scores = {int(delta): float(score) for delta, score in scores.items() if delta in delta_set}
            label = tile.get("labels", {}).get(heuristic)
            lookup[(video["video_path"], int(tile["start"]))] = {
                "scores": filtered_scores,
                "label": int(label) if label is not None else None,
            }
    return lookup


def attach_rewards(
    features: torch.Tensor,
    examples: Sequence[dict],
    labels_json: str,
    heuristic: str,
    deltas: Sequence[int],
    skip_bonus: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    lookup = build_score_lookup(labels_json, heuristic, deltas)
    delta_to_idx = {delta: idx for idx, delta in enumerate(deltas)}

    keep = []
    reward_rows = []
    valid_rows = []
    label_rows = []
    kept_examples = []
    for idx, example in enumerate(examples):
        key = (example["video_path"], int(example["start"]))
        item = lookup.get(key)
        if item is None:
            continue
        rewards = torch.full((len(deltas),), -1e9, dtype=torch.float32)
        valid = torch.zeros((len(deltas),), dtype=torch.bool)
        for delta, score in item["scores"].items():
            if delta not in delta_to_idx:
                continue
            action_idx = delta_to_idx[delta]
            rewards[action_idx] = -score + skip_bonus * math.log(delta)
            valid[action_idx] = True
        if not valid.any() or item["label"] not in delta_to_idx:
            continue
        keep.append(idx)
        reward_rows.append(rewards)
        valid_rows.append(valid)
        label_rows.append(delta_to_idx[item["label"]])
        kept_examples.append(example)

    if not keep:
        raise ValueError("No examples matched labels JSON.")
    keep_tensor = torch.tensor(keep, dtype=torch.long)
    return (
        features[keep_tensor],
        torch.stack(reward_rows),
        torch.stack(valid_rows),
        torch.tensor(label_rows, dtype=torch.long),
        kept_examples,
    )


def build_policy(input_dim: int, num_actions: int, kind: str) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_dim, num_actions)
    hidden_dim = max(64, input_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, num_actions),
    )


def masked_logits(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~valid, -1e9)


def train_bandit(
    x: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    train_idx: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[nn.Module, list[dict]]:
    model = build_policy(x.shape[1], rewards.shape[1], args.policy).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_x = x[train_idx].to(args.device)
    train_rewards = rewards[train_idx].to(args.device)
    train_valid = valid[train_idx].to(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    history = []

    for epoch in tqdm(range(args.epochs), desc="train bandit"):
        perm = torch.randperm(train_x.shape[0], generator=generator)
        losses = []
        expected_rewards = []
        entropies = []
        for start in range(0, train_x.shape[0], args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            logits = masked_logits(model(train_x[batch_idx]), train_valid[batch_idx])
            probs = logits.softmax(dim=-1)
            entropy = -(probs * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
            expected_reward = (probs * train_rewards[batch_idx]).sum(dim=-1).mean()
            loss = -expected_reward - args.entropy_bonus * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            expected_rewards.append(float(expected_reward.detach().cpu().item()))
            entropies.append(float(entropy.detach().cpu().item()))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "expected_reward": float(np.mean(expected_rewards)),
                "entropy": float(np.mean(entropies)),
            }
        )
    return model, history


def train_supervised(
    x: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    train_idx: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[nn.Module, list[dict]]:
    model = build_policy(x.shape[1], valid.shape[1], args.policy).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_x = x[train_idx].to(args.device)
    train_y = labels[train_idx].to(args.device)
    train_valid = valid[train_idx].to(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    history = []

    for epoch in tqdm(range(args.epochs), desc="train supervised"):
        perm = torch.randperm(train_x.shape[0], generator=generator)
        losses = []
        for start in range(0, train_x.shape[0], args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            logits = masked_logits(model(train_x[batch_idx]), train_valid[batch_idx])
            loss = F.cross_entropy(logits, train_y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return model, history


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor, valid: torch.Tensor, indices: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    preds = []
    for start in range(0, len(indices), 1024):
        batch_idx = indices[start : start + 1024]
        logits = masked_logits(model(x[batch_idx].to(device)), valid[batch_idx].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds)


def oracle_predictions(rewards: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return rewards[indices].argmax(dim=-1)


def fixed_predictions(delta: int, deltas: Sequence[int], valid: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    delta_to_idx = {value: idx for idx, value in enumerate(deltas)}
    preferred = delta_to_idx[delta]
    preds = []
    for row in valid[indices]:
        if row[preferred]:
            preds.append(preferred)
            continue
        valid_indices = row.nonzero(as_tuple=False).flatten().tolist()
        preds.append(max(valid_indices, key=lambda idx: deltas[idx]))
    return torch.tensor(preds, dtype=torch.long)


def evaluate(name: str, pred: torch.Tensor, labels: torch.Tensor, rewards: torch.Tensor, deltas: Sequence[int], indices: torch.Tensor) -> dict:
    idx = indices
    y = labels[idx]
    chosen_rewards = rewards[idx, pred]
    delta_values = torch.tensor(deltas, dtype=torch.float32)
    pred_delta = delta_values[pred]
    true_delta = delta_values[y]
    return {
        "name": name,
        "num_examples": int(len(idx)),
        "mean_reward": float(chosen_rewards.mean().item()),
        "label_accuracy": float((pred == y).float().mean().item()),
        "mean_delta": float(pred_delta.mean().item()),
        "mean_abs_delta_error": float((pred_delta - true_delta).abs().mean().item()),
        "prediction_counts": {str(deltas[k]): v for k, v in sorted(Counter(pred.tolist()).items())},
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    features = cache["features_by_representation"][args.feature_key].float()
    deltas = [int(delta) for delta in cache["metadata"]["deltas"]]
    examples = cache["examples"]

    features, rewards, valid, labels, examples = attach_rewards(
        features,
        examples,
        args.labels_json,
        args.heuristic,
        deltas,
        args.skip_bonus,
    )
    train_idx, val_idx = split_indices(examples, args.val_fraction)
    metadata_fields = parse_metadata_fields(args.metadata_fields)
    features = normalize(features[train_idx], features)
    metadata_features = build_metadata_features(examples, metadata_fields)
    if metadata_features.shape[1] > 0:
        metadata_features = normalize(metadata_features[train_idx], metadata_features)
        features = torch.cat([features, metadata_features], dim=1)

    bandit, bandit_history = train_bandit(features, rewards, valid, train_idx, args)
    supervised, supervised_history = train_supervised(features, labels, valid, train_idx, args)

    results = {
        "metadata": {
            "features": args.features,
            "feature_key": args.feature_key,
            "labels_json": args.labels_json,
            "heuristic": args.heuristic,
            "skip_bonus": args.skip_bonus,
            "entropy_bonus": args.entropy_bonus,
            "policy": args.policy,
            "metadata_fields": metadata_fields,
            "deltas": deltas,
            "num_examples": len(examples),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "objective": "maximize E_pi[-score + skip_bonus * log(delta)] from all-action logged scores",
        },
        "validation": [
            evaluate("bandit_policy", predict(bandit, features, valid, val_idx, args.device), labels, rewards, deltas, val_idx),
            evaluate(
                "supervised_ce_policy",
                predict(supervised, features, valid, val_idx, args.device),
                labels,
                rewards,
                deltas,
                val_idx,
            ),
            evaluate("oracle_best_reward", oracle_predictions(rewards, val_idx), labels, rewards, deltas, val_idx),
        ],
        "training_history": {
            "bandit": bandit_history,
            "supervised_ce": supervised_history,
        },
    }
    for delta in deltas:
        results["validation"].append(
            evaluate(
                f"fixed_delta_{delta}",
                fixed_predictions(delta, deltas, valid, val_idx),
                labels,
                rewards,
                deltas,
                val_idx,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
