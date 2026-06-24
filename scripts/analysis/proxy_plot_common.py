# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared plotting helpers for HLVid proxy experiments."""

from __future__ import annotations

import json
from pathlib import Path


METHOD_LABELS = {
    ("mean_novelty", "pixel_mse", None): "Pixel novelty",
    ("mean_novelty", "edge_l1", None): "Edge novelty",
    ("ssim_distance", None, None): "SSIM distance",
    ("semantic_mean_novelty", None, "siglip"): "SigLIP semantic novelty",
    ("semantic_mean_novelty", None, "dinov2"): "DINOv2 semantic novelty",
    ("semantic_diversity", None, "dinov2"): "DINOv2 semantic diversity",
    ("siglip_qgain", None, None): "SigLIP qgain",
    ("question_relevance", None, None): "Question relevance",
    ("siglip_cosine", None, None): "SigLIP cosine",
    ("dinov2_cosine", None, None): "DINOv2 cosine",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def proxy_identity(row: dict) -> tuple[str, str | None, str | None]:
    metric = row["proxy_metric"]
    semantic_model = row.get("semantic_model")
    if metric in ("siglip_qgain", "question_relevance", "siglip_cosine", "dinov2_cosine"):
        semantic_model = None
    return metric, row.get("proxy_base_metric"), semantic_model


def proxy_label(row: dict) -> str:
    return METHOD_LABELS.get(proxy_identity(row), row.get("proxy_metric") or "Other")
