# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic thumbnail metrics backed by frozen image/text encoders."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


SEMANTIC_MODELS = ("siglip", "dinov2")
SEMANTIC_PAIRWISE_METRICS = ("siglip_cosine", "dinov2_cosine", "siglip_qgain")
SEMANTIC_WINDOW_METRICS = ("semantic_mean_novelty", "semantic_diversity", "question_relevance")


class SiglipSemanticEncoder:
    """Frozen SigLIP image/text encoder for thumbnail-frame semantics."""

    def __init__(self, model_name_or_path: str, device: str, dtype: torch.dtype, batch_size: int):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device=device, dtype=dtype)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode_frames(self, frames: np.ndarray | Sequence[np.ndarray]) -> torch.Tensor:
        embeddings = []
        for batch in _frame_batches(frames, self.batch_size):
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
            frame_embeddings = self.model.get_image_features(pixel_values=pixel_values)
            embeddings.append(frame_embeddings.float())
        return F.normalize(torch.cat(embeddings, dim=0), dim=1).cpu()

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], padding=True, truncation=True, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        text_embedding = self.model.get_text_features(**inputs).float()[0]
        return F.normalize(text_embedding, dim=0).cpu()


class Dinov2SemanticEncoder:
    """Frozen DINOv2 image encoder for thumbnail-frame semantics."""

    def __init__(self, model_name_or_path: str, device: str, dtype: torch.dtype, batch_size: int):
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device=device, dtype=dtype)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode_frames(self, frames: np.ndarray | Sequence[np.ndarray]) -> torch.Tensor:
        embeddings = []
        for batch in _frame_batches(frames, self.batch_size):
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
            outputs = self.model(pixel_values=pixel_values)
            embeddings.append(_pooled_dinov2_embedding(outputs, self.model.config).float())
        return F.normalize(torch.cat(embeddings, dim=0), dim=1).cpu()


def cosine_distance_matrix(current: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Return pairwise cosine distances for normalized embedding matrices."""
    if current.ndim != 2 or candidate.ndim != 2:
        raise ValueError("current and candidate must both have shape (T,D)")
    if current.shape[1] != candidate.shape[1]:
        raise ValueError(f"Embedding dimensions differ: {current.shape[1]} vs {candidate.shape[1]}")
    return 1.0 - current @ candidate.T


def score_semantic_pairwise_metric(
    current: torch.Tensor,
    candidate: torch.Tensor,
    metric: str,
    *,
    question_embedding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score aligned current/candidate chunks with a semantic pairwise metric."""
    if current.shape[0] != candidate.shape[0]:
        raise ValueError(
            "Semantic pairwise metrics require equal chunk lengths, "
            f"got {current.shape[0]} and {candidate.shape[0]}"
        )

    if metric in ("siglip_cosine", "dinov2_cosine"):
        return cosine_distance_matrix(current, candidate).diagonal().mean()

    if metric == "siglip_qgain":
        if question_embedding is None:
            raise ValueError("siglip_qgain requires question_embedding")
        current_relevance = current @ question_embedding
        candidate_relevance = candidate @ question_embedding
        return (candidate_relevance - current_relevance).mean()

    raise ValueError(f"Unknown semantic pairwise metric: {metric}")


def score_semantic_window_metric(
    current: torch.Tensor,
    candidate: torch.Tensor,
    metric: str,
    *,
    question_embedding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score a candidate chunk using semantic embeddings."""
    if metric == "semantic_mean_novelty":
        distances = cosine_distance_matrix(current, candidate)
        return distances.min(dim=0).values.mean()

    if metric == "semantic_diversity":
        if candidate.shape[0] < 2:
            return candidate.new_tensor(0.0)
        distances = cosine_distance_matrix(candidate, candidate)
        row, col = torch.triu_indices(candidate.shape[0], candidate.shape[0], offset=1)
        return distances[row, col].mean()

    if metric == "question_relevance":
        if question_embedding is None:
            raise ValueError("question_relevance requires question_embedding")
        return (candidate @ question_embedding).max()

    raise ValueError(f"Unknown semantic window metric: {metric}")


def _frame_batches(frames: np.ndarray | Sequence[np.ndarray], batch_size: int) -> list[list[np.ndarray]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("Cannot encode an empty frame list")
    return [frame_list[start : start + batch_size] for start in range(0, len(frame_list), batch_size)]


def _pooled_dinov2_embedding(outputs, config) -> torch.Tensor:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    hidden = outputs.last_hidden_state
    num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
    patch_start = 1 + num_register_tokens
    if hidden.shape[1] > patch_start:
        return hidden[:, patch_start:].mean(dim=1)
    return hidden[:, 0]
