#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

module load cuda/12.8.1

export HF_HOME="${HF_HOME:-/storage/slurm/latn/data/huggingface_cache}"

OUT_DIR="outputs/delta_probe_features"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

FEATURES="${OUT_DIR}/autogaze_balanced5k_representations.pt"
PIXEL_LABELS="outputs/delta_labels/exp0b_pixel_mse_balanced5k_train.json"
VIDEOMAE_LABELS="outputs/delta_labels/exp0b_videomae_reconstruction_balanced5k_train.json"
REPRESENTATIONS="final_gaze,mean_gaze,final_frame_mean_gaze,mean_vision"

echo "[$(date --iso-8601=seconds)] Starting Experiment 1C representation ablation"

if [[ ! -f "${FEATURES}" ]]; then
  python scripts/runners/extract_delta_probe_features.py \
    --labels "${PIXEL_LABELS}" \
    --heuristic pixel_mse \
    --device cuda \
    --dtype float16 \
    --representations "${REPRESENTATIONS}" \
    --output "${FEATURES}" \
    > "${LOG_DIR}/exp1c_extract_representations_balanced5k.log" 2>&1
fi

for representation in final_gaze mean_gaze final_frame_mean_gaze mean_vision; do
  echo "[$(date --iso-8601=seconds)] Training pixel-MSE probe for ${representation}"
  python scripts/runners/train_delta_probe.py \
    --features "${FEATURES}" \
    --feature-key "${representation}" \
    --labels-json "${PIXEL_LABELS}" \
    --heuristic pixel_mse \
    --probe linear \
    --epochs 100 \
    --batch-size 512 \
    --device cuda \
    --output "${OUT_DIR}/exp1c_pixel_mse_${representation}_linear_balanced5k_metrics.json" \
    > "${LOG_DIR}/exp1c_pixel_mse_${representation}_linear_balanced5k.log" 2>&1

  echo "[$(date --iso-8601=seconds)] Training VideoMAE probe for ${representation}"
  python scripts/runners/train_delta_probe.py \
    --features "${FEATURES}" \
    --feature-key "${representation}" \
    --labels-json "${VIDEOMAE_LABELS}" \
    --heuristic videomae_reconstruction \
    --probe linear \
    --epochs 100 \
    --batch-size 512 \
    --device cuda \
    --output "${OUT_DIR}/exp1c_videomae_reconstruction_${representation}_linear_balanced5k_metrics.json" \
    > "${LOG_DIR}/exp1c_videomae_reconstruction_${representation}_linear_balanced5k.log" 2>&1
done

echo "[$(date --iso-8601=seconds)] Finished Experiment 1C representation ablation"
