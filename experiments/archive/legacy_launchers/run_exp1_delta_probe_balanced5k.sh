#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

module load cuda/12.8.1

export HF_HOME="${HF_HOME:-/storage/slurm/latn/data/huggingface_cache}"

OUT_DIR="outputs/delta_probe_features"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

FEATURES="${OUT_DIR}/autogaze_balanced5k_train.pt"
PIXEL_LABELS="outputs/delta_labels/exp0b_pixel_mse_balanced5k_train.json"
VIDEOMAE_LABELS="outputs/delta_labels/exp0b_videomae_reconstruction_balanced5k_train.json"

echo "[$(date --iso-8601=seconds)] Starting Experiment 1 balanced5k"

if [[ ! -f "${FEATURES}" ]]; then
  echo "[$(date --iso-8601=seconds)] Extracting frozen AutoGaze features"
  python scripts/runners/extract_delta_probe_features.py \
    --labels "${PIXEL_LABELS}" \
    --heuristic pixel_mse \
    --device cuda \
    --dtype float16 \
    --output "${FEATURES}" \
    > "${LOG_DIR}/exp1_extract_autogaze_balanced5k.log" 2>&1
else
  echo "[$(date --iso-8601=seconds)] Reusing existing feature cache ${FEATURES}"
fi

echo "[$(date --iso-8601=seconds)] Training pixel-MSE probe"
python scripts/runners/train_delta_probe.py \
  --features "${FEATURES}" \
  --labels-json "${PIXEL_LABELS}" \
  --heuristic pixel_mse \
  --probe linear \
  --epochs 100 \
  --batch-size 512 \
  --device cuda \
  --output "${OUT_DIR}/exp1a_pixel_mse_linear_balanced5k_metrics.json" \
  > "${LOG_DIR}/exp1a_pixel_mse_linear_balanced5k.log" 2>&1

echo "[$(date --iso-8601=seconds)] Training VideoMAE-reconstruction probe"
python scripts/runners/train_delta_probe.py \
  --features "${FEATURES}" \
  --labels-json "${VIDEOMAE_LABELS}" \
  --heuristic videomae_reconstruction \
  --probe linear \
  --epochs 100 \
  --batch-size 512 \
  --device cuda \
  --output "${OUT_DIR}/exp1b_videomae_reconstruction_linear_balanced5k_metrics.json" \
  > "${LOG_DIR}/exp1b_videomae_reconstruction_linear_balanced5k.log" 2>&1

echo "[$(date --iso-8601=seconds)] Finished Experiment 1 balanced5k"
