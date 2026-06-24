#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
set -u

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export HYDRA_FULL_ERROR=1

LABELS="outputs/delta_labels/exp0b_pixel_mse_t004_balanced25k_train.json"
FEATURES="outputs/delta_probe_features/autogaze_balanced25k_t004_representations.pt"
OUT_DIR="outputs/delta_probe_supervised_t004_balanced25k"
LOG_DIR="${OUT_DIR}/logs"
METADATA_FIELDS="effective_fps,frame_count,start"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" outputs/delta_probe_features

echo "[$(date --iso-8601=seconds)] Starting balanced25k supervised Pixel MSE t=0.04 probe run"
echo "labels=${LABELS}"
echo "features=${FEATURES}"
echo "out_dir=${OUT_DIR}"
echo "hostname=$(hostname)"
nvidia-smi || true

if [[ ! -s "${FEATURES}" ]]; then
  echo "[$(date --iso-8601=seconds)] Extracting frozen AutoGaze representations"
  python scripts/extract_delta_probe_features.py \
    --labels "${LABELS}" \
    --heuristic pixel_mse \
    --representations final_gaze,mean_vision \
    --device cuda \
    --dtype float16 \
    --output "${FEATURES}" \
    2>&1 | tee "${LOG_DIR}/extract_autogaze_balanced25k_t004.log"
else
  echo "[$(date --iso-8601=seconds)] Reusing existing feature cache ${FEATURES}"
fi

run_probe() {
  local name="$1"
  local feature_key="$2"
  local probe="$3"
  local class_weighting="$4"
  local output="${OUT_DIR}/${name}_metrics.json"
  local log="${LOG_DIR}/${name}.log"

  if [[ -s "${output}" ]]; then
    echo "[$(date --iso-8601=seconds)] Skipping existing ${output}"
    return 0
  fi

  echo "[$(date --iso-8601=seconds)] Training ${name}"
  python scripts/train_delta_probe.py \
    --features "${FEATURES}" \
    --feature-key "${feature_key}" \
    --labels-json "${LABELS}" \
    --heuristic pixel_mse \
    --metadata-fields "${METADATA_FIELDS}" \
    --probe "${probe}" \
    --class-weighting "${class_weighting}" \
    --epochs 50 \
    --batch-size 512 \
    --device cuda \
    --output "${output}" \
    2>&1 | tee "${log}"
}

run_probe "pixel_mse_t004_mean_vision_runtime_linear" "mean_vision" "linear" "none"
run_probe "pixel_mse_t004_mean_vision_runtime_mlp" "mean_vision" "mlp" "none"
run_probe "pixel_mse_t004_final_gaze_runtime_linear" "final_gaze" "linear" "none"
run_probe "pixel_mse_t004_mean_vision_runtime_linear_balanced" "mean_vision" "linear" "balanced"

python scripts/plot_delta_probe_losses.py \
  --inputs \
    "${OUT_DIR}/pixel_mse_t004_mean_vision_runtime_linear_metrics.json" \
    "${OUT_DIR}/pixel_mse_t004_mean_vision_runtime_mlp_metrics.json" \
    "${OUT_DIR}/pixel_mse_t004_final_gaze_runtime_linear_metrics.json" \
    "${OUT_DIR}/pixel_mse_t004_mean_vision_runtime_linear_balanced_metrics.json" \
  --output "${OUT_DIR}/pixel_mse_t004_balanced25k_loss_curves.png"

echo "[$(date --iso-8601=seconds)] Finished balanced25k supervised Pixel MSE t=0.04 probe run"
