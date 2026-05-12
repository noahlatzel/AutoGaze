#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

module load cuda/12.8.1

export HF_HOME="${HF_HOME:-/storage/slurm/latn/data/huggingface_cache}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

DATA_ROOT="/storage/slurm/latn/data/AutoGaze/AutoGaze-Training-Data"
OUT_DIR="outputs/delta_labels"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

COMMON_ARGS=(
  --split train
  --limit 1000
  --heuristics videomae_reconstruction
  --thresholds videomae_reconstruction=2.5
  --deltas 1,2,4,8,16,32
  --device cuda
  --dtype float16
)

run_dataset() {
  local name="$1"
  local root="$2"
  local output="${OUT_DIR}/exp0b_videomae_reconstruction_${name}_train_1000.json"
  local log="${LOG_DIR}/exp0b_videomae_reconstruction_${name}_train_1000.log"

  echo "[$(date --iso-8601=seconds)] Starting ${name}"
  python scripts/generate_delta_labels.py \
    --input-roots "${DATA_ROOT}/${root}" \
    "${COMMON_ARGS[@]}" \
    --output "${output}" \
    > "${log}" 2>&1
  echo "[$(date --iso-8601=seconds)] Finished ${name}: ${output}"
}

run_dataset "internvid" "InternVid_res448_250K"
run_dataset "100doh" "100DoH_res448_250K"
run_dataset "ego4d" "Ego4D_res448_250K"
run_dataset "scanning_sam" "scanning_SAM_res448_50K"
run_dataset "scanning_idl" "scanning_idl_res448_50K"

python scripts/combine_delta_label_files.py \
  --inputs \
    "${OUT_DIR}/exp0b_videomae_reconstruction_internvid_train_1000.json" \
    "${OUT_DIR}/exp0b_videomae_reconstruction_100doh_train_1000.json" \
    "${OUT_DIR}/exp0b_videomae_reconstruction_ego4d_train_1000.json" \
    "${OUT_DIR}/exp0b_videomae_reconstruction_scanning_sam_train_1000.json" \
    "${OUT_DIR}/exp0b_videomae_reconstruction_scanning_idl_train_1000.json" \
  --output "${OUT_DIR}/exp0b_videomae_reconstruction_balanced5k_train.json" \
  > "${LOG_DIR}/exp0b_videomae_reconstruction_balanced5k_combine.log" 2>&1

echo "[$(date --iso-8601=seconds)] Combined output: ${OUT_DIR}/exp0b_videomae_reconstruction_balanced5k_train.json"
