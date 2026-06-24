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

DATA_ROOT="${DATA_ROOT:-/storage/slurm/latn/data/AutoGaze/AutoGaze-Training-Data}"
OUT_DIR="${OUT_DIR:-outputs/delta_labels}"
LOG_DIR="${OUT_DIR}/logs"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

declare -a DATASET_NAMES=(
  "internvid"
  "100doh"
  "ego4d"
  "scanning_sam"
  "scanning_idl"
)

declare -a DATASET_ROOTS=(
  "InternVid_res448_250K"
  "100DoH_res448_250K"
  "Ego4D_res448_250K"
  "scanning_SAM_res448_50K"
  "scanning_idl_res448_50K"
)

run_dataset() {
  local idx="$1"
  local name="${DATASET_NAMES[$idx]}"
  local root="${DATASET_ROOTS[$idx]}"
  local output="${OUT_DIR}/exp0b_pixel_mse_t004_${name}_train_5000.json"
  local log="${LOG_DIR}/exp0b_pixel_mse_t004_${name}_train_5000.log"

  if [[ -s "${output}" ]]; then
    echo "[skip] ${name}: ${output} already exists"
    return 0
  fi

  {
    echo "pwd=$(pwd)"
    echo "hostname=$(hostname)"
    echo "date=$(date)"
    echo "dataset=${name}"
    echo "root=${DATA_ROOT}/${root}"
    echo "output=${output}"
  } | tee "${log}"

  python scripts/generate_delta_labels.py \
    --input-roots "${DATA_ROOT}/${root}" \
    --split train \
    --limit 5000 \
    --max-tiles-per-video 8 \
    --heuristics pixel_mse \
    --thresholds pixel_mse=0.04 \
    --deltas 1,2,4,8,16,32 \
    --resize 224 \
    --full-video-decode-max-frames 256 \
    --device cpu \
    --dtype float32 \
    --output "${output}" \
    2>&1 | tee -a "${log}"
}

echo "Starting Pixel MSE balanced25k generation with PARALLEL_JOBS=${PARALLEL_JOBS}"
echo "date=$(date)"

declare -a pids=()
declare -a names=()

for idx in "${!DATASET_NAMES[@]}"; do
  run_dataset "${idx}" &
  pids+=("$!")
  names+=("${DATASET_NAMES[$idx]}")

  while (( ${#pids[@]} >= PARALLEL_JOBS )); do
    if wait "${pids[0]}"; then
      echo "[done] ${names[0]}"
    else
      echo "[failed] ${names[0]}" >&2
      exit 1
    fi
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
  done
done

for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[done] ${names[$idx]}"
  else
    echo "[failed] ${names[$idx]}" >&2
    exit 1
  fi
done

python scripts/combine_delta_label_files.py \
  --inputs \
    "${OUT_DIR}/exp0b_pixel_mse_t004_internvid_train_5000.json" \
    "${OUT_DIR}/exp0b_pixel_mse_t004_100doh_train_5000.json" \
    "${OUT_DIR}/exp0b_pixel_mse_t004_ego4d_train_5000.json" \
    "${OUT_DIR}/exp0b_pixel_mse_t004_scanning_sam_train_5000.json" \
    "${OUT_DIR}/exp0b_pixel_mse_t004_scanning_idl_train_5000.json" \
  --output "${OUT_DIR}/exp0b_pixel_mse_t004_balanced25k_train.json" \
  2>&1 | tee "${LOG_DIR}/exp0b_pixel_mse_t004_balanced25k_combine.log"

echo "Finished Pixel MSE balanced25k generation"
echo "date=$(date)"
