#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

module load cuda/12.8.1

export HF_HOME="${HF_HOME:-/storage/slurm/latn/data/huggingface_cache}"

ROOT="/storage/slurm/latn/data/AutoGaze/AutoGaze-Training-Data"
OUTPUT_DIR="outputs/delta_labels"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

DATASETS=(
  "InternVid_res448_250K"
  "100DoH_res448_250K"
  "Ego4D_res448_250K"
  "scanning_SAM_res448_50K"
  "scanning_idl_res448_50K"
)

OUTPUTS=()
for DATASET in "${DATASETS[@]}"; do
  OUTPUT="${OUTPUT_DIR}/exp0b_siglip_${DATASET}_train_1000.json"
  OUTPUTS+=("${OUTPUT}")
  echo "[$(date --iso-8601=seconds)] Generating ${OUTPUT}"
  python scripts/generate_delta_labels.py \
    --input-roots "${ROOT}/${DATASET}" \
    --split train \
    --limit 1000 \
    --max-tiles-per-video 8 \
    --heuristics siglip \
    --thresholds siglip=0.0125 \
    --deltas 1,2,4,8,16,32 \
    --device cuda \
    --dtype float16 \
    --batch-size 64 \
    --output "${OUTPUT}" \
    2>&1 | tee "${LOG_DIR}/exp0b_siglip_${DATASET}_train_1000.log"
done

COMBINED="${OUTPUT_DIR}/exp0b_siglip_balanced5k_train.json"
python scripts/combine_delta_label_files.py \
  --inputs "${OUTPUTS[@]}" \
  --output "${COMBINED}" \
  2>&1 | tee "${LOG_DIR}/exp0b_siglip_balanced5k_combine.log"

echo "[$(date --iso-8601=seconds)] Wrote ${COMBINED}"
