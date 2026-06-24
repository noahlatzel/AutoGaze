#!/bin/bash
#SBATCH --job-name=pmse-labels
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --nodelist=node19,node20,node21
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/storage/user/latn/slurm/logs/slurm-%A_%a.out

set -euo pipefail

set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
set -u

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export HYDRA_FULL_ERROR=1

DATA_ROOT="/storage/slurm/latn/data/AutoGaze/AutoGaze-Training-Data"
OUT_DIR="outputs/delta_labels"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" /storage/user/latn/slurm/logs

DATASET_NAMES=(
  "internvid"
  "100doh"
  "ego4d"
  "scanning_sam"
  "scanning_idl"
)

DATASET_ROOTS=(
  "InternVid_res448_250K"
  "100DoH_res448_250K"
  "Ego4D_res448_250K"
  "scanning_SAM_res448_50K"
  "scanning_idl_res448_50K"
)

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
name="${DATASET_NAMES[$idx]}"
root="${DATASET_ROOTS[$idx]}"
output="${OUT_DIR}/exp0b_pixel_mse_t004_${name}_train_5000.json"

echo "pwd=$(pwd)"
echo "hostname=$(hostname)"
echo "dataset=${name}"
echo "root=${DATA_ROOT}/${root}"
echo "output=${output}"
date

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
  2>&1 | tee "${LOG_DIR}/exp0b_pixel_mse_t004_${name}_train_5000.log"
