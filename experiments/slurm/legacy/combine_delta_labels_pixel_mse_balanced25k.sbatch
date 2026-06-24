#!/bin/bash
#SBATCH --job-name=pmse-combine
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --nodelist=node19,node20,node21
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/slurm-%j.out

set -euo pipefail

set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
set -u

OUT_DIR="outputs/delta_labels"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}" /storage/user/latn/slurm/logs

echo "pwd=$(pwd)"
echo "hostname=$(hostname)"
date

python scripts/maintenance/combine_delta_label_files.py \
  --inputs \
    "${OUT_DIR}/exp0b_pixel_mse_t004_internvid_train_5000.json" \
    "${OUT_DIR}/exp0b_pixel_mse_t004_100doh_train_5000.json" \
    "${OUT_DIR}/exp0b_pixel_mse_t004_ego4d_train_5000.json" \
    "${OUT_DIR}/exp0b_pixel_mse_t004_scanning_sam_train_5000.json" \
    "${OUT_DIR}/exp0b_pixel_mse_t004_scanning_idl_train_5000.json" \
  --output "${OUT_DIR}/exp0b_pixel_mse_t004_balanced25k_train.json" \
  2>&1 | tee "${LOG_DIR}/exp0b_pixel_mse_t004_balanced25k_combine.log"
