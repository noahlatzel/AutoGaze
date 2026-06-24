#!/bin/bash
#SBATCH --job-name=hlvid-nvila-default
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/slurm-%j.out

set -eo pipefail

set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export HYDRA_FULL_ERROR=1

echo "pwd=$(pwd)"
echo "hostname=$(hostname)"
date
nvidia-smi

mkdir -p /storage/user/latn/slurm/logs
mkdir -p outputs/hlvid/nvila_default_autogaze

python scripts/runners/evaluate_hlvid_nvila.py \
    --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
    --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
    --autogaze-model-id /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze \
    --output outputs/hlvid/nvila_default_autogaze/results.jsonl \
    --summary-output outputs/hlvid/nvila_default_autogaze/summary.json \
    --resume \
    ${EXTRA_ARGS:-}
