#!/usr/bin/env bash
#SBATCH --job-name=gaze-k-pretrain
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-k-pretrain-%A_%a.out

set -eo pipefail

if (( SLURM_ARRAY_TASK_ID == 0 )); then
  budget=24
  report_budgets=(1 2 4 8 16 24)
else
  budget=36
  report_budgets=(1 2 4 8 16 24 36)
fi

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2c-fixed-budgets

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export OMP_NUM_THREADS=4

python scripts/human_gaze/evaluate_pretrained_exact16.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
  --evaluation-splits val \
  --exact-budget "$budget" \
  --report-budgets "${report_budgets[@]}" \
  --output "outputs/human_gaze/baselines/pretrained_fixedk${budget}_val.json"
