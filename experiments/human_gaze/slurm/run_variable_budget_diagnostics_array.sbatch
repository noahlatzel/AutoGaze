#!/usr/bin/env bash
#SBATCH --job-name=gaze-var-diagnose
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-var-diagnose-%A_%a.out

set -eo pipefail

base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
training_seed="$((640826 + SLURM_ARRAY_TASK_ID))"
run="r2d_variable_k16_base${base_seed}_seed${training_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2d-variable-budget

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4

python scripts/human_gaze/diagnose_variable_budget.py \
  --config experiments/human_gaze/configs/r2d_variable_budget.yaml \
  --checkpoint "outputs/human_gaze/grpo/${run}/checkpoint_latest_gaze" \
  --output-json "outputs/human_gaze/diagnostics/${run}/variable_budget_val.json" \
  --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r2d_variable_seed${base_seed}_diagnostics.png" \
  --batch-size 8 \
  --num-workers 4
