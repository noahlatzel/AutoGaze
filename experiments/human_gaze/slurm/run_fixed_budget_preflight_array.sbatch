#!/usr/bin/env bash
#SBATCH --job-name=gaze-k-smoke
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --array=0-1
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-k-smoke-%A_%a.out

set -eo pipefail

if (( SLURM_ARRAY_TASK_ID == 0 )); then
  budget=24
  report_budgets='[1,2,4,8,16,24]'
else
  budget=36
  report_budgets='[1,2,4,8,16,24,36]'
fi

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2c-fixed-budgets

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((62100 + SLURM_ARRAY_TASK_ID))"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage1 \
  task.exact_budget="$budget" \
  task.report_budgets="$report_budgets" \
  trainer.seed="$((940826 + SLURM_ARRAY_TASK_ID))" \
  trainer.exp_name="r2c_preflight_fixedk${budget}" \
  trainer.max_train_steps=1 \
  trainer.validate_at_start=false \
  trainer.save_at_end=false \
  trainer.skip_final_validation=true
