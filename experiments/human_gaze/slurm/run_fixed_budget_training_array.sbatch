#!/usr/bin/env bash
#SBATCH --job-name=gaze-fixed-k
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-fixed-k-%A_%a.out

set -eo pipefail

budget_index="$((SLURM_ARRAY_TASK_ID / 3))"
seed_index="$((SLURM_ARRAY_TASK_ID % 3))"
if (( budget_index == 0 )); then
  budget=24
  report_budgets='[1,2,4,8,16,24]'
else
  budget=36
  report_budgets='[1,2,4,8,16,24,36]'
fi
base_seed="$((440826 + seed_index))"
continuation_seed="$((540826 + seed_index))"
stage1_run="r2c_fixedk${budget}_stage1_seed${base_seed}"
stage2_run="r2c_fixedk${budget}_stage2_base${base_seed}_seed${continuation_seed}"
stage3_run="r2c_fixedk${budget}_stage3_base${base_seed}_seed${continuation_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2c-fixed-budgets

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((62200 + SLURM_ARRAY_TASK_ID))"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage1 \
  task.exact_budget="$budget" \
  task.report_budgets="$report_budgets" \
  trainer.seed="$base_seed" \
  trainer.exp_name="$stage1_run"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage2 \
  task.exact_budget="$budget" \
  task.report_budgets="$report_budgets" \
  trainer.seed="$continuation_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/${stage1_run}/checkpoint_latest_gaze" \
  trainer.exp_name="$stage2_run"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh20k_stage3 \
  task.exact_budget="$budget" \
  task.report_budgets="$report_budgets" \
  trainer.seed="$continuation_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/${stage2_run}/checkpoint_latest_gaze" \
  trainer.resume="outputs/human_gaze/grpo/${stage2_run}" \
  trainer.exp_name="$stage3_run"
