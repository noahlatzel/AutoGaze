#!/usr/bin/env bash
#SBATCH --job-name=gaze-variable-k-v2
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-variable-k-v2-%A_%a.out

set -eo pipefail

base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
training_seed="$((740826 + SLURM_ARRAY_TASK_ID))"
continuation_seed="$((540826 + SLURM_ARRAY_TASK_ID))"
initial_run="r2b_fresh20k_stage3_base${base_seed}_seed${continuation_seed}"
run="r2d_variable_k16_stable_base${base_seed}_seed${training_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2d-variable-budget

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((64500 + SLURM_ARRAY_TASK_ID))"

python -m autogaze.train \
  --config-name av_gaze_stavis_variable_budget_grpo \
  trainer.seed="$training_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/${initial_run}/checkpoint_latest_gaze" \
  trainer.exp_name="$run" \
  trainer.temp_schedule_args.exp.temp_start=0.3 \
  trainer.temp_schedule_args.exp.temp_end=0.3 \
  task.initial_token_cost=0.005 \
  task.dual_lr=2.5e-7 \
  task.dual_ema_decay=0.995 \
  task.max_token_cost=0.02

python scripts/human_gaze/diagnose_variable_budget.py \
  --config experiments/human_gaze/configs/r2d_variable_budget_v2.yaml \
  --checkpoint "outputs/human_gaze/grpo/${run}/checkpoint_latest_gaze" \
  --output-json "outputs/human_gaze/diagnostics/${run}/variable_budget_val.json" \
  --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r2d_variable_v2_seed${base_seed}_diagnostics.png" \
  --batch-size 8 \
  --num-workers 4
