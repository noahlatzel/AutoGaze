#!/usr/bin/env bash
#SBATCH --job-name=gaze-var-stability
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-var-stability-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2d-variable-budget

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((65000 + SLURM_JOB_ID % 500))"

python -m autogaze.train \
  --config-name av_gaze_stavis_variable_budget_grpo \
  trainer.seed=840826 \
  trainer.gaze_weights=outputs/human_gaze/grpo/r2b_fresh20k_stage3_base440826_seed540826/checkpoint_latest_gaze \
  trainer.exp_name=r2d_variable_stability_v2b_base440826_seed840826 \
  trainer.max_train_steps=500 \
  trainer.val_nsteps=50 \
  trainer.save_nsteps=500 \
  trainer.temp_schedule_args.exp.temp_start=0.3 \
  trainer.temp_schedule_args.exp.temp_end=0.3 \
  task.initial_token_cost=0.005 \
  task.dual_lr=2.5e-7 \
  task.dual_ema_decay=0.995 \
  task.max_token_cost=0.02

python scripts/human_gaze/diagnose_variable_budget.py \
  --config experiments/human_gaze/configs/r2d_variable_budget_v2.yaml \
  --checkpoint outputs/human_gaze/grpo/r2d_variable_stability_v2b_base440826_seed840826/checkpoint_latest_gaze \
  --output-json outputs/human_gaze/diagnostics/r2d_variable_stability_v2b_base440826_seed840826/variable_budget_val.json \
  --output-plot /home/stud/latn/master-thesis/wiki_gazing/assets/r2d_variable_stability_v2b_diagnostics.png \
  --batch-size 8 \
  --num-workers 4

python scripts/human_gaze/check_variable_budget_stability.py \
  --history outputs/human_gaze/grpo/r2d_variable_stability_v2b_base440826_seed840826/validation_metrics.jsonl \
  --diagnostics outputs/human_gaze/diagnostics/r2d_variable_stability_v2b_base440826_seed840826/variable_budget_val.json \
  --reference-fixed-k16 0.45216500477354366
