#!/usr/bin/env bash
#SBATCH --job-name=gaze-var-preflight
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-var-preflight-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2d-variable-budget

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((63000 + SLURM_JOB_ID % 2000))"

python -m autogaze.train \
  --config-name av_gaze_stavis_variable_budget_grpo \
  trainer.seed=940826 \
  trainer.gaze_weights=outputs/human_gaze/grpo/r2b_fresh20k_stage3_base440826_seed540826/checkpoint_latest_gaze \
  trainer.exp_name=r2d_variable_preflight_seed940826 \
  trainer.n_epochs=1 \
  trainer.max_train_steps=1 \
  trainer.validate_at_start=false \
  trainer.skip_final_validation=true \
  trainer.save_at_end=false
