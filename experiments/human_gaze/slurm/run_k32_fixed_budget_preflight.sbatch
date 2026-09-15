#!/usr/bin/env bash
#SBATCH --job-name=gaze-k32-preflight
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-k32-preflight-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2c-k32-reproduction

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((62300 + SLURM_JOB_ID % 1000))"

python scripts/human_gaze/evaluate_coverage_baselines.py \
  --config experiments/human_gaze/configs/b1_k32_fixed_budget_baselines_fold1.yaml

python scripts/human_gaze/evaluate_pretrained_exact16.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
  --evaluation-splits val \
  --exact-budget 32 \
  --report-budgets 1 2 4 8 16 24 32 \
  --output outputs/human_gaze/baselines/pretrained_fixedk32_val.json

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage1 \
  task.exact_budget=32 \
  task.report_budgets='[1,2,4,8,16,24,32]' \
  trainer.seed=940832 \
  trainer.exp_name=r2c_k32_reproduction_preflight \
  trainer.max_train_steps=1 \
  trainer.validate_at_start=false \
  trainer.save_at_end=false \
  trainer.skip_final_validation=true

python scripts/human_gaze/verify_k32_fixed_budget.py \
  --config experiments/human_gaze/configs/r2c_k32_fixed_budget_reproduction.yaml \
  --preflight-run outputs/human_gaze/grpo/r2c_k32_reproduction_preflight \
  --reference-root /storage/user/latn/worktrees/autogaze-r2c-fixed-budgets/outputs/human_gaze \
  --mode preflight \
  --output-json outputs/human_gaze/r2c_k32_fixed_budget_reproduction/preflight_verification.json
