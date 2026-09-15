#!/usr/bin/env bash
#SBATCH --job-name=gaze-k32-curate
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-k32-curate-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2c-k32-reproduction

export OMP_NUM_THREADS=2

python scripts/human_gaze/verify_k32_fixed_budget.py \
  --config experiments/human_gaze/configs/r2c_k32_fixed_budget_reproduction.yaml \
  --run-root outputs/human_gaze/grpo \
  --diagnostics-root outputs/human_gaze/diagnostics \
  --reference-root /storage/user/latn/worktrees/autogaze-r2c-fixed-budgets/outputs/human_gaze \
  --mode complete \
  --output-json outputs/human_gaze/r2c_k32_fixed_budget_reproduction/verification.json

python scripts/human_gaze/curate_k32_fixed_budget.py \
  --config experiments/human_gaze/configs/r2c_k32_fixed_budget_reproduction.yaml \
  --run-root outputs/human_gaze/grpo \
  --diagnostics-root outputs/human_gaze/diagnostics \
  --reference-root /storage/user/latn/worktrees/autogaze-r2c-fixed-budgets/outputs/human_gaze \
  --verification outputs/human_gaze/r2c_k32_fixed_budget_reproduction/verification.json \
  --output-dir outputs/human_gaze/r2c_k32_fixed_budget_reproduction/curated \
  --figure-dir outputs/human_gaze/r2c_k32_fixed_budget_reproduction/figures \
  --training-job-id "${TRAINING_JOB_ID:?TRAINING_JOB_ID is required}" \
  --diagnostics-job-id "${DIAGNOSTICS_JOB_ID:?DIAGNOSTICS_JOB_ID is required}" \
  --code-commit "${CODE_COMMIT:?CODE_COMMIT is required}"
