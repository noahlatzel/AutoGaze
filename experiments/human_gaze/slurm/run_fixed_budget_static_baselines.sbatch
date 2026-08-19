#!/usr/bin/env bash
#SBATCH --job-name=gaze-k-baseline
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-k-baseline-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2c-fixed-budgets

export OMP_NUM_THREADS=4

python scripts/human_gaze/evaluate_coverage_baselines.py \
  --config experiments/human_gaze/configs/b1_fixed_budget_baselines_fold1.yaml
