#!/usr/bin/env bash
#SBATCH --job-name=gaze-var-plot-v2
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-var-plot-v2-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2d-variable-budget

export OMP_NUM_THREADS=1

python scripts/human_gaze/plot_variable_budget.py \
  --config experiments/human_gaze/configs/r2d_variable_budget_v2.yaml \
  --runs-root outputs/human_gaze/grpo \
  --diagnostics-root outputs/human_gaze/diagnostics \
  --output-dir /home/stud/latn/master-thesis/wiki_gazing/assets
