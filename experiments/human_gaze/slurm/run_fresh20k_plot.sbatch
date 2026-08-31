#!/usr/bin/env bash
#SBATCH --job-name=gaze-20k-plot
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-20k-plot-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2b-fresh20k

export OMP_NUM_THREADS=1

python scripts/human_gaze/plot_fresh20k_analysis.py \
  --config experiments/human_gaze/configs/r2b_fresh20k_continuation.yaml \
  --runs-root outputs/human_gaze/grpo \
  --diagnostics-root outputs/human_gaze/diagnostics \
  --output-dir /home/stud/latn/master-thesis/wiki_gazing/assets
