#!/usr/bin/env bash
#SBATCH --job-name=gaze-10k-plot
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-10k-plot-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2-fresh10k

export OMP_NUM_THREADS=1

python scripts/human_gaze/plot_fresh10k_validation.py \
  --config experiments/human_gaze/configs/r2_fresh10k_direct_coverage.yaml \
  --runs-root outputs/human_gaze/grpo \
  --output /home/stud/latn/master-thesis/wiki_gazing/assets/r2_fresh10k_validation_macro_k16.png
