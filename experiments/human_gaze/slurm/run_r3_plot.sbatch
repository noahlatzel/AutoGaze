#!/usr/bin/env bash
#SBATCH --job-name=gaze-r3-plot
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r3-plot-%j.out

set -eo pipefail
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r3-analysis
export OMP_NUM_THREADS=2

python scripts/human_gaze/plot_r3_temporal_position.py \
  --output-dir /home/stud/latn/master-thesis/wiki_gazing/assets \
  --summary-json outputs/human_gaze/r3b_temporal_diagnostics/summary.json
