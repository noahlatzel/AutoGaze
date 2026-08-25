#!/usr/bin/env bash
#SBATCH --job-name=gaze-r4-plot
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r4-plot-%j.out

set -eo pipefail
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r4-analysis
export OMP_NUM_THREADS=2

python scripts/human_gaze/verify_r4_causal_difference.py \
  --output-json outputs/human_gaze/r4_causal_difference_diagnostics/verification.json \
  --max-signal-ratio 0.25

python scripts/human_gaze/plot_r4_causal_difference.py \
  --output-dir /home/stud/latn/master-thesis/wiki_gazing/assets \
  --summary-json outputs/human_gaze/r4_causal_difference_diagnostics/summary.json
