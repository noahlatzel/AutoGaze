#!/usr/bin/env bash
#SBATCH --job-name=gaze-r5-plot
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r5-plot-%j.out

set -eo pipefail
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r5-analysis
export OMP_NUM_THREADS=2

python scripts/human_gaze/verify_r5_feature_transport.py \
  --output-json outputs/human_gaze/r5_feature_transport_diagnostics/verification.json \
  --max-absolute-logit-gate 1.0

python scripts/human_gaze/plot_r5_feature_transport.py \
  --transport-json outputs/human_gaze/r5_feature_transport_diagnostics/transport_prior.json \
  --output-dir /home/stud/latn/master-thesis/wiki_gazing/assets \
  --summary-json outputs/human_gaze/r5_feature_transport_diagnostics/summary.json
