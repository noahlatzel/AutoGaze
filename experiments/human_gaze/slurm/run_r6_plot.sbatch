#!/usr/bin/env bash
#SBATCH --job-name=gaze-r6-plot
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r6-plot-%j.out

set -eo pipefail
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r6-analysis
export OMP_NUM_THREADS=2

python scripts/human_gaze/verify_r6_recurrent_state.py \
  --code-commit 7acfcadf8d4be1b899329ec2ac343f6a9d317548 \
  --output-json outputs/human_gaze/r6_recurrent_state_diagnostics/verification.json

python scripts/human_gaze/plot_r6_recurrent_state.py \
  --recurrent-json outputs/human_gaze/r6_recurrent_state_diagnostics/recurrent_trajectory.json \
  --control-history-json outputs/human_gaze/r3b_temporal_diagnostics/control_policy_history.json \
  --verification-json outputs/human_gaze/r6_recurrent_state_diagnostics/verification.json \
  --output-dir /home/stud/latn/master-thesis/wiki_gazing/assets \
  --summary-json outputs/human_gaze/r6_recurrent_state_diagnostics/summary.json
