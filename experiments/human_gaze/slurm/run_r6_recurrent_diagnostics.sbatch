#!/usr/bin/env bash
#SBATCH --job-name=gaze-r6-state-diagnostic
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r6-state-diagnostic-%j.out

set -eo pipefail
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r6-analysis
export OMP_NUM_THREADS=5

python scripts/human_gaze/diagnose_recurrent_trajectory_state.py \
  --config experiments/human_gaze/configs/r6_recurrent_trajectory_state.yaml \
  --seeds 440826 440827 440828 \
  --batch-size 8 \
  --num-workers 4 \
  --output-json outputs/human_gaze/r6_recurrent_state_diagnostics/recurrent_trajectory.json \
  --output-plot /home/stud/latn/master-thesis/wiki_gazing/assets/r6_recurrent_state_state_and_persistence.png
