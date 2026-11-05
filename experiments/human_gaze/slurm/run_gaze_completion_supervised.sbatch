#!/usr/bin/env bash
#SBATCH --job-name=gaze-slcurate
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-slcurate-%A_%a.out
set -euo pipefail
: "${GAZE_COMPLETION_WORKTREE:?Supply frozen clean source checkout}"
: "${GAZE_COMPLETION_ROOT:?Supply new append-only artifact run directory}"
export GAZE_COMPLETION_RUN_ID="${GAZE_COMPLETION_RUN_ID:-$(basename "$GAZE_COMPLETION_ROOT")}"
cd "$GAZE_COMPLETION_WORKTREE"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled OMP_NUM_THREADS=5
export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export SUPERVISED_ANALYSIS_CONFIG=experiments/human_gaze/configs/supervised_k16_restart_analysis.yaml
seeds=(440826 440827 440828 440829 440831)
# The existing wrapper uses a Python executable; this adapter keeps uv as the interpreter launcher.
export AUTOGAZE_PYTHON="$PWD/scripts/human_gaze/uv_autogaze_python.sh"
bash scripts/human_gaze/export_supervised_k16_seed_bundle.sh \
  "${seeds[$SLURM_ARRAY_TASK_ID]}" \
  /storage/user/latn/artifacts/autogaze-supervised-k16/20260914-0158_supervised-k16-comparison_ec320a5 \
  "$GAZE_COMPLETION_ROOT/supervised_actions"
