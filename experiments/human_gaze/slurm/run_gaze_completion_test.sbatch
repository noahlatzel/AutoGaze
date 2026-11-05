#!/usr/bin/env bash
#SBATCH --job-name=gaze-test20k
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-6%1
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-test20k-%A_%a.out
set -euo pipefail
: "${GAZE_COMPLETION_WORKTREE:?Supply frozen clean source checkout}"
: "${GAZE_COMPLETION_ROOT:?Supply new append-only artifact run directory}"
export GAZE_COMPLETION_RUN_ID="${GAZE_COMPLETION_RUN_ID:-$(basename "$GAZE_COMPLETION_ROOT")}"
cd "$GAZE_COMPLETION_WORKTREE"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled OMP_NUM_THREADS=5
export HF_HOME=/storage/slurm/latn/data/huggingface_cache
models=(rl440826 rl440827 rl440828 rl440829 rl440830 rl440831 pretrained)
uv run --no-project --python /home/stud/latn/miniconda3/envs/autogaze/bin/python python \
  scripts/human_gaze/complete_gaze_test.py export \
  --model "${models[$SLURM_ARRAY_TASK_ID]}" \
  --output "$GAZE_COMPLETION_ROOT/test_actions/${models[$SLURM_ARRAY_TASK_ID]}"
