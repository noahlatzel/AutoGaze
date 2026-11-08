#!/usr/bin/env bash
#SBATCH --job-name=sb1-saliency
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/sb1-saliency-%j.out

# Gate SB1: TASED-Net (DHF1K) saliency maps for the 663 STAViS fold-1 test
# clips x 16 frames; top-16 fine cells per frame. Inference only.
set -euo pipefail
: "${SB1_WORKTREE:?Set SB1_WORKTREE to the frozen clean checkout}"
: "${SB1_COMMIT:?Set SB1_COMMIT to the immutable pushed commit}"
: "${SB1_RUN_ID:?Set SB1_RUN_ID}"
: "${SB1_ARTIFACT_ROOT:?Set SB1_ARTIFACT_ROOT to the append-only artifact directory}"
cd "$SB1_WORKTREE"
[[ "$(git rev-parse HEAD)" == "$SB1_COMMIT" ]] || { echo "Commit mismatch" >&2; exit 3; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Dirty worktree" >&2; exit 4; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled OMP_NUM_THREADS=6
mkdir -p "$SB1_ARTIFACT_ROOT"
echo "run_id=$SB1_RUN_ID commit=$SB1_COMMIT job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)" \
  | tee -a "$SB1_ARTIFACT_ROOT/job_log.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv | tee "$SB1_ARTIFACT_ROOT/gpu.csv"
uv run --no-project --python /home/stud/latn/miniconda3/envs/autogaze/bin/python python \
  scripts/human_gaze/sb1_saliency_baseline.py export \
  --run-id "$SB1_RUN_ID" --output "$SB1_ARTIFACT_ROOT/saliency"
echo "end=$(date --iso-8601=seconds)" | tee -a "$SB1_ARTIFACT_ROOT/job_log.txt"
