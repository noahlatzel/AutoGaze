#!/usr/bin/env bash
#SBATCH --job-name=f1-curate
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/f1-curate-%j.out
set -euo pipefail
: "${F1_WORKTREE:?}" "${F1_COMMIT:?}" "${F1_RUN_ID:?}" "${F1_ARTIFACT_ROOT:?}" "${F1_RESULT_DIR:?}"
cd "$F1_WORKTREE"
[[ "$(git rev-parse HEAD)" == "$F1_COMMIT" ]] || { echo "Commit mismatch" >&2; exit 3; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
uv run --no-project --python /home/stud/latn/miniconda3/envs/autogaze/bin/python python \
  scripts/human_gaze/curate_f1_selection_floor.py \
  --config experiments/human_gaze/configs/f1_hlvid_selection_floor_v1.yaml \
  --raw-root "$F1_ARTIFACT_ROOT" --run-id "$F1_RUN_ID" --expected-commit "$F1_COMMIT" \
  --output-dir "$F1_RESULT_DIR"
