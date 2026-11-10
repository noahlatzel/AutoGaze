#!/usr/bin/env bash
#SBATCH --job-name=l2-curate
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/l2-curate-%j.out
set -euo pipefail
: "${L2_WORKTREE:?}" "${L2_COMMIT:?}" "${L2_RUN_ID:?}" "${L2_ARTIFACT_ROOT:?}" "${L2_RESULT_DIR:?}"
cd "$L2_WORKTREE"
[[ "$(git rev-parse HEAD)" == "$L2_COMMIT" ]] || { echo "Commit mismatch" >&2; exit 3; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
/home/stud/latn/miniconda3/envs/autogaze/bin/python scripts/human_gaze/curate_l2_consistent_layout.py \
  --config experiments/human_gaze/configs/l2_hlvid_consistent_layout_v1.yaml \
  --raw-root "$L2_ARTIFACT_ROOT" --run-id "$L2_RUN_ID" --expected-commit "$L2_COMMIT" \
  --output-dir "$L2_RESULT_DIR"
