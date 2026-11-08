#!/usr/bin/env bash
#SBATCH --job-name=n1-curate
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/n1-curate-%j.out

# Gate N1 curation: pixel-union coverage for native and saved GT1 fine-only
# actions; aborts without metrics.json if the frozen implementation check fails.
set -euo pipefail
: "${N1_WORKTREE:?}" "${N1_COMMIT:?}" "${N1_RUN_ID:?}" "${N1_ARTIFACT_ROOT:?}" "${N1_RESULT_DIR:?}"
cd "$N1_WORKTREE"
[[ "$(git rev-parse HEAD)" == "$N1_COMMIT" ]] || { echo "Commit mismatch" >&2; exit 3; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Dirty worktree" >&2; exit 4; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8
uv run --no-project --python /home/stud/latn/miniconda3/envs/autogaze/bin/python python \
  scripts/human_gaze/n1_native_coverage.py curate --workers 8 \
  --run-id "$N1_RUN_ID" --actions "$N1_ARTIFACT_ROOT/actions" --output "$N1_RESULT_DIR"
