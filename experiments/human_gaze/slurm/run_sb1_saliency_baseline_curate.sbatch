#!/usr/bin/env bash
#SBATCH --job-name=sb1-curate
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/user/latn/slurm/logs/sb1-curate-%j.out

# Gate SB1 curation: frozen analysis of the saliency top-16 against the saved
# GT1 actions; aborts without metrics.json if the implementation check fails.
set -euo pipefail
: "${SB1_WORKTREE:?}" "${SB1_COMMIT:?}" "${SB1_RUN_ID:?}" "${SB1_ARTIFACT_ROOT:?}" "${SB1_RESULT_DIR:?}"
cd "$SB1_WORKTREE"
[[ "$(git rev-parse HEAD)" == "$SB1_COMMIT" ]] || { echo "Commit mismatch" >&2; exit 3; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Dirty worktree" >&2; exit 4; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
uv run --no-project --python /home/stud/latn/miniconda3/envs/autogaze/bin/python python \
  scripts/human_gaze/sb1_saliency_baseline.py curate \
  --run-id "$SB1_RUN_ID" --saliency "$SB1_ARTIFACT_ROOT/saliency" --output "$SB1_RESULT_DIR"
