#!/usr/bin/env bash
#SBATCH --job-name=n1-native-cov
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/n1-native-cov-%j.out

# Gate N1: two native multi-scale AutoGaze passes (Native-16, Native-default)
# over the 663 STAViS fold-1 test clips. Inference only.
set -euo pipefail
: "${N1_WORKTREE:?Set N1_WORKTREE to the frozen clean checkout}"
: "${N1_COMMIT:?Set N1_COMMIT to the immutable pushed commit}"
: "${N1_RUN_ID:?Set N1_RUN_ID}"
: "${N1_ARTIFACT_ROOT:?Set N1_ARTIFACT_ROOT to the append-only artifact directory}"
cd "$N1_WORKTREE"
[[ "$(git rev-parse HEAD)" == "$N1_COMMIT" ]] || { echo "Commit mismatch" >&2; exit 3; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Dirty worktree" >&2; exit 4; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled OMP_NUM_THREADS=5
export HF_HOME=/storage/slurm/latn/data/huggingface_cache
mkdir -p "$N1_ARTIFACT_ROOT"
echo "run_id=$N1_RUN_ID commit=$N1_COMMIT job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)" \
  | tee -a "$N1_ARTIFACT_ROOT/job_log.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv | tee "$N1_ARTIFACT_ROOT/gpu.csv"
runner=(uv run --no-project --python /home/stud/latn/miniconda3/envs/autogaze/bin/python python)
for arm in native16 native_default; do
  "${runner[@]}" scripts/human_gaze/n1_native_coverage.py export \
    --arm "$arm" --run-id "$N1_RUN_ID" --output "$N1_ARTIFACT_ROOT/actions/$arm"
done
echo "end=$(date --iso-8601=seconds)" | tee -a "$N1_ARTIFACT_ROOT/job_log.txt"
