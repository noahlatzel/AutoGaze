#!/usr/bin/env bash
#SBATCH --job-name=gaze-hlvid-allk
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --nodelist=node21
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-hlvid-allk-%A_%a.out

set -euo pipefail

: "${AUTOGAZE_WORKTREE:=/home/stud/latn/.codex/worktrees/3f12/master-thesis/AutoGaze}"
: "${AUTOGAZE_COMMIT:?Submit with AUTOGAZE_COMMIT set to the published implementation commit}"
: "${CHECKPOINT_ROOT:=/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/grpo}"
: "${EVAL_BUDGETS:=}"

base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
continuation_seed="$((540826 + SLURM_ARRAY_TASK_ID))"
if [[ -n "$EVAL_BUDGETS" ]]; then
  read -r -a budgets <<< "$EVAL_BUDGETS"
elif (( SLURM_ARRAY_TASK_ID < 3 )); then
  budgets=(16 24 36)
else
  budgets=(16)
fi

module load cuda/12.8.1
set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u
cd "$AUTOGAZE_WORKTREE"

observed_commit="$(git rev-parse HEAD)"
if [[ "$observed_commit" != "$AUTOGAZE_COMMIT" ]]; then
  echo "Commit mismatch: $observed_commit != $AUTOGAZE_COMMIT" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked AutoGaze worktree is dirty" >&2
  git status --short >&2
  exit 4
fi

preflight="experiments/human_gaze/results/r2e_hlvid_nvila_fixed_budget_all_seeds/preflight.json"
python - "$preflight" "$AUTOGAZE_COMMIT" <<'PY_CHECK'
import hashlib
import json
import pathlib
import subprocess
import sys

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

record = json.load(open(sys.argv[1]))
assert record["status"] == "pass", record["failures"]
submitted = sys.argv[2]
subprocess.run(
    ["git", "merge-base", "--is-ancestor", record["code_commit"], submitted],
    check=True,
)
assert record["code_dirty"] is False
assert record["config_sha256"] == digest(
    pathlib.Path("experiments/human_gaze/configs/r2e_hlvid_nvila_fixed_budget_all_seeds.yaml")
)
for path, expected in record["implementation_files"].items():
    assert digest(pathlib.Path(path)) == expected, path
assert record["verified_protocol"]["num_video_frames"] == 128
assert record["verified_protocol"]["num_video_frames_thumbnail"] == 64
assert record["verified_protocol"]["max_tiles_video"] == 48
assert record["dataset"]["human_gaze_protected_test_accessed"] is False
PY_CHECK

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

raw_root="outputs/human_gaze/hlvid_nvila/r2e_fixed_budget_all_seeds_mtv48"
for budget in "${budgets[@]}"; do
  case "$budget" in
    16)
      checkpoint="${CHECKPOINT_ROOT}/r2b_fresh20k_stage3_base${base_seed}_seed${continuation_seed}/checkpoint_latest_gaze"
      ;;
    24|32|36)
      checkpoint="${CHECKPOINT_ROOT}/r2c_fixedk${budget}_stage3_base${base_seed}_seed${continuation_seed}/checkpoint_latest_gaze"
      ;;
    *)
      echo "Unsupported budget: $budget" >&2
      exit 5
      ;;
  esac
  if [[ ! -f "$checkpoint/model.safetensors" ]]; then
    echo "Missing handed-off checkpoint: $checkpoint" >&2
    exit 6
  fi
  python - "$preflight" "$base_seed" "$budget" "$checkpoint" <<'PY_CHECKPOINT'
import hashlib
import json
import pathlib
import sys

record = json.load(open(sys.argv[1]))
base_seed, budget, checkpoint = int(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
matches = [
    row for row in record["checkpoints"]
    if row["base_seed"] == base_seed and row["budget"] == budget
]
assert len(matches) == 1, (base_seed, budget, len(matches))
expected = matches[0]
assert checkpoint.resolve() == pathlib.Path(expected["path"]).resolve()
digest = hashlib.sha256()
with (checkpoint / "model.safetensors").open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == expected["model_safetensors_sha256"], expected
PY_CHECKPOINT
  run_dir="${raw_root}/fixedk${budget}_seed${base_seed}"
  mkdir -p "$run_dir"
  echo "Evaluating seed=$base_seed budget=$budget checkpoint=$checkpoint"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
  python scripts/human_gaze/evaluate_hlvid_nvila_fixed_budget.py \
    --exact-budget "$budget" \
    --base-seed "$base_seed" \
    --continuation-seed "$continuation_seed" \
    --policy-label "fixedk${budget}_seed${base_seed}_20k" \
    --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
    --expected-dataset-sha256 ed2a1a47603fd19f5d7fae0db4f9792369f5aff6548f35e0225c3b4bd35e112a \
    --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
    --autogaze-model-id "$checkpoint" \
    --num-video-frames 128 \
    --num-video-frames-thumbnail 64 \
    --max-tiles-video 48 \
    --tile-len 16 \
    --max-new-tokens 16 \
    --max-batch-size-autogaze 16 \
    --max-batch-size-siglip 32 \
    --torch-dtype bfloat16 \
    --fine-action-offset 69 \
    --actions-per-frame 265 \
    --resume \
    --output "$run_dir/results.jsonl" \
    --summary-output "$run_dir/summary.json"
done
