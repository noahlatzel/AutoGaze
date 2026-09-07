#!/usr/bin/env bash
#SBATCH --job-name=gaze-hlvid-k32
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --nodelist=node21
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-hlvid-k32-%A_%a.out

set -euo pipefail

: "${AUTOGAZE_WORKTREE:=/home/stud/latn/.codex/worktrees/3f12/master-thesis/AutoGaze}"
: "${AUTOGAZE_COMMIT:?Submit with AUTOGAZE_COMMIT set to the frozen evaluation commit}"
: "${LEGACY_CHECKPOINT_ROOT:=/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/grpo}"
: "${K32_CHECKPOINT_ROOT:=/storage/user/latn/worktrees/autogaze-r2c-k32-reproduction/outputs/human_gaze/grpo}"

base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
continuation_seed="$((540826 + SLURM_ARRAY_TASK_ID))"
case "$SLURM_ARRAY_TASK_ID" in
  0) budgets=(32) ;;
  1) budgets=(24 32) ;;
  2) budgets=(16 24 32) ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 5 ;;
esac

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

config="experiments/human_gaze/configs/r2e_hlvid_nvila_fixed_budget_k16_k24_k32.yaml"
preflight="experiments/human_gaze/results/r2e_hlvid_nvila_fixed_budget_all_seeds/preflight_k16_k24_k32.json"
legacy_preflight="experiments/human_gaze/results/r2e_hlvid_nvila_fixed_budget_all_seeds/preflight.json"
launcher="experiments/human_gaze/slurm/run_hlvid_nvila_k16_k24_k32_repair.sbatch"
python - "$preflight" "$config" "$legacy_preflight" "$launcher" "$SLURM_ARRAY_TASK_ID" <<'PY_CHECK'
import hashlib
import json
import pathlib
import sys

import yaml


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


preflight_path, config_path, legacy_path, launcher_path = map(pathlib.Path, sys.argv[1:5])
task_id = int(sys.argv[5])
record = json.loads(preflight_path.read_text())
config = yaml.safe_load(config_path.read_text())
legacy = json.loads(legacy_path.read_text())
assert record["status"] == "pass"
assert legacy["status"] == "pass"
assert digest(config_path) == record["config_sha256"]
assert digest(legacy_path) == record["existing_k16_k24_preflight"]["sha256"]
assert digest(launcher_path) == record["execution_files"][str(launcher_path)]
for path, expected in record["execution_files"].items():
    if path == str(launcher_path):
        continue
    assert digest(path) == expected, path

protocol = config["protocol"]
assert protocol["num_video_frames"] == 128
assert protocol["num_video_frames_thumbnail"] == 64
assert protocol["max_tiles_video"] == 48
assert protocol["tile_len"] == 16
assert protocol["max_new_tokens"] == 16
assert protocol["torch_dtype"] == "bfloat16"
assert protocol["generation"] == "greedy"
assert protocol["tile_policy"] == {
    "exact_decoder_actions_per_autogaze_frame": "budget",
    "allowed_action_ids": [69, 264],
    "unique_actions": True,
    "allow_eos": False,
}
dataset = pathlib.Path(config["dataset"]["root"]) / config["dataset"]["parquet"]
assert digest(dataset) == record["dataset"]["parquet_sha256"]
assert config["authorization"]["human_gaze_protected_test_accessed"] is False

verification_path = pathlib.Path(record["k32_handoff"]["source_verification"])
assert digest(verification_path) == record["k32_handoff"]["source_verification_sha256"]
verification = json.loads(verification_path.read_text())
assert verification["status"] == "verified" and verification["mode"] == "complete"
assert verification["matched_base_seeds"] == [440826, 440827, 440828]
assert verification["matched_continuation_seeds"] == [540826, 540827, 540828]
assert verification["fixed_cumulative_optimizer_updates"] == 20000
assert verification["exact_actions_per_frame"] == 32
assert verification["test_split_opened"] is False

plan = record["run_plan"][str(task_id)]
assert plan["base_seed"] == 440826 + task_id
assert plan["continuation_seed"] == 540826 + task_id
assert 36 not in plan["budgets"]
seed_row = next(row for row in config["seed_matrix"] if row["base_seed"] == plan["base_seed"])
k32 = next(row for row in record["k32_handoff"]["checkpoints"] if row["base_seed"] == plan["base_seed"])
assert pathlib.Path(seed_row["checkpoints"][32]).resolve() == pathlib.Path(k32["path"]).resolve()
for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
    expected = k32[{"model.safetensors": "model_safetensors_sha256", "config.json": "config_sha256", "preprocessor_config.json": "preprocessor_config_sha256"}[name]]
    assert digest(pathlib.Path(k32["path"]) / name) == expected

legacy_checkpoints = {
    (row["base_seed"], row["budget"]): row for row in legacy["checkpoints"]
}
checkpoint_root = pathlib.Path(config["models"]["checkpoint_root"])
for budget in plan["budgets"]:
    configured = pathlib.Path(seed_row["checkpoints"][budget])
    checkpoint = configured if configured.is_absolute() else checkpoint_root / configured
    if budget in (16, 24):
        expected = legacy_checkpoints[(plan["base_seed"], budget)]
        assert checkpoint.resolve() == pathlib.Path(expected["path"]).resolve()
        assert digest(checkpoint / "model.safetensors") == expected["model_safetensors_sha256"]
PY_CHECK

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

raw_root="outputs/human_gaze/hlvid_nvila/r2e_fixed_budget_all_seeds_mtv48"
for budget in "${budgets[@]}"; do
  case "$budget" in
    16)
      checkpoint="${LEGACY_CHECKPOINT_ROOT}/r2b_fresh20k_stage3_base${base_seed}_seed${continuation_seed}/checkpoint_latest_gaze"
      ;;
    24)
      checkpoint="${LEGACY_CHECKPOINT_ROOT}/r2c_fixedk24_stage3_base${base_seed}_seed${continuation_seed}/checkpoint_latest_gaze"
      ;;
    32)
      checkpoint="${K32_CHECKPOINT_ROOT}/r2c_k32_reproduction_stage3_base${base_seed}_seed${continuation_seed}/checkpoint_ep40_iter0/checkpoint_gaze"
      ;;
    *)
      echo "Unsupported or abandoned budget: $budget" >&2
      exit 6
      ;;
  esac
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
