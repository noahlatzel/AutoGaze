#!/usr/bin/env bash
#SBATCH --job-name=gaze-supervised-k16
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-supervised-k16-%A_%a.out

set -euo pipefail

worktree="${AUTOGAZE_WORKTREE:?Set AUTOGAZE_WORKTREE to the detached admitted checkout}"
execution_commit="${AUTOGAZE_EXECUTION_COMMIT:?Set the exact admitted execution commit}"
run_id="${SUPERVISED_K16_RUN_ID:?Set the immutable experiment run ID}"
artifact_root=/storage/user/latn/artifacts/autogaze-supervised-k16
execution_config=experiments/human_gaze/configs/supervised_k16_execution.yaml
python=/home/stud/latn/miniconda3/envs/autogaze/bin/python

if [[ ! "$run_id" =~ ^[0-9]{8}-[0-9]{4}_supervised-k16-comparison_[0-9a-f]{7}$ ]]; then
  echo "SUPERVISED_K16_RUN_ID does not follow the immutable run-ID contract" >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-5]$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be one of 0..5" >&2
  exit 3
fi
base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
continuation_seed="$((540826 + SLURM_ARRAY_TASK_ID))"
seed_root="${artifact_root}/${run_id}/seed${base_seed}_train${continuation_seed}"
stage1_dir="${seed_root}/stage1"
stage2_dir="${seed_root}/stage2"

if [[ -e "$seed_root" ]]; then
  echo "Refusing to overwrite existing seed artifact directory: $seed_root" >&2
  exit 4
fi
mkdir -p "$seed_root"

module load cuda/12.8.1
set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
set -u
cd "$worktree"

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MASTER_PORT="$((43000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 20000))"

precheck="${seed_root}/admission_precheck.json"
stage1_verification="${seed_root}/stage1_completion_verification.json"
stage2_verification="${seed_root}/stage2_completion_verification.json"
stage1_time="${seed_root}/stage1_time.txt"
stage2_time="${seed_root}/stage2_time.txt"
gpu_telemetry="${seed_root}/gpu_memory.csv"

"$python" scripts/human_gaze/verify_supervised_k16_execution.py precheck \
  --execution-config "$execution_config" \
  --expected-commit "$execution_commit" \
  --array-index "$SLURM_ARRAY_TASK_ID" \
  --output "$precheck"

date --iso-8601=seconds
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
"$python" - <<'PY_ENV'
import json
import platform
import torch
import transformers
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda": torch.version.cuda,
}, sort_keys=True))
PY_ENV

nvidia-smi \
  --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop-ms=1000 > "$gpu_telemetry" &
telemetry_pid=$!
stop_telemetry() {
  if kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap stop_telemetry EXIT

dataset_root=/storage/user/zverev/datasets/av-gaze-stavis
manifest=/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/d0_stavis_fold1_validated/clips.jsonl
cell_mass=/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/d0_stavis_fold1_validated/cell_mass.npy
pretrained=/storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze

/usr/bin/time -v -o "$stage1_time" \
  "$python" -m autogaze.train \
    --config-name av_gaze_stavis_supervised_k16_stage1 \
    dataset.root="$dataset_root" \
    dataset.manifest_path="$manifest" \
    dataset.cell_mass_path="$cell_mass" \
    trainer.gaze_weights="$pretrained" \
    trainer.seed="$base_seed" \
    trainer.logdir="$seed_root" \
    trainer.exp_name=stage1

"$python" scripts/human_gaze/verify_supervised_k16_execution.py checkpoint \
  --stage stage1 \
  --run-dir "$stage1_dir" \
  --base-seed "$base_seed" \
  --continuation-seed "$continuation_seed" \
  --output "$stage1_verification"

/usr/bin/time -v -o "$stage2_time" \
  "$python" -m autogaze.train \
    --config-name av_gaze_stavis_supervised_k16_stage2 \
    dataset.root="$dataset_root" \
    dataset.manifest_path="$manifest" \
    dataset.cell_mass_path="$cell_mass" \
    trainer.gaze_weights="$stage1_dir/checkpoint_latest_gaze" \
    trainer.seed="$continuation_seed" \
    trainer.logdir="$seed_root" \
    trainer.exp_name=stage2

"$python" scripts/human_gaze/verify_supervised_k16_execution.py checkpoint \
  --stage stage2 \
  --run-dir "$stage2_dir" \
  --base-seed "$base_seed" \
  --continuation-seed "$continuation_seed" \
  --output "$stage2_verification"

stop_telemetry
trap - EXIT

"$python" scripts/human_gaze/verify_supervised_k16_execution.py manifest \
  --run-id "$run_id" \
  --precheck "$precheck" \
  --stage1-verification "$stage1_verification" \
  --stage2-verification "$stage2_verification" \
  --stage1-time "$stage1_time" \
  --stage2-time "$stage2_time" \
  --gpu-telemetry "$gpu_telemetry" \
  --output "$seed_root/execution_manifest.json"

echo "Completed fixed supervised K16 endpoint for base_seed=$base_seed continuation_seed=$continuation_seed"
echo "HLVid was not executed by this launcher."
