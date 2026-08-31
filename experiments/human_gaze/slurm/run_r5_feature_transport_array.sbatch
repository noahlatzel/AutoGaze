#!/usr/bin/env bash
#SBATCH --job-name=gaze-r5-transport
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r5-transport-%A_%a.out

set -eo pipefail

base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
endpoint_seed="$((base_seed + 100000))"
training_seed="$((base_seed + 200000))"
initial_run="r2b_fresh20k_stage3_base${base_seed}_seed${endpoint_seed}"
run="r5_feature_transport_base${base_seed}_seed${training_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r5-transport

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((49000 + SLURM_ARRAY_TASK_ID))"

code_commit="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked R5 worktree is dirty" >&2
  exit 2
fi

python scripts/human_gaze/write_r5_run_manifests.py \
  --base-seed "$base_seed" \
  --job-id "$SLURM_ARRAY_JOB_ID" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --code-commit "$code_commit"

python -m autogaze.train \
  --config-name av_gaze_stavis_r5_feature_transport \
  trainer.seed="$training_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/${initial_run}/checkpoint_latest_gaze" \
  trainer.exp_name="$run"
