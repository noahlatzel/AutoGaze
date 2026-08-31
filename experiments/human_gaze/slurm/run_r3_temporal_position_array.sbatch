#!/usr/bin/env bash
#SBATCH --job-name=gaze-r3-time
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r3-time-%A_%a.out

set -eo pipefail

seed_index="$((SLURM_ARRAY_TASK_ID % 3))"
arm_index="$((SLURM_ARRAY_TASK_ID / 3))"
base_seed="$((440826 + seed_index))"
endpoint_seed="$((540826 + seed_index))"
training_seed="$((640826 + seed_index))"
if (( arm_index == 0 )); then
  arm=control
  temporal_mode=none
else
  arm=position
  temporal_mode=normalized_sinusoidal_scalar_gate
fi
initial_run="r2b_fresh20k_stage3_base${base_seed}_seed${endpoint_seed}"
run="r3b_temporal_${arm}_base${base_seed}_seed${training_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r3-temporal

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((47000 + SLURM_ARRAY_TASK_ID))"

python -m autogaze.train \
  --config-name av_gaze_stavis_r3_temporal_position \
  model.gaze_model_config.temporal_position_encoding="$temporal_mode" \
  trainer.seed="$training_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/${initial_run}/checkpoint_latest_gaze" \
  trainer.exp_name="$run"
