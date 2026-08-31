#!/usr/bin/env bash
#SBATCH --job-name=gaze-fresh10k
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-fresh10k-%A_%a.out

set -eo pipefail

base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
continuation_seed="$((540826 + SLURM_ARRAY_TASK_ID))"
stage1_run="r2_fresh10k_stage1_seed${base_seed}"
stage2_run="r2_fresh10k_stage2_base${base_seed}_seed${continuation_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2-fresh10k

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((35000 + SLURM_JOB_ID % 10000 + SLURM_ARRAY_TASK_ID))"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage1 \
  trainer.seed="$base_seed" \
  trainer.exp_name="$stage1_run"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage2 \
  trainer.seed="$continuation_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/${stage1_run}/checkpoint_latest_gaze" \
  trainer.exp_name="$stage2_run"
