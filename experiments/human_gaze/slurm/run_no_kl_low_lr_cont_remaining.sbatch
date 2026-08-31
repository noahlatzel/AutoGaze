#!/usr/bin/env bash
#SBATCH --job-name=gaze-cont-rem
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-cont-rem-%A_%a.out

set -eo pipefail

source_seeds=(140827 140828)
run_seeds=(240827 240828)
source_seed="${source_seeds[$SLURM_ARRAY_TASK_ID]}"
run_seed="${run_seeds[$SLURM_ARRAY_TASK_ID]}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /home/stud/latn/master-thesis/AutoGaze

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((33000 + SLURM_JOB_ID % 10000 + SLURM_ARRAY_TASK_ID))"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_low_lr_cont \
  trainer.seed="$run_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/r$((SLURM_ARRAY_TASK_ID + 1))_no_kl_5ep_seed${source_seed}/checkpoint_latest_gaze" \
  trainer.exp_name="t$((SLURM_ARRAY_TASK_ID + 1))_no_kl_cont_lr3e6_seed${run_seed}"
