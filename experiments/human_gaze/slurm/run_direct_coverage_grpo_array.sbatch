#!/usr/bin/env bash
#SBATCH --job-name=gaze-grpo
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-grpo-%A_%a.out

set -euo pipefail

arms=(no_kl no_kl no_kl small_kl small_kl small_kl)
seeds=(140826 140827 140828 140826 140827 140828)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
config="av_gaze_stavis_human_coverage_grpo_${arm}"
run_id="r${SLURM_ARRAY_TASK_ID}_${arm}_seed${seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /home/stud/latn/master-thesis/AutoGaze

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5

python -m autogaze.train \
  --config-name "$config" \
  trainer.seed="$seed" \
  trainer.exp_name="$run_id"
