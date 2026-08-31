#!/usr/bin/env bash
#SBATCH --job-name=gaze-cont2
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-cont2-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /home/stud/latn/master-thesis/AutoGaze

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT="$((34000 + SLURM_JOB_ID % 10000))"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_low_lr_cont \
  trainer.seed=340826 \
  trainer.gaze_weights=outputs/human_gaze/grpo/t0_no_kl_cont_lr3e6_seed240826/checkpoint_latest_gaze \
  trainer.exp_name=u0_no_kl_cont2_lr3e6_seed340826
