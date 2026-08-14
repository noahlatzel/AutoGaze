#!/usr/bin/env bash
#SBATCH --job-name=gaze-center-check
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-center-check-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /home/stud/latn/master-thesis/AutoGaze

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5

python scripts/human_gaze/diagnose_center_collapse.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
  --checkpoint outputs/human_gaze/grpo/t0_no_kl_cont_lr3e6_seed240826/checkpoint_latest_gaze \
  --split val \
  --output-json outputs/human_gaze/diagnostics/t0_no_kl_cont_lr3e6_seed240826/center_collapse_val.json \
  --output-plot ../wiki_gazing/assets/learned_policy_selection_frequency_val.png
