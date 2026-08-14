#!/usr/bin/env bash
#SBATCH --job-name=gaze-test
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-test-%A_%a.out

set -eo pipefail

source_seeds=(240826 240827 240828)
run_seed="${source_seeds[$SLURM_ARRAY_TASK_ID]}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /home/stud/latn/master-thesis/AutoGaze

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5

python scripts/human_gaze/evaluate_pretrained_exact16.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
  --checkpoint "outputs/human_gaze/grpo/t${SLURM_ARRAY_TASK_ID}_no_kl_cont_lr3e6_seed${run_seed}/checkpoint_latest_gaze" \
  --evaluation-splits test \
  --output "outputs/human_gaze/test/t${SLURM_ARRAY_TASK_ID}_no_kl_cont_lr3e6_seed${run_seed}.json"
