#!/usr/bin/env bash
#SBATCH --job-name=gaze-10k-center
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-10k-center-%A_%a.out

set -eo pipefail

base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
continuation_seed="$((540826 + SLURM_ARRAY_TASK_ID))"
run_id="r2_fresh10k_stage2_base${base_seed}_seed${continuation_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2-fresh10k

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2

python scripts/human_gaze/diagnose_center_collapse.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
  --checkpoint "outputs/human_gaze/grpo/${run_id}/checkpoint_latest_gaze" \
  --split val \
  --batch-size 4 \
  --num-workers 2 \
  --output-json "outputs/human_gaze/diagnostics/${run_id}/center_collapse_val_within_source.json" \
  --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r2_fresh10k_seed${base_seed}_center_collapse.png"
