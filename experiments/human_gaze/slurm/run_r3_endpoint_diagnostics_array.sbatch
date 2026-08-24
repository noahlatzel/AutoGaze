#!/usr/bin/env bash
#SBATCH --job-name=gaze-r3-center
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r3-center-%A_%a.out

set -eo pipefail

seed_index="$((SLURM_ARRAY_TASK_ID % 3))"
arm_index="$((SLURM_ARRAY_TASK_ID / 3))"
base_seed="$((440826 + seed_index))"
training_seed="$((640826 + seed_index))"
if (( arm_index == 0 )); then arm=control; else arm=position; fi
run="r3b_temporal_${arm}_base${base_seed}_seed${training_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r3-analysis
export OMP_NUM_THREADS=3

python scripts/human_gaze/diagnose_center_collapse.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
  --checkpoint "outputs/human_gaze/grpo/${run}/checkpoint_latest_gaze" \
  --output-json "outputs/human_gaze/diagnostics/${run}/center_collapse_val.json" \
  --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/${run}_center_collapse.png" \
  --batch-size 16 \
  --num-workers 3
