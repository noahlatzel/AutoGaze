#!/usr/bin/env bash
#SBATCH --job-name=gaze-r6-center
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r6-center-%A_%a.out

set -eo pipefail

base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
training_seed="$((base_seed + 200000))"
run="r6_recurrent_state_base${base_seed}_seed${training_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r6-analysis
export OMP_NUM_THREADS=3

python scripts/human_gaze/diagnose_center_collapse.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
  --checkpoint "outputs/human_gaze/grpo/${run}/checkpoint_latest_gaze" \
  --output-json "outputs/human_gaze/diagnostics/${run}/center_collapse_val.json" \
  --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/${run}_center_collapse.png" \
  --batch-size 16 \
  --num-workers 3
