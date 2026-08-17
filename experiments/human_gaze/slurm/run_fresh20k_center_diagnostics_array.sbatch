#!/usr/bin/env bash
#SBATCH --job-name=gaze-20k-center
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-20k-center-%A_%a.out

set -eo pipefail

seed_index="$((SLURM_ARRAY_TASK_ID % 6))"
base_seed="$((440826 + seed_index))"
continuation_seed="$((540826 + seed_index))"
run_id="r2b_fresh20k_stage3_base${base_seed}_seed${continuation_seed}"

if (( SLURM_ARRAY_TASK_ID < 6 )); then
  cumulative_step=15000
  checkpoint_args=(
    --run-dir "outputs/human_gaze/grpo/${run_id}"
    --train-step 12685
  )
else
  cumulative_step=20000
  checkpoint_args=(
    --checkpoint "outputs/human_gaze/grpo/${run_id}/checkpoint_latest_gaze"
  )
fi

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2b-fresh20k

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2

python scripts/human_gaze/diagnose_center_collapse.py \
  --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
  "${checkpoint_args[@]}" \
  --split val \
  --batch-size 4 \
  --num-workers 2 \
  --output-json "outputs/human_gaze/diagnostics/${run_id}/center_collapse_val_step${cumulative_step}.json" \
  --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r2b_fresh20k_seed${base_seed}_step${cumulative_step}_center_collapse.png"
