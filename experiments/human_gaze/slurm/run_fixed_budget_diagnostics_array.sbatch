#!/usr/bin/env bash
#SBATCH --job-name=gaze-k-diagnose
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-k-diagnose-%A_%a.out

set -eo pipefail

run_index="$SLURM_ARRAY_TASK_ID"
budget_index="$((run_index / 3))"
seed_index="$((run_index % 3))"
if (( budget_index == 0 )); then budget=24; else budget=36; fi
base_seed="$((440826 + seed_index))"
continuation_seed="$((540826 + seed_index))"
stage2_run="r2c_fixedk${budget}_stage2_base${base_seed}_seed${continuation_seed}"
stage3_run="r2c_fixedk${budget}_stage3_base${base_seed}_seed${continuation_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2c-fixed-budgets

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2

run_diagnostic() {
  local cumulative_step="$1"
  local run_id="$2"
  shift 2
  python scripts/human_gaze/diagnose_center_collapse.py \
    --config experiments/human_gaze/configs/m0_pretrained_exact16_fold1.yaml \
    --exact-budget "$budget" \
    "$@" \
    --split val \
    --batch-size 4 \
    --num-workers 2 \
    --output-json "outputs/human_gaze/diagnostics/${run_id}/center_collapse_val_step${cumulative_step}.json" \
    --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r2c_fixedk${budget}_seed${base_seed}_step${cumulative_step}_center_collapse.png"
}

run_diagnostic 10000 "$stage2_run" \
  --checkpoint "outputs/human_gaze/grpo/${stage2_run}/checkpoint_latest_gaze"
run_diagnostic 15000 "$stage3_run" \
  --run-dir "outputs/human_gaze/grpo/${stage3_run}" --train-step 12685
run_diagnostic 20000 "$stage3_run" \
  --checkpoint "outputs/human_gaze/grpo/${stage3_run}/checkpoint_latest_gaze"
