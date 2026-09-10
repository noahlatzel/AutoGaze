#!/usr/bin/env bash
# Export the five fixed supervised checkpoints and matched RL endpoint for one seed.
# Run this only inside a separately admitted GPU allocation after training completes.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BASE_SEED SUPERVISED_TRAINING_ROOT ACTION_EXPORT_ROOT" >&2
  exit 2
fi

base_seed="$1"
training_root="$2"
action_root="$3"
if [[ ! "$base_seed" =~ ^44082[6-9]$|^44083[01]$ ]]; then
  echo "BASE_SEED must be one of 440826..440831" >&2
  exit 3
fi
training_seed="$((base_seed + 100000))"
python="${AUTOGAZE_PYTHON:-/home/stud/latn/miniconda3/envs/autogaze/bin/python}"
dataset_root=/storage/user/zverev/datasets/av-gaze-stavis
manifest=/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/d0_stavis_fold1_validated/clips.jsonl
cell_mass=/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/d0_stavis_fold1_validated/cell_mass.npy
seed_root="${training_root}/seed${base_seed}_train${training_seed}"
stage1_dir="${SUPERVISED_STAGE1_DIR:-${seed_root}/stage1}"
stage2_dir="${SUPERVISED_STAGE2_DIR:-${seed_root}/stage2}"

steps=(2315 5000 10000 15000 20000)
phases=(stage1 stage2 stage2 stage2 stage2)
phase_steps=(2315 2685 7685 12685 17685)
for index in "${!steps[@]}"; do
  step="${steps[$index]}"
  phase="${phases[$index]}"
  phase_step="${phase_steps[$index]}"
  if [[ "$phase" == stage1 ]]; then
    phase_dir="$stage1_dir"
  else
    phase_dir="$stage2_dir"
  fi
  "$python" scripts/human_gaze/export_supervised_k16_actions.py \
    --dataset-root "$dataset_root" \
    --manifest "$manifest" \
    --cell-mass "$cell_mass" \
    --run-dir "$phase_dir" \
    --phase-train-step "$phase_step" \
    --method supervised \
    --base-seed "$base_seed" \
    --training-seed "$training_seed" \
    --cumulative-update "$step" \
    --output-dir "${action_root}/supervised/base${base_seed}_train${training_seed}/step${step}" \
    --batch-size 16 \
    --num-workers 4
done

rl_checkpoint="/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/grpo/r2b_fresh20k_stage3_base${base_seed}_seed${training_seed}/checkpoint_latest_gaze"
"$python" scripts/human_gaze/export_supervised_k16_actions.py \
  --dataset-root "$dataset_root" \
  --manifest "$manifest" \
  --cell-mass "$cell_mass" \
  --checkpoint "$rl_checkpoint" \
  --method rl \
  --base-seed "$base_seed" \
  --training-seed "$training_seed" \
  --cumulative-update 20000 \
  --output-dir "${action_root}/rl/base${base_seed}_train${training_seed}/step20000" \
  --batch-size 16 \
  --num-workers 4
