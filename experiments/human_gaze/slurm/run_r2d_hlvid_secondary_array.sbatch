#!/usr/bin/env bash
#SBATCH --job-name=gaze-r2f-hlvid
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --nodelist=node19,node20,node21
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r2f-hlvid-%A_%a.out

set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
  0) base_seed=440826; training_seed=740826; mode=variable; eos_bias=2.3046875; model_sha=91398442482aa9faf3bfa61aafb235095a7d6f36dd3796f826420e2b1530dbaa; calibration_sha=95bb0ebfe145510669b1a5d256092146ff5c55170b074dbdc74d96a224246ccf ;;
  1) base_seed=440826; training_seed=740826; mode=forced_k16; eos_bias=2.3046875; model_sha=91398442482aa9faf3bfa61aafb235095a7d6f36dd3796f826420e2b1530dbaa; calibration_sha=95bb0ebfe145510669b1a5d256092146ff5c55170b074dbdc74d96a224246ccf ;;
  2) base_seed=440827; training_seed=740827; mode=variable; eos_bias=2.421875; model_sha=0d1736f07c381e4c77a4f297fa09cfa61b4483c48f02303a8e16ccdf09ee5363; calibration_sha=c7c8b5041f081aab7f2bed40408a181a4f30d0bc9cba5a95faba126262e519eb ;;
  3) base_seed=440827; training_seed=740827; mode=forced_k16; eos_bias=2.421875; model_sha=0d1736f07c381e4c77a4f297fa09cfa61b4483c48f02303a8e16ccdf09ee5363; calibration_sha=c7c8b5041f081aab7f2bed40408a181a4f30d0bc9cba5a95faba126262e519eb ;;
  4) base_seed=440828; training_seed=740828; mode=variable; eos_bias=2.3828125; model_sha=297e436cc599a6904478554763a746f6733854c544faabf9a3aa652179f370c9; calibration_sha=15756e1577173a7b1be3a67f81544d13f2cc819370f3caf42cacb198ce7ba3e8 ;;
  5) base_seed=440828; training_seed=740828; mode=forced_k16; eos_bias=2.3828125; model_sha=297e436cc599a6904478554763a746f6733854c544faabf9a3aa652179f370c9; calibration_sha=15756e1577173a7b1be3a67f81544d13f2cc819370f3caf42cacb198ce7ba3e8 ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

heavy_root=/storage/user/latn/worktrees/autogaze-r2d-variable-budget
checkpoint="$heavy_root/outputs/human_gaze/grpo/r2d_variable_k16_stable_base${base_seed}_seed${training_seed}/checkpoint_latest_gaze"
calibration="$heavy_root/outputs/human_gaze/diagnostics/r2d_variable_k16_stable_base${base_seed}_seed${training_seed}/variable_budget_val.json"
worktree="${R2F_AUTOGAZE_WORKTREE:-/home/stud/latn/.codex/worktrees/064d/master-thesis/AutoGaze}"
run_id="${R2F_RUN_ID:?Set R2F_RUN_ID to the committed implementation run ID}"
output_root="${R2F_OUTPUT_ROOT:-/storage/user/latn/artifacts/autogaze-r2f}"
output_dir="$output_root/$run_id/rollouts/seed${base_seed}_${mode}"

module load cuda/12.8.1
set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u
cd "$worktree"
export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output_dir"

python scripts/human_gaze/evaluate_hlvid_nvila_r2d.py \
  --r2d-mode "$mode" \
  --base-seed "$base_seed" \
  --training-seed "$training_seed" \
  --policy-label "r2d_seed${base_seed}_${mode}" \
  --checkpoint-model-sha256 "$model_sha" \
  --checkpoint-config-sha256 2fbb670402ba0ac378b97a7ef82f4f69740fa94b4523cc34ca6b6bcd2a2f9887 \
  --calibration-json "$calibration" \
  --calibration-sha256 "$calibration_sha" \
  --eos-logit-bias "$eos_bias" \
  --legacy-runner-dir /home/stud/latn/AutoGaze/scripts/runners \
  --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
  --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
  --autogaze-model-id "$checkpoint" \
  --frame-source uniform \
  --tile-len 16 \
  --delta-gap 0 \
  --max-tiles-video 48 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-new-tokens 16 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --torch-dtype bfloat16 \
  --resume \
  --output "$output_dir/results.jsonl" \
  --summary-output "$output_dir/summary.json"
