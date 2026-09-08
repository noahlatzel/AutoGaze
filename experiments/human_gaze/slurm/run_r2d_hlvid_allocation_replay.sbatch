#!/usr/bin/env bash
#SBATCH --job-name=gaze-r2f-replay
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --nodelist=node17
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r2f-replay-%j.out

set -euo pipefail

worktree="${R2F_REPLAY_WORKTREE:?Set R2F_REPLAY_WORKTREE to the immutable AutoGaze checkout}"
code_commit="${R2F_REPLAY_CODE_COMMIT:?Set R2F_REPLAY_CODE_COMMIT to the immutable commit}"
run_id="${R2F_REPLAY_RUN_ID:?Set R2F_REPLAY_RUN_ID to the allocation-replay run ID}"
output_root="${R2F_REPLAY_OUTPUT_ROOT:-/storage/user/latn/artifacts/autogaze-r2f}"
heavy_root=/storage/user/latn/worktrees/autogaze-r2d-variable-budget
reference_summary=/storage/user/latn/artifacts/autogaze-r2f/20260901-0156_r2f-r2d-hlvid-secondary_ecf1535/preflight/seed440826_variable/summary.json

module load cuda/12.8.1
set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u
cd "$worktree"
export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

for base_seed in 440826 440827 440828; do
  case "$base_seed" in
    440826)
      training_seed=740826
      eos_bias=2.3046875
      model_sha=91398442482aa9faf3bfa61aafb235095a7d6f36dd3796f826420e2b1530dbaa
      calibration_sha=95bb0ebfe145510669b1a5d256092146ff5c55170b074dbdc74d96a224246ccf
      validation_args=(--validation-reference-summary "$reference_summary")
      ;;
    440827)
      training_seed=740827
      eos_bias=2.421875
      model_sha=0d1736f07c381e4c77a4f297fa09cfa61b4483c48f02303a8e16ccdf09ee5363
      calibration_sha=c7c8b5041f081aab7f2bed40408a181a4f30d0bc9cba5a95faba126262e519eb
      validation_args=(--validated-replay-summary "$output_root/$run_id/replays/seed440826_variable/summary.json")
      ;;
    440828)
      training_seed=740828
      eos_bias=2.3828125
      model_sha=297e436cc599a6904478554763a746f6733854c544faabf9a3aa652179f370c9
      calibration_sha=15756e1577173a7b1be3a67f81544d13f2cc819370f3caf42cacb198ce7ba3e8
      validation_args=(--validated-replay-summary "$output_root/$run_id/replays/seed440826_variable/summary.json")
      ;;
  esac

  checkpoint="$heavy_root/outputs/human_gaze/grpo/r2d_variable_k16_stable_base${base_seed}_seed${training_seed}/checkpoint_latest_gaze"
  calibration="$heavy_root/outputs/human_gaze/diagnostics/r2d_variable_k16_stable_base${base_seed}_seed${training_seed}/variable_budget_val.json"
  output_dir="$output_root/$run_id/replays/seed${base_seed}_variable"
  mkdir -p "$output_dir"
  python scripts/human_gaze/replay_r2d_hlvid_allocation.py \
    --r2d-mode variable \
    --base-seed "$base_seed" \
    --training-seed "$training_seed" \
    --checkpoint-model-sha256 "$model_sha" \
    --checkpoint-config-sha256 2fbb670402ba0ac378b97a7ef82f4f69740fa94b4523cc34ca6b6bcd2a2f9887 \
    --calibration-json "$calibration" \
    --calibration-sha256 "$calibration_sha" \
    --eos-logit-bias "$eos_bias" \
    --legacy-runner-dir /home/stud/latn/AutoGaze/scripts/runners \
    --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
    --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
    --autogaze-model-id "$checkpoint" \
    --num-video-frames 128 \
    --num-video-frames-thumbnail 64 \
    --max-tiles-video 48 \
    --max-batch-size-autogaze 16 \
    --expected-code-commit "$code_commit" \
    --resume \
    --output "$output_dir/results.jsonl" \
    --summary-output "$output_dir/summary.json" \
    --attempt-log "$output_dir/attempts.jsonl" \
    "${validation_args[@]}"
done
