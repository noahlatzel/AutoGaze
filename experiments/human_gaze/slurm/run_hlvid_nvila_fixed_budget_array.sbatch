#!/usr/bin/env bash
#SBATCH --job-name=gaze-hlvid-k
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --nodelist=node19,node20,node21
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-hlvid-k-%A_%a.out

set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
  0)
    budget=24
    base_seed=440827
    continuation_seed=540827
    ;;
  1)
    budget=36
    base_seed=440826
    continuation_seed=540826
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

run_name="r2e_fixedk${budget}_seed${base_seed}_mtv48"
checkpoint="outputs/human_gaze/grpo/r2c_fixedk${budget}_stage3_base${base_seed}_seed${continuation_seed}/checkpoint_latest_gaze"
output_dir="outputs/human_gaze/hlvid_nvila/r2e_fixed_budget_mtv48/${run_name}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
module load cuda/12.8.1
cd /storage/user/latn/worktrees/autogaze-r2e-hlvid

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$output_dir"
echo "run=$run_name checkpoint=$checkpoint"
date
nvidia-smi

python scripts/human_gaze/evaluate_hlvid_nvila_fixed_budget.py \
  --exact-budget "$budget" \
  --fine-action-offset 69 \
  --actions-per-frame 265 \
  --policy-label "fixedk${budget}_seed${base_seed}_20k" \
  --legacy-runner-dir /home/stud/latn/AutoGaze/scripts/runners \
  --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
  --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
  --autogaze-model-id "$checkpoint" \
  --frame-source uniform \
  --tile-len 16 \
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
