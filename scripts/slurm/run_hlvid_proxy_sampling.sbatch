#!/bin/bash
#SBATCH --job-name=hlvid-proxy
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/stud/latn/AutoGaze/outputs/slurm_logs/slurm-%j.out

set -euo pipefail

set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export HYDRA_FULL_ERROR=1

: "${RUN_NAME:?Set RUN_NAME, e.g. uniform_128 or pixel_mse_128_t002}"
: "${SAMPLING:?Set SAMPLING to dense or a fixed frame count, e.g. 128}"
: "${SELECTION:?Set SELECTION to uniform or proxy}"

: "${OUTPUT_ROOT:=outputs/hlvid/proxy_sampling}"
: "${MAX_TILES_VIDEO:=1}"
: "${NUM_VIDEO_FRAMES_THUMBNAIL:=64}"
: "${MAX_NEW_TOKENS:=16}"

OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p outputs/slurm_logs
mkdir -p "${OUT_DIR}"

echo "pwd=$(pwd)"
echo "hostname=$(hostname)"
echo "RUN_NAME=${RUN_NAME}"
echo "SAMPLING=${SAMPLING}"
echo "SELECTION=${SELECTION}"
echo "PROXY_METRIC=${PROXY_METRIC:-}"
echo "PROXY_BASE_METRIC=${PROXY_BASE_METRIC:-pixel_mse}"
echo "SEMANTIC_MODEL=${SEMANTIC_MODEL:-siglip}"
echo "MAX_BATCH_SIZE_SEMANTIC=${MAX_BATCH_SIZE_SEMANTIC:-32}"
echo "LAMBDA_COST=${LAMBDA_COST:-0}"
echo "TEMPORAL_COST=${TEMPORAL_COST:-none}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MAX_TILES_VIDEO=${MAX_TILES_VIDEO}"
echo "NUM_VIDEO_FRAMES_THUMBNAIL=${NUM_VIDEO_FRAMES_THUMBNAIL}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "MAX_DENSE_SAMPLED_FRAMES=${MAX_DENSE_SAMPLED_FRAMES:-}"
date
nvidia-smi

CMD=(
  python scripts/evaluate_hlvid_proxy_sampling.py
  --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid
  --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video
  --autogaze-model-id /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze
  --sampling "${SAMPLING}"
  --selection "${SELECTION}"
  --max-tiles-video "${MAX_TILES_VIDEO}"
  --num-video-frames-thumbnail "${NUM_VIDEO_FRAMES_THUMBNAIL}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --output "${OUT_DIR}/results.jsonl"
  --summary-output "${OUT_DIR}/summary.json"
  --resume
)

if [[ -n "${MAX_DENSE_SAMPLED_FRAMES:-}" ]]; then
  CMD+=(--max-dense-sampled-frames "${MAX_DENSE_SAMPLED_FRAMES}")
fi

if [[ "${SELECTION}" == "proxy" ]]; then
  : "${PROXY_METRIC:?Set PROXY_METRIC when SELECTION=proxy}"
  CMD+=(
    --proxy-metric "${PROXY_METRIC}"
    --proxy-base-metric "${PROXY_BASE_METRIC:-pixel_mse}"
    --semantic-model "${SEMANTIC_MODEL:-siglip}"
    --max-batch-size-semantic "${MAX_BATCH_SIZE_SEMANTIC:-32}"
    --lambda-cost "${LAMBDA_COST:-0}"
    --temporal-cost "${TEMPORAL_COST:-none}"
  )
  if [[ -n "${SIGLIP_MODEL:-}" ]]; then
    CMD+=(--siglip-model "${SIGLIP_MODEL}")
  fi
  if [[ -n "${DINOV2_MODEL:-}" ]]; then
    CMD+=(--dinov2-model "${DINOV2_MODEL}")
  fi
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
