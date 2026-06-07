#!/bin/bash
#SBATCH --job-name=hlvid-proxy
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/slurm-%j.out

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

OUT_DIR="outputs/hlvid/proxy_sampling/${RUN_NAME}"
mkdir -p /storage/user/latn/slurm/logs
mkdir -p "${OUT_DIR}"

echo "pwd=$(pwd)"
echo "hostname=$(hostname)"
echo "RUN_NAME=${RUN_NAME}"
echo "SAMPLING=${SAMPLING}"
echo "SELECTION=${SELECTION}"
echo "PROXY=${PROXY:-}"
echo "THRESHOLD=${THRESHOLD:-}"
echo "MAX_TILES_VIDEO=${MAX_TILES_VIDEO:-legacy-default}"
echo "MAX_DENSE_SAMPLED_FRAMES=${MAX_DENSE_SAMPLED_FRAMES:-}"
date
nvidia-smi

: "${MAX_TILES_VIDEO:=1}"

CMD=(
  python scripts/evaluate_hlvid_proxy_sampling.py
  --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid
  --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video
  --autogaze-model-id /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze
  --sampling "${SAMPLING}"
  --selection "${SELECTION}"
  --max-tiles-video "${MAX_TILES_VIDEO}"
  --output "${OUT_DIR}/results.jsonl"
  --summary-output "${OUT_DIR}/summary.json"
  --resume
)

if [[ -n "${MAX_DENSE_SAMPLED_FRAMES:-}" ]]; then
  CMD+=(--max-dense-sampled-frames "${MAX_DENSE_SAMPLED_FRAMES}")
fi

if [[ "${SELECTION}" == "proxy" ]]; then
  : "${PROXY:?Set PROXY when SELECTION=proxy}"
  : "${THRESHOLD:?Set THRESHOLD when SELECTION=proxy}"
  CMD+=(--proxy "${PROXY}" --threshold "${THRESHOLD}")
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
