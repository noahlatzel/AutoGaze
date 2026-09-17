#!/usr/bin/env bash
#SBATCH --job-name=gaze-hlvid-k16-at-k8
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-hlvid-k16-at-k8-%A_%a.out
#SBATCH --error=/storage/user/latn/slurm/logs/gaze-hlvid-k16-at-k8-%A_%a.err
set -euo pipefail
: "${AUTOGAZE_WORKTREE:?}"
: "${AUTOGAZE_COMMIT:?}"
: "${RUN_PREFIX:?}"
module load cuda/12.8.1
set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u
cd "$AUTOGAZE_WORKTREE"
[[ "$(git rev-parse HEAD)" == "$AUTOGAZE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
base_seed="$((440826 + SLURM_ARRAY_TASK_ID))"
continuation_seed="$((base_seed + 100000))"
checkpoint="/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/grpo/r2b_fresh20k_stage3_base${base_seed}_seed${continuation_seed}/checkpoint_ep40_iter0/checkpoint_gaze"
run_id="${RUN_PREFIX}_seed${base_seed}"
run_dir="outputs/human_gaze/hlvid_nvila/r2e_k16_at_k8/${run_id}"
mkdir -p "$run_dir"
config="experiments/human_gaze/configs/r2e_hlvid_k16_at_k8.yaml"
cp "$config" "$run_dir/config.yaml"
uv run --no-project --python "$CONDA_PREFIX/bin/python" - "$config" "$base_seed" <<'PY_VERIFY'
import hashlib,sys,yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
row=next(r for r in cfg['seeds'] if r['base_seed']==int(sys.argv[2]))
for name,expected in row['checkpoint_sha256'].items():
 h=hashlib.sha256()
 with (Path(row['checkpoint'])/name).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==expected, name
print('Checkpoint hash verification passed',row['base_seed'])
PY_VERIFY
echo "run_id=$run_id code=$AUTOGAZE_COMMIT"
date --iso-8601=seconds
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
uv run --no-project --python "$CONDA_PREFIX/bin/python" scripts/human_gaze/evaluate_hlvid_nvila_fixed_budget.py \
  --exact-budget 8 --base-seed "$base_seed" --continuation-seed "$continuation_seed" \
  --policy-label "k16trained_20k_evalk8_seed${base_seed}" \
  --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
  --expected-dataset-sha256 ed2a1a47603fd19f5d7fae0db4f9792369f5aff6548f35e0225c3b4bd35e112a \
  --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
  --autogaze-model-id "$checkpoint" \
  --num-video-frames 128 --num-video-frames-thumbnail 64 --max-tiles-video 48 \
  --tile-len 16 --max-new-tokens 16 --max-batch-size-autogaze 16 --max-batch-size-siglip 32 \
  --torch-dtype bfloat16 --fine-action-offset 69 --actions-per-frame 265 \
  --resume --output "$run_dir/results.jsonl" --evidence-dir "$run_dir/evidence_records" \
  --event-output "$run_dir/events.jsonl" --run-manifest "$run_dir/run_manifest.json" \
  --summary-output "$run_dir/summary.json"
