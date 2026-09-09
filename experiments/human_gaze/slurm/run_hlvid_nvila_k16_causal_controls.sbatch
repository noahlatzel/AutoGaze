#!/usr/bin/env bash
#SBATCH --job-name=gaze-hlvid-k16-ctrl
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --nodelist=node21
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-hlvid-k16-ctrl-%A_%a.out
#SBATCH --error=/storage/user/latn/slurm/logs/gaze-hlvid-k16-ctrl-%A_%a.err

set -euo pipefail

: "${AUTOGAZE_WORKTREE:=/home/stud/latn/.codex/worktrees/3f12/master-thesis/AutoGaze}"
: "${AUTOGAZE_COMMIT:?Set AUTOGAZE_COMMIT to the immutable pushed control commit}"
: "${DECODE_AUDIT_DIR:?Set DECODE_AUDIT_DIR to the inspected WP0 audit directory}"

module load cuda/12.8.1
set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u
cd "$AUTOGAZE_WORKTREE"

observed_commit="$(git rev-parse HEAD)"
if [[ "$observed_commit" != "$AUTOGAZE_COMMIT" ]]; then
  echo "Commit mismatch: $observed_commit != $AUTOGAZE_COMMIT" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked AutoGaze execution worktree is dirty" >&2
  git status --short >&2
  exit 4
fi
if [[ ! -f "$DECODE_AUDIT_DIR/protocol_runtime_manifest.json" ]] || \
   [[ ! -f "$DECODE_AUDIT_DIR/summary.json" ]] || \
   [[ ! -f "$DECODE_AUDIT_DIR/decode_audit.jsonl" ]] || \
   [[ ! -f "$DECODE_AUDIT_DIR/live_runtime_supplement.json" ]]; then
  echo "Incomplete decode audit bundle: $DECODE_AUDIT_DIR" >&2
  exit 5
fi

case "$SLURM_ARRAY_TASK_ID" in
  0)
    policy_kind="pretrained"
    policy_label="pretrained_exact_k16"
    ;;
  1)
    policy_kind="center16"
    policy_label="stavis_center16"
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 6
    ;;
esac

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

run_dir="outputs/human_gaze/hlvid_nvila/r2e_k16_causal_controls_mtv48/${policy_label}"
mkdir -p "$run_dir"
date --iso-8601=seconds
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv

run_evaluator() {
  python scripts/human_gaze/evaluate_hlvid_nvila_fixed_budget.py \
    --exact-budget 16 \
    --policy-kind "$policy_kind" \
    --policy-label "$policy_label" \
    --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
    --expected-dataset-sha256 ed2a1a47603fd19f5d7fae0db4f9792369f5aff6548f35e0225c3b4bd35e112a \
    --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
    --autogaze-model-id /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze \
    --decode-audit-manifest "$DECODE_AUDIT_DIR" \
    --num-video-frames 128 \
    --num-video-frames-thumbnail 64 \
    --max-tiles-video 48 \
    --tile-len 16 \
    --max-new-tokens 16 \
    --max-batch-size-autogaze 16 \
    --max-batch-size-siglip 32 \
    --torch-dtype bfloat16 \
    --fine-action-offset 69 \
    --actions-per-frame 265 \
    --resume \
    --output "$run_dir/results.jsonl" \
    --evidence-dir "$run_dir/evidence_records" \
    --event-output "$run_dir/events.jsonl" \
    --run-manifest "$run_dir/run_manifest.json" \
    --summary-output "$run_dir/summary.json" \
    "$@"
}

admission_record="$run_dir/admission_record.json"
python scripts/human_gaze/check_hlvid_control_admission.py \
  --config experiments/human_gaze/configs/r2e_hlvid_k16_causal_controls.yaml \
  --smoke-report experiments/human_gaze/results/r2e_hlvid_k16_causal_controls/preflight/processor_smoke_224.json \
  --expected-code-commit "$AUTOGAZE_COMMIT" \
  --output "$admission_record"

# The first invocation is the bounded generation/plumbing smoke. A durable
# attestation binds question 0 to this run identity. Restarts with more than
# one completed answer revalidate question 0 from the full evidence directory
# and never re-enter a one-row evaluator subset.
preflight_attestation="$run_dir/preflight_pass.json"
if [[ ! -f "$run_dir/summary.json" ]]; then
  run_evaluator --limit 1
fi
python scripts/human_gaze/check_hlvid_control_preflight.py \
  --summary "$run_dir/summary.json" \
  --evidence-dir "$run_dir/evidence_records" \
  --policy-kind "$policy_kind" \
  --admission-record "$admission_record" \
  --attestation "$preflight_attestation"
run_evaluator
