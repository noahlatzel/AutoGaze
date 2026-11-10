#!/usr/bin/env bash
#SBATCH --job-name=l2-gaze-hlvid
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=20:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/storage/user/latn/slurm/logs/l2-gaze-hlvid-%A_%a.out
#SBATCH --error=/storage/user/latn/slurm/logs/l2-gaze-hlvid-%A_%a.err

# Gate L2: the minimal Study III HLVid set with a processor layout that matches
# the vision tower. One arm per array task, every arm on the same GPU model.
# Cumulative budget 20 h per arm across requeues; resumable from evidence records.

set -euo pipefail

: "${L2_WORKTREE:?Set L2_WORKTREE to the frozen clean checkout}"
: "${L2_COMMIT:?Set L2_COMMIT to the immutable pushed commit}"
: "${L2_RUN_ID:?Set L2_RUN_ID}"
: "${L2_ARTIFACT_ROOT:?Set L2_ARTIFACT_ROOT to the append-only artifact directory}"

CONFIG=experiments/human_gaze/configs/l2_hlvid_consistent_layout_v1.yaml
DECODE_AUDIT_DIR=/home/stud/latn/.codex/worktrees/r2f-hlvid-curation/outputs/hlvid/wp0_protocol_audit/20260909-0211_wp0-hlvid-protocol-audit_b98970d
PRETRAINED=/storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze
GRPO=/home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/grpo
BUDGET=72000

case "$SLURM_ARRAY_TASK_ID" in
  0) arm=center16;       kind=center16;   label=l2_center16;                checkpoint="$PRETRAINED"; extra=() ;;
  1) arm=random16;       kind=random16;   label=l2_random16_uniform;        checkpoint="$PRETRAINED"; extra=(--random-cells-seed 20261106) ;;
  2) arm=pretrained;     kind=pretrained; label=l2_pretrained_exact_k16;    checkpoint="$PRETRAINED"; extra=() ;;
  3) arm=trained_440826; kind=trained;    label=l2_fixedk16_seed440826_20k; checkpoint="$GRPO/r2b_fresh20k_stage3_base440826_seed540826/checkpoint_latest_gaze"; extra=(--base-seed 440826 --continuation-seed 540826) ;;
  4) arm=trained_440827; kind=trained;    label=l2_fixedk16_seed440827_20k; checkpoint="$GRPO/r2b_fresh20k_stage3_base440827_seed540827/checkpoint_latest_gaze"; extra=(--base-seed 440827 --continuation-seed 540827) ;;
  *) echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2; exit 6 ;;
esac

module load cuda/12.8.1
set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u
cd "$L2_WORKTREE"

if [[ "$(git rev-parse HEAD)" != "$L2_COMMIT" ]]; then
  echo "Commit mismatch: $(git rev-parse HEAD) != $L2_COMMIT" >&2; exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked L2 execution worktree is dirty" >&2; git status --short >&2; exit 4
fi
for f in protocol_runtime_manifest.json summary.json decode_audit.jsonl live_runtime_supplement.json; do
  [[ -f "$DECODE_AUDIT_DIR/$f" ]] || { echo "Incomplete decode audit bundle" >&2; exit 5; }
done

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

run_dir="$L2_ARTIFACT_ROOT/$arm"
mkdir -p "$run_dir"

# Cumulative time accounting that survives preemption and requeue.
used_file="$run_dir/budget_used_seconds"
used=0
[[ -f "$used_file" ]] && used=$(cat "$used_file")
started=$(date +%s)
record_usage() {
  echo $((used + $(date +%s) - started)) > "$used_file.tmp" && mv "$used_file.tmp" "$used_file"
}
( while sleep 60; do record_usage; done ) &
ticker=$!
trap 'record_usage; kill "$ticker" 2>/dev/null || true' EXIT
left=$((BUDGET - used))
if [[ "$left" -le 600 ]]; then
  echo "cumulative budget of ${BUDGET}s exhausted for $arm" >&2; exit 7
fi

echo "run_id=$L2_RUN_ID arm=$arm commit=$L2_COMMIT job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date --iso-8601=seconds) used=${used}s" \
  | tee -a "$run_dir/job_log.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader | tee -a "$run_dir/gpu.csv"

python scripts/human_gaze/check_hlvid_control_admission.py \
  --config "$CONFIG" \
  --smoke-report experiments/human_gaze/results/r2e_hlvid_k16_causal_controls/preflight/processor_smoke_224.json \
  --expected-code-commit "$L2_COMMIT" \
  --output "$run_dir/admission_record.json"

evaluator() {
  timeout --signal=TERM --kill-after=120 "${left}s" \
    python scripts/human_gaze/evaluate_hlvid_nvila_fixed_budget.py \
    --exact-budget 16 \
    --policy-kind "$kind" \
    --policy-label "$label" \
    --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
    --expected-dataset-sha256 ed2a1a47603fd19f5d7fae0db4f9792369f5aff6548f35e0225c3b4bd35e112a \
    --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
    --autogaze-model-id "$checkpoint" \
    --decode-audit-manifest "$DECODE_AUDIT_DIR" \
    --num-video-frames 128 --num-video-frames-thumbnail 64 --max-tiles-video 48 \
    --tile-len 16 --max-new-tokens 16 --max-batch-size-autogaze 16 --max-batch-size-siglip 32 \
    --torch-dtype bfloat16 --fine-action-offset 69 --actions-per-frame 265 \
    --target-scales 64,128,224,448 --target-patch-size 16 \
    --resume \
    --output "$run_dir/results.jsonl" \
    --evidence-dir "$run_dir/evidence_records" \
    --event-output "$run_dir/events.jsonl" \
    --run-manifest "$run_dir/run_manifest.json" \
    --summary-output "$run_dir/summary.json" \
    "${extra[@]}" "$@"
}

# Question 0 first; the full run starts only after its record passes the preflight.
if [[ ! -f "$run_dir/summary.json" ]]; then
  evaluator --limit 1
fi
python scripts/human_gaze/check_hlvid_control_preflight.py \
  --summary "$run_dir/summary.json" \
  --evidence-dir "$run_dir/evidence_records" \
  --policy-kind "$kind" \
  --admission-record "$run_dir/admission_record.json" \
  --attestation "$run_dir/preflight_pass.json"
left=$((BUDGET - used - $(date +%s) + started))
evaluator
echo "end=$(date --iso-8601=seconds) arm=$arm status=0" | tee -a "$run_dir/job_log.txt"
