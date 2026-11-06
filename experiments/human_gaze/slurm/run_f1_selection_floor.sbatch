#!/usr/bin/env bash
#SBATCH --job-name=f1-hlvid-floor
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:80G
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/f1-hlvid-floor-%j.out
#SBATCH --error=/storage/user/latn/slurm/logs/f1-hlvid-floor-%j.err

# Gate F1: Random16 and an unchanged Center16 rerun, as two concurrent
# processes on one GPU in one allocation (each arm ~34 GB CUDA peak; serial
# execution of the CPU-bound preprocessing would exceed the 12 h ceiling).

set -euo pipefail

: "${F1_WORKTREE:?Set F1_WORKTREE to the frozen clean checkout}"
: "${F1_COMMIT:?Set F1_COMMIT to the immutable pushed commit}"
: "${F1_RUN_ID:?Set F1_RUN_ID}"
: "${F1_ARTIFACT_ROOT:?Set F1_ARTIFACT_ROOT to the append-only artifact directory}"

CONFIG=experiments/human_gaze/configs/f1_hlvid_selection_floor_v1.yaml
DECODE_AUDIT_DIR=/home/stud/latn/.codex/worktrees/r2f-hlvid-curation/outputs/hlvid/wp0_protocol_audit/20260909-0211_wp0-hlvid-protocol-audit_b98970d
RANDOM_CELLS_SEED=20261106

module load cuda/12.8.1
set +u
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate vila-autogaze-eval
set -u
cd "$F1_WORKTREE"

if [[ "$(git rev-parse HEAD)" != "$F1_COMMIT" ]]; then
  echo "Commit mismatch: $(git rev-parse HEAD) != $F1_COMMIT" >&2; exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked F1 execution worktree is dirty" >&2; git status --short >&2; exit 4
fi
for f in protocol_runtime_manifest.json summary.json decode_audit.jsonl live_runtime_supplement.json; do
  [[ -f "$DECODE_AUDIT_DIR/$f" ]] || { echo "Incomplete decode audit bundle" >&2; exit 5; }
done
grep -q "random_cells_seed: $RANDOM_CELLS_SEED" "$CONFIG" || { echo "Seed differs from config" >&2; exit 6; }

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

mkdir -p "$F1_ARTIFACT_ROOT"
echo "run_id=$F1_RUN_ID commit=$F1_COMMIT job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)" \
  | tee -a "$F1_ARTIFACT_ROOT/job_log.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv | tee "$F1_ARTIFACT_ROOT/gpu.csv"

admission_record="$F1_ARTIFACT_ROOT/admission_record.json"
python scripts/human_gaze/check_hlvid_control_admission.py \
  --config "$CONFIG" \
  --smoke-report experiments/human_gaze/results/r2e_hlvid_k16_causal_controls/preflight/processor_smoke_224.json \
  --expected-code-commit "$F1_COMMIT" \
  --output "$admission_record"

run_arm() {
  local arm_id="$1" policy_kind="$2" policy_label="$3"; shift 3
  local run_dir="$F1_ARTIFACT_ROOT/$arm_id"
  mkdir -p "$run_dir"
  cp -n "$admission_record" "$run_dir/admission_record.json"
  evaluator() {
    python scripts/human_gaze/evaluate_hlvid_nvila_fixed_budget.py \
      --exact-budget 16 \
      --policy-kind "$policy_kind" \
      --policy-label "$policy_label" \
      --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
      --expected-dataset-sha256 ed2a1a47603fd19f5d7fae0db4f9792369f5aff6548f35e0225c3b4bd35e112a \
      --model-path /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--NVILA-8B-HD-Video \
      --autogaze-model-id /storage/slurm/latn/data/AutoGaze/hf_models/nvidia--AutoGaze \
      --decode-audit-manifest "$DECODE_AUDIT_DIR" \
      --num-video-frames 128 --num-video-frames-thumbnail 64 --max-tiles-video 48 \
      --tile-len 16 --max-new-tokens 16 --max-batch-size-autogaze 16 --max-batch-size-siglip 32 \
      --torch-dtype bfloat16 --fine-action-offset 69 --actions-per-frame 265 \
      --resume \
      --output "$run_dir/results.jsonl" \
      --evidence-dir "$run_dir/evidence_records" \
      --event-output "$run_dir/events.jsonl" \
      --run-manifest "$run_dir/run_manifest.json" \
      --summary-output "$run_dir/summary.json" \
      "$@"
  }
  if [[ ! -f "$run_dir/summary.json" ]]; then
    evaluator "$@" --limit 1
  fi
  python scripts/human_gaze/check_hlvid_control_preflight.py \
    --summary "$run_dir/summary.json" \
    --evidence-dir "$run_dir/evidence_records" \
    --policy-kind "$policy_kind" \
    --admission-record "$run_dir/admission_record.json" \
    --attestation "$run_dir/preflight_pass.json"
  evaluator "$@"
}

run_arm random16 random16 random16_uniform --random-cells-seed "$RANDOM_CELLS_SEED" \
  > "$F1_ARTIFACT_ROOT/random16.log" 2>&1 &
pid_random=$!
run_arm stavis_center16 center16 stavis_center16 \
  > "$F1_ARTIFACT_ROOT/stavis_center16.log" 2>&1 &
pid_center=$!

status=0
wait "$pid_random" || { echo "random16 arm failed" >&2; status=1; }
wait "$pid_center" || { echo "stavis_center16 arm failed" >&2; status=1; }
echo "end=$(date --iso-8601=seconds) status=$status" | tee -a "$F1_ARTIFACT_ROOT/job_log.txt"
exit "$status"
