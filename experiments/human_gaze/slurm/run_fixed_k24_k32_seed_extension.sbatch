#!/usr/bin/env bash
#SBATCH --job-name=gaze-k24-k32-ext
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-k24-k32-ext-%A_%a.out

set -eo pipefail

: "${AUTOGAZE_WORKTREE:=/home/stud/latn/.codex/worktrees/3f12/master-thesis/AutoGaze}"
: "${AUTOGAZE_COMMIT:?Submit with AUTOGAZE_COMMIT set to the published training commit}"

if (( SLURM_ARRAY_TASK_ID < 3 )); then
  budget=24
  report_budgets='[1,2,4,8,16,24]'
  run_prefix='r2c_fixedk24'
else
  budget=32
  report_budgets='[1,2,4,8,16,24,32]'
  run_prefix='r2c_k32_reproduction'
fi
seed_index="$((SLURM_ARRAY_TASK_ID % 3))"
base_seed="$((440829 + seed_index))"
continuation_seed="$((540829 + seed_index))"
stage1_run="${run_prefix}_stage1_seed${base_seed}"
stage2_run="${run_prefix}_stage2_base${base_seed}_seed${continuation_seed}"
stage3_run="${run_prefix}_stage3_base${base_seed}_seed${continuation_seed}"

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd "$AUTOGAZE_WORKTREE"

observed_commit="$(git rev-parse HEAD)"
if [[ "$observed_commit" != "$AUTOGAZE_COMMIT" ]]; then
  echo "Commit mismatch: $observed_commit != $AUTOGAZE_COMMIT" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked AutoGaze worktree is dirty" >&2
  git status --short >&2
  exit 4
fi

for run in "$stage1_run" "$stage2_run" "$stage3_run"; do
  if [[ -e "outputs/human_gaze/grpo/$run" ]]; then
    echo "Refusing to overwrite existing run: outputs/human_gaze/grpo/$run" >&2
    exit 5
  fi
done

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MASTER_PORT="$((62600 + SLURM_ARRAY_TASK_ID))"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage1 \
  task.exact_budget="$budget" \
  task.report_budgets="$report_budgets" \
  trainer.seed="$base_seed" \
  trainer.exp_name="$stage1_run"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage2 \
  task.exact_budget="$budget" \
  task.report_budgets="$report_budgets" \
  trainer.seed="$continuation_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/${stage1_run}/checkpoint_latest_gaze" \
  trainer.exp_name="$stage2_run"

python -m autogaze.train \
  --config-name av_gaze_stavis_human_coverage_grpo_no_kl_fresh20k_stage3 \
  task.exact_budget="$budget" \
  task.report_budgets="$report_budgets" \
  trainer.seed="$continuation_seed" \
  trainer.gaze_weights="outputs/human_gaze/grpo/${stage2_run}/checkpoint_latest_gaze" \
  trainer.resume="outputs/human_gaze/grpo/${stage2_run}" \
  trainer.exp_name="$stage3_run"
