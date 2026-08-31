#!/usr/bin/env bash
#SBATCH --job-name=gaze-var-stab-diag
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-var-stab-diag-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r2d-variable-budget

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5

run=r2d_variable_stability_v2b_base440826_seed840826
diagnostic_run=${run}_diag_v3

python scripts/human_gaze/diagnose_variable_budget.py \
  --config experiments/human_gaze/configs/r2d_variable_budget_v2.yaml \
  --checkpoint "outputs/human_gaze/grpo/${run}/checkpoint_latest_gaze" \
  --output-json "outputs/human_gaze/diagnostics/${diagnostic_run}/variable_budget_val.json" \
  --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r2d_variable_stability_v2b_diagnostics_v3.png" \
  --batch-size 8 \
  --num-workers 4

python scripts/human_gaze/check_variable_budget_stability.py \
  --history "outputs/human_gaze/grpo/${run}/validation_metrics.jsonl" \
  --diagnostics "outputs/human_gaze/diagnostics/${diagnostic_run}/variable_budget_val.json" \
  --reference-fixed-k16 0.45216500477354366
