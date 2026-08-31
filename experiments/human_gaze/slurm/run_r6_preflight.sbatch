#!/usr/bin/env bash
#SBATCH --job-name=gaze-r6-preflight
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r6-preflight-%j.out

set -eo pipefail

source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r6-recurrent

export HF_HOME=/storage/slurm/latn/data/huggingface_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=5
export MASTER_PORT=50106

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked R6 worktree is dirty" >&2
  exit 2
fi

smoke_run="outputs/human_gaze/grpo/r6_recurrent_state_smoke_2step"
if [[ -e "${smoke_run}/config.yaml" ]]; then
  echo "Refusing to overwrite completed R6 smoke output" >&2
  exit 3
fi

python -m pytest -q \
  tests/test_recurrent_trajectory_state.py \
  tests/test_r6_run_manifests.py \
  tests/test_fine_only_generation.py

python scripts/human_gaze/check_r6_zero_gate_identity.py \
  --output-json outputs/human_gaze/r6_recurrent_state_smoke/zero_gate_identity.json

python -m autogaze.train \
  --config-name av_gaze_stavis_r6_recurrent_state \
  trainer.seed=640826 \
  trainer.gaze_weights=outputs/human_gaze/grpo/r2b_fresh20k_stage3_base440826_seed540826/checkpoint_latest_gaze \
  trainer.exp_name=r6_recurrent_state_smoke_2step \
  trainer.max_train_steps=2 \
  trainer.val_nsteps=100 \
  trainer.save_nsteps=100 \
  trainer.n_epochs=1 \
  trainer.validate_at_start=false \
  trainer.skip_final_validation=true

python scripts/human_gaze/verify_r6_smoke.py \
  --training-metrics "${smoke_run}/training_metrics.jsonl" \
  --output-json outputs/human_gaze/r6_recurrent_state_smoke/two_update_verification.json
