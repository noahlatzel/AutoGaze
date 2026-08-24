#!/usr/bin/env bash
#SBATCH --job-name=gaze-r3-history
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r3-history-%j.out

set -eo pipefail
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r3-analysis
export OMP_NUM_THREADS=5

for arm in control position; do
  python scripts/human_gaze/diagnose_temporal_policy_history.py \
    --config experiments/human_gaze/configs/r3_temporal_diagnostics_gated_position.yaml \
    --seeds 440826 440827 440828 \
    --checkpoint-seed-offset 200000 \
    --checkpoint-template "outputs/human_gaze/grpo/r3b_temporal_${arm}_base{seed}_seed{continuation_seed}/checkpoint_latest_gaze" \
    --batch-size 16 \
    --num-workers 4 \
    --output-json "outputs/human_gaze/r3b_temporal_diagnostics/${arm}_policy_history.json" \
    --output-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r3b_temporal_${arm}_history_sensitivity.png" \
    --trajectory-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r3b_temporal_${arm}_trajectory_examples.png" \
    --motion-plot "/home/stud/latn/master-thesis/wiki_gazing/assets/r3b_temporal_${arm}_motion_response.png"
done
