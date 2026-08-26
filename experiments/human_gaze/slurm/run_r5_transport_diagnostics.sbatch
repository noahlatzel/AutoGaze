#!/usr/bin/env bash
#SBATCH --job-name=gaze-r5-transport-diagnostic
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-r5-transport-diagnostic-%j.out

set -eo pipefail
source /home/stud/latn/miniconda3/etc/profile.d/conda.sh
conda activate autogaze
cd /storage/user/latn/worktrees/autogaze-r5-analysis
export OMP_NUM_THREADS=5

python scripts/human_gaze/diagnose_feature_transport.py \
  --config experiments/human_gaze/configs/r5_causal_feature_transport.yaml \
  --seeds 440826 440827 440828 \
  --batch-size 16 \
  --num-workers 4 \
  --output-json outputs/human_gaze/r5_feature_transport_diagnostics/transport_prior.json \
  --output-plot /home/stud/latn/master-thesis/wiki_gazing/assets/r5_feature_transport_prior.png
