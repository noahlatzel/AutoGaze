#!/usr/bin/env bash
#SBATCH --job-name=gaze-supervised-k16-restart
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%1
#SBATCH --no-requeue
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-supervised-k16-restart-%A_%a.out

set -euo pipefail
# The sole owner supplies the new immutable source/run IDs. The shared launcher
# still refuses existing seed directories and serializes both original phases.
export SUPERVISED_EXECUTION_CONFIG=experiments/human_gaze/configs/supervised_k16_restart_execution.yaml
exec bash experiments/human_gaze/slurm/run_supervised_k16_comparison_array.sbatch
