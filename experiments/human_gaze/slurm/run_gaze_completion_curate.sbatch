#!/usr/bin/env bash
#SBATCH --job-name=gaze-curate
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/user/latn/slurm/logs/gaze-curate-%j.out
set -euo pipefail
: "${GAZE_COMPLETION_WORKTREE:?Supply frozen clean source checkout}"
: "${GAZE_COMPLETION_ROOT:?Supply new append-only artifact run directory}"
export GAZE_COMPLETION_RUN_ID="${GAZE_COMPLETION_RUN_ID:-$(basename "$GAZE_COMPLETION_ROOT")}"
cd "$GAZE_COMPLETION_WORKTREE"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=5
runner=(uv run --no-project --python /home/stud/latn/miniconda3/envs/autogaze/bin/python python)
case "${1:?Specify test or supervised}" in
  test)
    "${runner[@]}" scripts/human_gaze/complete_gaze_test.py curate \
      --actions "$GAZE_COMPLETION_ROOT/test_actions" \
      --output "$GAZE_COMPLETION_ROOT/test_curation"
    ;;
  supervised)
    "${runner[@]}" scripts/human_gaze/curate_supervised_k16_comparison.py \
      --config experiments/human_gaze/configs/supervised_k16_restart_analysis.yaml \
      --action-root "$GAZE_COMPLETION_ROOT/supervised_actions" \
      --quality-only --output-dir "$GAZE_COMPLETION_ROOT/supervised_curation"
    "${runner[@]}" scripts/human_gaze/plot_supervised_k16_comparison.py \
      --metrics "$GAZE_COMPLETION_ROOT/supervised_curation/metrics.json" \
      --output-dir "$GAZE_COMPLETION_ROOT/supervised_curation/figures"
    ;;
  *) echo "Mode must be test or supervised" >&2; exit 2 ;;
esac
