# Hierarchical Sampling Experiments

This family tests top-down temporal partitioning for HLVid videos. The goal is
to avoid action-sequential frame skipping by first building a set of coherent
temporal leaves, then passing the resulting 16-frame leaves to the normal
AutoGaze/NVILA path.

The first stage is dry-run only:

- sample a 16-frame representative tile from each interval,
- score interval incoherence with a window metric,
- split high-incoherence intervals,
- record leaf counts, selected frame counts, depths, and scores.

The low-resolution thumbnails are only used for scoring the hierarchy. A
promoted NVILA run should use the selected source-frame indices to reload frames
from the original video and then pass them through the normal AutoGaze/NVILA
processor.

This is intentionally separate from `proxy_sampling`, whose current runner
selects the next temporal tile sequentially.

## Current Runner

```bash
python scripts/runners/run_hierarchical_coherence_dry_sweep.py \
  --dataset-root /storage/slurm/latn/data/AutoGaze/HLVid \
  --coherence-metric semantic_diversity \
  --semantic-model dinov2 \
  --target-leaf-tiles 16 \
  --output outputs/hierarchical_sampling/hlvid_hierarchical_coherence_dry_v0/results.jsonl \
  --summary-output outputs/hierarchical_sampling/hlvid_hierarchical_coherence_dry_v0/summary.json
```

## Promotion Criterion

Promote a configuration to full NVILA only after the dry run shows a sensible
frame budget and a non-degenerate split pattern. Compare promoted runs against
uniform sampling at the same number of selected frames or visual tokens.
