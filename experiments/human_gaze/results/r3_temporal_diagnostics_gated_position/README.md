# R3 temporal diagnostics and gated temporal position

This bundle contains the compact evidence for both phases of the preregistered
R3 gate.
The gaze targets have clear temporal continuity, and all six completed K16
policies depend strongly on causal visual history. The learned policies move
their selection centroids less than the ground-truth gaze centroids on five of
six sources.

The isolated temporal-position treatment is **inconclusive**. At the fixed
10,000-update validation endpoint all three paired seeds improved, but only by
`+0.001851`, `+0.000235`, and `+0.000399`; the paired mean `+0.000828` is far
below the preregistered `+0.005` pass threshold. Its matched-curve AUC mean is
`-0.000087`, so the small endpoint advantage was not sustained throughout
training. The scalar gate remained norm-controlled (maximum absolute
signal/feature RMS `0.000677` over all logged rows), no source collapsed, and every treatment seed
remained strongly dynamic relative to Center-16 and its learned static Top-16.
History sensitivity and centroid-motion behavior are effectively unchanged.

`phase2_metrics.json` contains endpoint seed values, paired uncertainty,
per-source effects, center/static/shuffled controls, and count-weighted
history/motion comparisons. `phase2_verification.json` proves the fixed update
grid, paired config invariants, controlled norm, and unopened test split.

Heavy per-source distributions remain in the ignored artifact directory listed
in `manifest.json`. Interpretation and figures live in the linked
`wiki_gazing/Experiments/R3 Temporal Diagnostics and Gated Temporal Position.md`.
