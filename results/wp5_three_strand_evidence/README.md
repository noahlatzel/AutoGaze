# WP5 three-strand thesis evidence package

This package integrates three complementary component studies: within-encoder token merging, temporal frame selection, and within-frame gaze selection. Each study retains its declared primary metric and uncertainty unit. Unrelated metrics are not forced onto one axis. Independent resource savings are not multiplied, and this package does not claim an integrated end-to-end system.

Verified inputs currently comprise the complete R2e HLVid fixed-budget gaze evaluation and curated existing in-domain gaze evidence. Token-merging, temporal-selection, new K16 causal-control, and refreshed variable-budget handoffs remain explicitly missing.

## Files

- `manifest.json`: immutable base, package status, and evidence boundaries.
- `studies.csv`: study-level primary metrics and interpretation boundaries.
- `artifacts.csv`: artifact provenance.
- `metrics.csv`: normalized compact source data.
- `claims.csv`: claim-to-artifact and claim-to-metric index.
- `figures.csv`: source tables, formats, regeneration commands, and status.
- `HANDOFF_SCHEMA.md`: owner handoff contract.
- `STYLE.md` and `scripts/publication_style.py`: shared publication defaults.
- `addenda/`: preserved study-specific source tables and identities.

Missing values are empty, never zero. Provisional, partial, snapshot, unsupported, missing, and historical-exploratory evidence cannot silently support a final empirical claim.
