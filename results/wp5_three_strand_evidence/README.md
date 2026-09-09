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

`seed_id_role` distinguishes base-seed identifiers from continuation training
seeds. R2d retains its original base IDs in the CSVs; the in-domain addendum
manifest maps them to the actual training seeds 740826--740828. Deterministic
Center16 and Prior16 evaluations have no training-seed count, so their
`n_seeds` fields are empty and explicitly marked inapplicable.

Variable-budget resource rows distinguish the source-balanced mean number of
fine actions from the mean over pooled validation frames. Neither number is
measured encoder cost or latency. The source-balanced calibration band is
`[15.5,16.5]`; R2d's source-balanced mean K is 15.155946 and its pooled frame
mean K is 14.571282. Both are preserved as historical source observations.
The dataset hash and seed mapping are attributed to the original R2d
`verification.json`, linked in the in-domain addendum manifest.

The fixed-budget publication figures are available in `figures/hlvid_fixed/`.
They use exactly the three common seeds at K16, K24, and K32, retaining primary
question-micro QA and a separate secondary aggregation sensitivity. The renderer
reads preserved compact tables and writes new WP5 assets; it never invokes the
heavy aggregator or writes into the completed experimental bundle.
`figures.csv` records exact repository-root regeneration commands, source tables,
formats, captions, and validation metadata. See `FIGURE_REGENERATION.md` for the
optional plotting environment and reproduction notes.
