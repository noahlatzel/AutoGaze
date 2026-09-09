# WP5 three-strand thesis evidence package

This package integrates three complementary component studies: within-encoder token merging, temporal frame selection, and within-frame gaze selection. Each study preserves its source metric and uncertainty definitions, including any missing primary-metric declaration. Unrelated metrics are not forced onto one axis. Independent resource savings are not multiplied, and this package does not claim an integrated end-to-end system.

Verified inputs comprise the complete R2e HLVid fixed-budget gaze evaluation, curated existing in-domain gaze evidence, the present-day HLVid decode audit (WP0), corrected historical temporal evidence (WP1), and the completed DINO encoder comparison (WP2). The overall package remains partial only because K16 causal controls and refreshed variable-budget/replay results remain pending or partial.

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

Historical exploratory observations support explicitly bounded descriptive claims;
they are not promoted to confirmatory evidence by completing their curation.
The temporal source does not declare a primary metric. Its curated figures use
video-macro QA over 77 equally weighted videos; question-micro exact match over
268 questions is retained separately. Accordingly, temporal `primary` fields
remain empty. Frames, visual tokens and timing are question means, and the
explicit `bootstrap_seed` column records resampling RNG identities without
implying training-seed replication.

The temporal addendum preserves all 13 compact source files byte for byte and
records the hashes and pinned repository links for all 30 source-bundle files
(29 manifest outputs plus the manifest itself). All 18 paired contrasts remain
available with their original 90% intervals and validity labels. The four
published PDF/SVG/PNG figure triplets are linked in `figures.csv`; assets are not
duplicated or regenerated. See `addenda/temporal/manifest.json` for source commit
`9b26feb3f53c9c2bce131f02c8bdf363b00bf315`, executed curator
`50c1949819efbaf25338c0f3d62c5180ff581847`, and the superseded bundle identity.

The learned temporal policy used maximum spatial tile budget 1, whereas its
uniform/fixed-gap QA references used 48. Those QA contrasts are resolution
confounded. Its single realized schedule across all 77 videos is a separate
trajectory finding. Exposed threshold sweeps remain exploratory, and historical
cache-stratified timing does not establish a controlled speedup. T0d execution
and curation lineage is retained in the source manifest; its distortion results
are not relabeled as QA evidence.

The WP0 audit found no unusable video, requested-read failure, metadata-count
correction or output substitution across 77 videos and 268 question rows.
`addenda/wp0_decode/manifest.json` retains all five source-file hashes. This
qualifies present-day requested decoding for reuse; it does not retrospectively
certify historical inputs or establish live model-execution behavior.

The DINO addendum preserves 18 compact source files byte for byte and the hashes
of all 30 published bundle files (29 outputs plus the manifest). Result commit
`1accff45ea28f1694b090e3d46c64af2d42a8512`, curator
`1b1c8f7a0c692fcbbdbda0e4c440d1f8d159fe98`, raw publication
`af16dfe712f9efdeca280169ccc815c5da516a4e` and executed benchmark
`796ee4f074cf38f2259ead32bf9deb0566331a6e` retain distinct provenance roles.
The three published figure triplets are linked without regenerating assets.

DINO fidelity averages 16 frames within each video, two videos within each
source, then six sources equally. The six human training endpoints use base
IDs 440826--440831, mapped to continuation training seeds 540826--540831 in
`addenda/dino/manifest.json`. The 16 random-mask RNG seeds are
2026090900--2026090915. Bootstrap RNG 20260909 identifies a third, separate
randomization: whole-video draws within fixed sources, shared across arms.
The post-hoc 90% intervals condition on all observed training endpoints and
mask seeds; the three contrasts are unadjusted. Human mean differences against
pretrained and random controls have intervals spanning zero; the center
contrast is below zero under this conditional descriptive analysis. Lower CLS
distance means closer encoder representation, not higher downstream QA.

Analytical encoder and matching FLOPs stay separate, with FMA=2 and full dense
patch embedding paid by every arm. Final patch count does not determine compute:
cumulative ToMe changes the layer schedule, whereas temporary spatial merging
restores the full grid after every block. Batch-16 timing is amortized over
16-frame input calls across all 12 videos. The separate batch-one panel uses
two predeclared videos and only random-mask seed 2026090900, with no population
interval. ToMe is faster than dense under the measured batch-16 protocol and
slower under batch one. K16 uses per-image ragged sequences internally. These
are observations about the executed implementations, not a general method-speed
ranking. Selector inference, preparation, transfers and downstream heads are
excluded from encoder timing.

The 84 existing selector-export observations are retained separately. Their
single-pass, no-warmup protocol and own environment do not constitute a
controlled latency benchmark, and their times are not added to encoder timings.
The conceptual intervention figure now links verified wrapper assets at
`b14792a8c50e614770ad4921d7a9d978571ce075`; its six source/asset hashes are recorded
in `addenda/conceptual/manifest.json`. It contains no empirical finding or
evaluated combined pipeline.

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
