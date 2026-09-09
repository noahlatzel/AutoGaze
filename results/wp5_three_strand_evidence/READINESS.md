# Implementation and evidence readiness

The overall WP5 package is partial. Completed curated evidence and implementation
readiness have distinct claim scopes; execution completion alone adds no pending
numerical result to metrics.csv.

## Completed protocol audit (WP0)

The present-day requested-decode audit covered all 77 videos and 268 questions.
No requested-read failures, unusable videos, metadata-count corrections, output
substitutions or affected question rows were observed. All five compact source
files are hash-verified through addenda/wp0_decode/manifest.json. Historical
certification remains false. Protocol settings are not live model-execution evidence.

## Completed historical temporal curation (WP1)

Published noahlatzel/video-frame-selection commit
`9b26feb3f53c9c2bce131f02c8bdf363b00bf315` contains the corrected bundle executed
by curator `50c1949819efbaf25338c0f3d62c5180ff581847`. The independent audit verified
29 output hashes (30 files including the manifest), all 18 paired contrasts and
all four figure triplets. No new inference was run. WP5 preserves compact source
bytes and links published figures; it does not rerun bootstrap or rendering.
The superseded bundle is identified in the addendum and wrapper `b14792a`.

The source declares no primary metric. Video-macro QA is the curated comparison
axis; question-micro QA is retained separately. Costs are question means, and
bootstrap RNG seeds are not training seeds. Learned MTV1 QA contrasts against
MTV48 references are confounded; fixed-schedule collapse is supported directly
by the trajectories. Exposed thresholds remain exploratory and historical
mixed/cache-stratified timing cannot establish controlled speedup. T0d execution
and curation lineage is preserved without relabeling distortion as QA.

## Completed DINO evidence (WP2)

Published result commit `1accff45ea28f1694b090e3d46c64af2d42a8512` contains the
completed curation from `1b1c8f7a0c692fcbbdbda0e4c440d1f8d159fe98`, based on raw
publication `af16dfe712f9efdeca280169ccc815c5da516a4e` and executed benchmark
`796ee4f074cf38f2259ead32bf9deb0566331a6e`. All 29 output hashes, the 30-file
bundle inventory and three figure triplets were verified. The curator passed
six focused CPU tests and reproduced 14 packaged metric arms. No WP5 bootstrap,
inference or figure generation was performed.

The outcome-independent panel has 12 validation videos, two per source, with
16 frames each. Six human training endpoints and 16 random-mask seeds stay fixed
under the conditional paired video bootstrap. The three post-hoc contrasts are
unadjusted. Human versus pretrained/random intervals span zero; the center
contrast is below zero in this descriptive analysis. Claims remain about
representation fidelity, not downstream accuracy.

Analytical encoder and matching FLOPs remain separate. Batch-16 timing covers
all 12 videos; batch-one timing covers two predeclared videos and one random-mask
seed. The batching reversal is retained, and no end-to-end claim is supported.
The 84 single-pass exporter observations are preserved separately as operational
evidence, not a controlled selector-latency benchmark.

Export used AutoGaze `a8239fce8528335ca147adfd23e246ca06bdc7dc`, with exact-zero-only
historical-buffer compatibility. Prior validation was 36 core exporter tests
plus 12 focused compatibility tests; benchmark validation was 57 prior tests
plus three focused central-control tests. These describe separate test scopes,
not fresh combined totals.

## Controls and refreshed variable budget remain incomplete

K16 controls run from frozen AutoGaze
`b795de084001c3a8ef1c147bc850d7764c867cb3` under Slurm 1700678; results are pending.
Variable-budget job 1690935 is partial, without a final aggregate. Allocation
replay job 1700681 depends on completion. No partial-run numerical aggregate is
promoted. Existing R2d in-domain evidence remains historical.

Integration `d649ad05c15b8f58edd0990140e05c23a4b9c183` had 109 native Windows CPU
passes and one Linux-directory-fsync limitation. Production code and tests were
preserved. The parent reports a durable Linux receipt at
`f3487e112f36699407587d9b9d5eb4e8f990016b`: 12 tests passed in 18.37 seconds,
including the exact resume/attestation test before the frozen control commit.
This is not 110 native Windows passes and does not imply experimental completion.

## Conceptual intervention figure

The published wrapper figure at `b14792a8c50e614770ad4921d7a9d978571ce075` includes
PDF, SVG, PNG, JSON, source and README. All six pinned Git blob hashes were
verified and linked through `addenda/conceptual/manifest.json`. The main
token-merging illustration is cumulative ToMe; temporary spatial_block restores
the grid after each block and is not final-K16. The illustration adds no empirical
claim or evaluated combined pipeline.
