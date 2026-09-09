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

## DINO execution complete, curation pending (WP2)

The owner reports completion of the real-weight experiment at video-tokenization
`796ee4f074cf38f2259ead32bf9deb0566331a6e`, including cumulative ToMe, temporary
spatial_block and central K16. Its outcome-independent validation panel contains
12 distinct videos, two per source, with 16 frames per clip. Final numerical
curation, claim review and the DINO addendum remain pending.

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

The main token-merging illustration is cumulative ToMe. Temporary spatial_block
merges before attention and restores the grid after each block; it is not a
final-K16 mechanism. The illustration adds no empirical claim.
