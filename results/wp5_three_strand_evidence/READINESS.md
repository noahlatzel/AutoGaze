# Implementation and evidence readiness

These records describe implementation and test readiness only. They are not scientific findings and add no empirical metric rows.

## HLVid runtime and decode audit

AutoGaze commit `38fffdac34cee682f87a75f12af05dc6e118ac17` is published and was verified through the GitHub commit API. The parent reports 23 CPU tests passing. Actual Linux decode-audit outputs remain pending, so the implementation cannot yet certify fresh runtime behavior or historical reads.

## DINO benchmark supplement

Published `noahlatzel/video-tokenization` commit `796ee4f074cf38f2259ead32bf9deb0566331a6e` includes the fixed central K16 control in the main panel and separate batch-one timing panel. Validation comprises 57 prior core/supplement CPU tests plus three focused central-control tests; this is not a fresh combined 60-test run. Actual panel export, checkpoint equivalence checks and GPU results remain pending.

The outcome-independent DINO panel is corrected to 12 distinct validation videos, exactly two per source. Two sources have only two validation videos, so a larger distinct-video quota would be impossible without changing the population. The panel contains all 16 frames per selected clip.

Published AutoGaze exporter commit `8dbaa3f5dcf60fa7ea58d1a394f4a14f328be7e3` passed 36 focused CPU tests. It freezes panel identities and RGB hashes before selector inference, validates all seven checkpoint identities and exports fine cells against those cached RGB bytes. These checks establish implementation readiness; actual checkpoint inference and the completed RGB/mask export remain pending.

## Temporal evidence

Temporal curation is assigned to task `01a0691b-8544-7be0-ba5c-3344adf04982`. Reviewed `noahlatzel/video-frame-selection` commit `8c48a81eef2ad0af1db85260fddbddd040629f1f` keeps MTV1 and MTV48 QA resolution contracts separate and corrects cached frame-feature count interpretation. The original temporal results remain historical. Compact source-table integration and a renderer follow-up remain pending; no new temporal metric or figure is promoted by this readiness update.

## Conceptual intervention figure

The main token-merging illustration is cumulative ToMe. Temporary `spatial_block` merging is a distinct supplemental mechanism that merges before attention and restores the grid after each block; it is not a final-K16 mechanism.

These readiness statements must remain separate from final claims in `claims.csv`.
