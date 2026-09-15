# R2f R2d HLVid secondary evaluation

This bundle compares each completed R2d checkpoint under its actual calibrated
variable-EOS policy and its coherent exact forced-K16 rollout. It uses the
established HLVid protocol: 128 uniform video frames, 64 full thumbnails,
`max_tiles_video=48`, 16-frame tiles, all 268 test questions, and exact answer
letter scoring. This is secondary evidence and is separate from the primary
fixed-budget comparison.

Primary descriptive estimand: equal-seed mean of macro-video exact accuracy.
Question-micro accuracy is also retained. `metrics.json` contains paired
video-bootstrap and three-seed intervals; `paired_questions.csv` preserves the
auditable within-question contrasts, while `paired_videos.csv` is the source
table for clustered comparisons.

The variable allocation/resource fields, when available, come from a validated processor-only
replay over the same source videos; the replay makes zero NVILA generation
calls and reproduces an uninterrupted VARIABLE preflight exactly. Legacy
process-exit counters are not accepted unless their observation coverage is
explicitly complete. Forced-K16 action and recovered-patch lengths are nominal
from the coherent exact-K contract; unavailable legacy per-question context
fields remain null rather than being inferred.

Allocation status in this bundle: `validated_replay_complete`.
The processor-only replay is complete and validated. Across the three
seeds, actual variable EOS used 7.710 spatial actions and
30.839 retained patches per frame on average—51.8% below
the exact-K16 contract. Mean visual and expanded-context counts were
27217.495 and 27301.387, respectively.
Training-only calibration means were 16.091, 16.004, 16.099 against
the K16 target, so the roughly K7.7 HLVid deployment is a calibration-transfer
miss rather than evidence that the target budget was met.
Forced-K16 visual/context counts are unavailable in the preserved legacy QA, so
no visual-token ratio or end-to-end speedup is inferred. The primary
variable-minus-forced accuracy difference was -1.75
percentage points; both retained uncertainty views are reported in
`metrics.json`.

Slurm job 1705996 consumed
14.31 allocated A40-hours across its preserved parent attempts
(node17 preempted 10.83 h, node15 completed 3.48 h). This is allocated lane wall time, not measured CUDA utilization.
Exact accounting rows and the admission/node-relaxation receipts are under
`scheduler/`.

Seed t95 intervals describe variation across three independently trained paired
seeds. Video intervals resample the same 77 video clusters jointly across both
policies and all seeds (10,000 replicates; 90% percentile); they do not resample
seeds. Question-micro sensitivity weights each sampled video by its question
count. These are separate uncertainty views, not a joint population interval.

This is a deployment-policy contrast, not an isolated allocation experiment:
stopping at EOS also changes which later spatial actions are emitted. The
earlier R2d validation analysis isolates allocation only by holding the common
forced-K36 spatial ordering fixed and comparing actual against shuffled
lengths. Its small allocation signal does not override the observed in-domain
coverage degradation or require a positive HLVid story.
