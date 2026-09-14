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

Allocation status in this bundle: `unavailable_pending_validated_replay`. An accuracy-only
bundle contains all three complete QA pairs but **does not establish actual
variable cost or efficiency**. Missing variable values are null, not K16,
calibration lengths, tail-only process means, or zero. A later validated replay
will be published as a separate immutable joint bundle, preserving this stage.

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
