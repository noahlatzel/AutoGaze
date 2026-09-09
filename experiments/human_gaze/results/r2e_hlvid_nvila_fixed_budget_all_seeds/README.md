# R2e HLVid NVILA fixed-budget all-seed evaluation

Status: complete and independently validated.

This bundle supersedes the unexecuted two-random-seed utility probe for the
purpose of the September cross-benchmark evaluation. It evaluates every
available fixed-budget human-gaze checkpoint, without checkpoint selection:

- K16: six base seeds 440826--440831;
- K24: three base seeds 440826--440828;
- K32: three fixed-20k replications for base seeds 440826--440828;
- K36: abandoned by user direction; one 260/268 prefix is preserved outside
  primary reporting.

## Frozen protocol

The comparability invariant is **`max_tiles_video=48` (MTV48)**. It is not a
48-frame temporal input. The preserved upstream HLVid artifact records 128
uniformly sampled tiled frames, 64 full thumbnail frames, MTV48, bfloat16
NVILA-8B-HD-Video, greedy generation, and exact standalone option-letter
scoring. This evaluation keeps each of those settings. The human-gaze policy
uses exactly K unique fine actions (IDs 69--264) per AutoGaze frame with EOS
disabled.

HLVid's official Hugging Face split is named `test` and contains 268 questions
on 77 videos. This explicitly authorized cross-benchmark evaluation does not
read the separate protected 663-clip AV-gaze-STAViS human-gaze test split.

The primary metric is official question-micro exact-match accuracy. Secondary
reporting includes macro-video and category accuracy. All-seed endpoints use a
90% Student-t interval across seeds. Cross-budget comparisons use only the
three matched seeds and a 10,000-draw video-cluster bootstrap. The upstream
AutoGaze reference is descriptive, not budget- or seed-matched.

## Files

- `preflight.json` and `preflight_k16_k24_k32.json`: original and K32-extension
  protocol, dataset, environment, and checkpoint fingerprints.
- `config.yaml`: resolved execution configuration.
- `metrics.json`: aggregate results, validation record, and uncertainty.
- `manifest.json`: code, job, checkpoint, and heavy-artifact provenance.
- `admission_ledger_20260909T013942+0200.json`: shared Slurm admission and
  submitted-control/replay receipt.
- `linux_validation_receipt_b795de0.json`: original Linux pass evidence for
  the exact restart-safe question-zero preflight test at the immutable control
  source commit; this closes the later Windows platform-specific failure
  without rerunning the unchanged test.
- `scheduler_node_relaxation_receipt_20260909T124648+0200.json`: before/after
  provenance for clearing only the live node17 pin on pending same-A40 R2f and
  replay jobs, including the rejected batched syntax and successful individual
  commands. Immutable launchers still record the original node17 directive;
  final curation records every actual allocation segment's node from Slurm.
- `control_a40_replacement_receipt_20260909T190415+0200.json`: explicit user
  approval, zero-work H100 cancellation, immutable A40 replacement submission,
  per-element same-A40 node-pin relaxation, full-protocol VRAM evidence, and
  the descriptive cross-hardware QA/latency claim boundary.
- `per_seed.csv`, `per_category.csv`, and
  `matched_budget_differences.csv`: reviewable tables.
- `hlvid_fixed_budget_all_seeds.{png,pdf}`: final figure.

## Results

| Budget | Seeds | Mean question-micro accuracy | 90% t interval |
|---:|---:|---:|---:|
| K16 | 6 | 0.5031 | [0.4941, 0.5121] |
| K24 | 3 | 0.4789 | [0.4550, 0.5027] |
| K32 | 3 | 0.4913 | [0.4711, 0.5115] |

On matched base seeds 440826--440828, the means are 0.4975, 0.4789, and
0.4913 for K16, K24, and K32. The video-cluster-bootstrap differences are:

| Contrast | Difference | 90% interval |
|---|---:|---:|
| K16 - K24 | +0.0187 | [-0.0194, +0.0537] |
| K16 - K32 | +0.0062 | [-0.0583, +0.0672] |
| K32 - K24 | +0.0124 | [-0.0300, +0.0606] |

All three intervals include zero. The evidence therefore does not establish a
reliable fixed-budget ordering or monotonic benefit from increasing K. K16 has
the highest point mean, while K32 recovers the K24 drop, but the cross-seed and
video-cluster uncertainty is too large for a stronger claim. The preserved
upstream AutoGaze reference is 0.4851 and remains descriptive rather than a
budget- or seed-matched statistical baseline.

All 12 primary streams contain exactly 268 unique ordered questions and passed
recomputation against the checksummed parquet, exact 128-frame sampling rule,
answer parser, correctness, summary hashes, checkpoint identities, and frozen
protocol. No protected human-gaze test data was accessed. K36 remains excluded
from all primary summaries, comparisons, and figures.
