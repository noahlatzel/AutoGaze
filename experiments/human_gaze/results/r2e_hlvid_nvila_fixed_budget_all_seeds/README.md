# R2e HLVid NVILA fixed-budget all-seed evaluation

Status: preregistered and preflighted; evidential inference pending.

This bundle supersedes the unexecuted two-random-seed utility probe for the
purpose of the September cross-benchmark evaluation. It evaluates every
available fixed-budget human-gaze checkpoint, without checkpoint selection:

- K16: six base seeds 440826--440831;
- K24: three base seeds 440826--440828;
- K36: three base seeds 440826--440828;
- K32: not scheduled until the separate three-seed training handoff exists.

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

- `preflight.json`: dataset, reference artifact, environment, protocol, and all
  12 available checkpoint fingerprints.
- `config.yaml`: resolved frozen configuration (added after completion).
- `metrics.json`: aggregate results and uncertainty (added after completion).
- `manifest.json`: code/job/artifact provenance (added after completion).
- `per_seed.csv`, `per_category.csv`, and
  `matched_budget_differences.csv`: reviewable tables (added after completion).
- `hlvid_fixed_budget_all_seeds.{png,pdf}`: final figure (added after completion).

## K32 integration point

After all three K32 endpoint checkpoints are handed off, set
`k32_integration.status` to `handed_off`, add budget key `32` to the first
three rows of the frozen seed matrix, rerun the same preflight,
and submit array tasks 0--2 only with `EVAL_BUDGETS=32`. The unchanged
aggregator will then add K32 to the matched-seed table and figure. No K32 job is
scheduled before that handoff.
