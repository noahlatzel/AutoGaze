# K16-trained human-gaze policies evaluated at K8 on HLVid

## Result

Six fixed-20k K16-trained policies evaluated at K8 obtain **49.19% ± 0.59
percentage points** (population SD across training seeds). Historical normal
AutoGaze obtains **48.51%** (130/268). The paired mean difference is **+0.68
percentage points**, with a post-hoc 90% video-cluster bootstrap interval of
**[−2.28, +3.33] points**. This does not establish improved QA accuracy or
statistical equivalence. No new training or inference was performed for curation.

| Base seed | Correct / 268 | Accuracy | Wins / losses against reference |
|---|---:|---:|---:|
| 440826 | 133 | 49.63% | 16 / 13 |
| 440827 | 131 | 48.88% | 13 / 12 |
| 440828 | 132 | 49.25% | 16 / 14 |
| 440829 | 129 | 48.13% | 12 / 13 |
| 440830 | 132 | 49.25% | 16 / 14 |
| 440831 | 134 | 50.00% | 16 / 12 |

## Protocol and comparability

All 268 official HLVid test questions over 77 videos; frozen NVILA-8B-HD-Video,
128 uniformly sampled frames, 64 full thumbnails, max_tiles_video=48, tile
length 16, BF16, greedy answers of at most 16 tokens. Each policy emits exactly
eight unique fine actions per inference crop, with EOS disabled. Checkpoints
are fixed endpoints, not selected by HLVid performance. Human-gaze protected
test was not accessed.

The paired reference is the preserved normal-AutoGaze results stream, not the
pretrained exact-K16 control. Question IDs, video paths, answers, categories,
recorded model path, frame/thumbnail/tile settings, dtype and batch sizes agree.
The historical reference lacks exact decoded-frame traces, model/runtime
fingerprints and visual-token counts. Earlier protocol audits and current
legacy runner code support the intended sampling/prompt/scoring recipe, but
cannot retrospectively prove that every historical runtime setting and frame
was identical. Therefore this is a **descriptive historical comparison**, not
an isolated causal effect of human-gaze training.

K8 was chosen by the user before this evaluation. Raw K8 is verified, but an
exact end-to-end token match to the historical reference cannot be established.
The measured K8 path retains **32 patches per tile-frame after resolution
adaptation**, with **32 padded slots** and **27,904 visual tokens per question**.
Frame observations include repeated processing across tiles; these are not
counts of distinct source-video frames. No reference-relative token saving or
speedup is claimed.

## Analysis

`paired_correctness.csv` provides compact paired binary outcomes. Sampling unit
for the post-hoc interval is the video: 10,000 bootstrap draws, RNG seed 170926,
keeping all questions and all six trained policies within each sampled video.
The estimator is question-micro accuracy, not the equal-weight video average.
The six-policy set and single historical reference are fixed; this interval
does not jointly estimate training-seed uncertainty. Seed SD is reported
separately. These analysis choices follow existing HLVid curation conventions
and were fixed for this analysis after endpoint inspection, not preregistered
as a confirmatory test. There is no pass/fail decision threshold.

## Evidence and resource accounting

`manifest.json` links immutable run IDs, artifact roots, checkpoint hashes,
summary/results/evidence fingerprints and historical reference fingerprints.
`config.yaml` includes the frozen execution settings and explicitly post-hoc
analysis settings. `metrics.json`, `per_seed.csv`, `resources.csv` and
`accuracy_by_seed.{png,pdf}` are compact final evidence.

All six answer streams passed source-parquet identity, exact answer-parser
recomputation, durable completion identity, checkpoint and file hash checks,
exact 128-frame decoding, eight unique fine-action checks, no context truncation,
and measured resource-count verification in the curation script.
`verification.json` records those checks and the 40 passing targeted tests.
Durable evidence contains 1,611 started attempts, 1,608 completions, two
recorded decode failures, and one unterminated preempted attempt.

Total allocation: **67.31 H100 GPU-hours**, including two failed decode attempts,
two explicit resumes, and preemption/requeue of seed 440831. The latter's last
1:20:12 allocation is only its final segment; its earlier 10:34:20 segment is
also charged. `slurm_allocations.psv` was collected using `sacct -D` so both
segments are retained. There are 1,608 completed answer passes; unterminated
attempts may have performed additional work and must not be counted as zero-cost.
Raw artifact directories were read only during curation.

## Reproduce curation

From the AutoGaze repository, with its existing `autogaze` environment:

```bash
PYTHONPATH="$PWD" uv run --no-project \
  --python /home/stud/latn/miniconda3/envs/autogaze/bin/python \
  scripts/human_gaze/curate_hlvid_k8.py \
  --raw-root /storage/user/latn/worktrees/autogaze-hlvid-k16-at-k8/outputs/human_gaze/hlvid_nvila/r2e_k16_at_k8 \
  --reference /home/stud/latn/AutoGaze/outputs/hlvid/nvila_default_autogaze \
  --config experiments/human_gaze/configs/r2e_hlvid_k16_at_k8.yaml \
  --output /path/to/new/curation-directory
```

Copy the included `slurm_allocations.psv` into the new output directory before
running; it preserves historical scheduler accounting without depending on
Slurm retention. Inference was performed at commit `b1cf8c2`; curation integrates
that K8 allowlist, launcher and tests into the reconciled code history.
