# Completed HLVid control and paired variable accuracy index

This additive index uses published AutoGaze main
`473662d93d9005becb5e996e92b87ef6fa8b6368`. It adds reviewed control result
`51b41662b801afb4b3ec3ae870b80403ce3abcb3` and variable paired accuracy result
`03cd606257b9f3f28ac295e62533453b2e2b2612`, without scientific recomputation,
new model calls, rendering or training. Source hashes are checked against the
published result commits; raw QA, video, checkpoint and hardware hashes are
preserved from the owner manifests, not freshly recomputed here.

## Controls: question-micro remains primary

Two deterministic controls complete 536 answers. The comparison reuses all six
trained K16 endpoints (1,608 answers) unchanged. Mean primary question-micro
accuracy is 0.5031094527363185 for trained K16, 0.5 for pretrained and
0.5111940298507462 for tile-local Center16. Trained-minus-control primary
differences are +0.003109452736318406 and −0.008084577114427862. Both primary
seed90 intervals and both paired-video90 intervals include zero; superiority
and equivalence are not established. Macro-video remains sensitivity only,
even where its seed interval excludes zero. Deterministic controls have no
training-seed count; one evaluation is not one independently trained seed.

Seed uncertainty is 90% Student t over six independent trained endpoints, df5.
Video uncertainty resamples 77 paired video clusters, conditional on those
six endpoints, with 10,000 replicates. The configured bootstrap base seed is
20260909. The source curator uses effective RNG seeds 20261909 for the aggregate
pretrained contrast and 20261910 for the aggregate Center16 contrast. Each
per-endpoint contrast uses 20260909 + endpoint base ID + 100 times the zero-based
control index (pretrained 0, Center16 1). `metrics.csv` records these effective
seeds, preserving the source intervals without drawing new samples. Per-endpoint
video intervals remain available. These are separate uncertainty views, not
one combined interval. Base IDs 440826–440831 map to continuation training seeds
540826–540831; bootstrap RNG is neither kind of training seed.

QA comparisons across the trained H100 NVL and control A40 executions were
explicitly approved. Latency remains hardware-specific. The control source
retains actual A40 allocation nodes/segments, preemptions, host-memory fields,
and per-question acquisition/selector/encoder/generation telemetry. Reserved
A40 seconds (66,309 pretrained; 56,486 Center16) are not CUDA utilization or
generation latency. Host MaxRSS is not peak VRAM. Resumed process summaries
with incomplete counter coverage are not full-arm cost estimates. No reference
rerun was performed solely to match hardware or add historical telemetry.

## Variable deployment: video-macro remains primary

All three paired trained endpoints complete 1,608 answers. Mean primary
video-macro QA is 0.4829547436690294 under variable EOS and 0.5004150611293469
under coherent forced K16. The difference is −0.017460317460317436; its video90
interval is [−0.05562770562770563, +0.017536075036074985] and its seed95 interval
is [−0.05318752970278, +0.018266894782145122]. All three primary seed differences
are negative. This supports a bounded negative descriptive deployment contrast,
not a definitive population effect. Question-micro sensitivity remains separate;
its negative seed95 interval does not override primary macro or video uncertainty.

Seed uncertainty is 95% Student t over three paired endpoints, df2. The video90
bootstrap uses RNG 20260901, 10,000 shared video draws across both policies and
all three endpoints, and does not resample seeds. The micro sensitivity weights
each sampled video's question count. Base IDs 440826–440828 map to R2d training
seeds 740826–740828. Endpoints were fixed at 20,000 updates; EOS calibration
used 48 source-balanced training clips, not HLVid answers. No new checkpoint
selection, training, calibration or answer generation occurred during curation.

This is not allocation-only: EOS stopping changes which later spatial actions
are emitted. The earlier in-domain shuffled-length comparison holds a common
forced-K36 ordering fixed and answers a different question; its small signal
cannot rescue the worse joint selector or turn this deployment contrast into
an allocation effect.

Actual HLVid variable actions, recovered patches, visual tokens and expanded
context counts remain null. Forced K16's 16 fine actions and 64 recovered
patches are nominal exact-contract values, not measured variable cost. Forced
expanded-token/context counters also remain unavailable. Calibration lengths,
resumed process-local tail means, A40 capacity and reservation walltime are not
full-benchmark efficiency measurements. No resource saving or speedup is claimed.
Actual allocation nodes and every accounting segment are preserved in the
source scheduler tables. Original replay 1700681 failed before observations.

Both additions reuse the official HLVid test (77 videos, 268 questions), which
has historical exposure. This is not a newly unseen benchmark. The separate
human-gaze test was not accessed by these curations. The current decode
audit qualifies requested decoding, but does not certify all historical reads.

## Source tables, figures and provenance

`r2e_controls/manifest.json` and `r2f_variable_accuracy/manifest.json` retain
every published source-file SHA256, byte count, Git blob and pinned link.
Twenty-six compact source files are copied byte for byte, including resolved
configs, metric JSON, seed/video tables, telemetry and actual scheduler
segments. The already published question-level paired audit stays linked,
not duplicated. Checkpoint/calibration/code/data identities are retained in
the copied owner manifests; no heavy inputs are rehashed by WP5.

`../tables.csv` inventories tables without forcing different metrics onto one
axis. `../figures.csv` links the three published PDF/PNG figure sets and source
captions. These bundles contain no SVG exports: SVG fields stay empty. Fixed
budget, temporal, DINO and conceptual figures remain unchanged. No WP5 rendering
or bootstrap was run for this addition.

## Exact index regeneration and verification

From an AutoGaze checkout with the committed index files:

```text
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-project python results/wp5_three_strand_evidence/scripts/index_hlvid_completion.py --check
```

The standard-library checker verifies source Git blobs/hashes, all 305 original
metric rows, unique metric IDs, claim references, copied source bytes, explicit
missing rows and every protected original file. It loads no model or dataset,
and invokes no scheduler. `hlvid_completion_validation.json` preserves its
reviewed initial output. `hlvid_completion_metadata_review.json` records the
independent Git-blob preservation and effective-RNG correction review; it does
not claim a new Linux checker run or scientific recomputation. The checker is
required index-behavior verification, not a
fresh scientific test or raw QA certification.

To reconstruct the additive data from the exact base, make an isolated worktree
at `473662d…`, transfer this script there, and run:

```text
uv run --no-project python results/wp5_three_strand_evidence/scripts/index_hlvid_completion.py --patch
uv run --no-project python results/wp5_three_strand_evidence/scripts/index_hlvid_completion.py --source-copy-commands
```

The first command emits an apply_patch transfer for manifests and appended CSV
rows; apply it in that isolated checkout. The second emits exact `cp` commands
for preserving compact source bytes, including original CRLF line endings.
Execute those commands only in the isolated destination, then apply the focused
commit's presentation/readiness addenda and run `--check`. Original completed
source bundles and protected figure files are never regeneration destinations.

## Optional original figure/table reproduction (not executed by WP5)

Original control curation, including raw validation, bootstrap and plotting:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-project /home/stud/latn/miniconda3/envs/vila-autogaze-eval/bin/python scripts/human_gaze/curate_hlvid_k16_controls.py --config experiments/human_gaze/configs/r2e_hlvid_k16_causal_curation_v1.yaml --output-dir artifacts/wp5/control-reproduction-NEW_RUN_ID
```

Run at control result commit `51b4166`, using the owner's durable Linux inputs.
The curator refuses an existing output destination. Variable reproduction at
`03cd606`, including its original bootstrap and plotting, is:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-project /home/stud/latn/miniconda3/envs/vila-autogaze-eval/bin/python scripts/human_gaze/curate_r2d_hlvid_secondary.py --run-id 20260901-0156_r2f-r2d-hlvid-secondary_ecf1535 --curation-run-id WP5_REPRODUCTION_NEW_RUN_ID --artifact-root /storage/user/latn/artifacts/autogaze-r2f --accuracy-only --config experiments/human_gaze/configs/r2f_r2d_hlvid_secondary.yaml --output-dir artifacts/wp5/variable-reproduction-NEW_RUN_ID --code-commit ecf1535617250b706528b4b5ca278fccfbd85f4c --scheduler-provenance-dir results/human_gaze/20260914-0202_r2f-hlvid-complete-paired-accuracy_6b4eef7/scheduler --supporting-processor-smoke results/human_gaze/20260914-0202_r2f-hlvid-complete-paired-accuracy_6b4eef7/processor_initialization_smoke.json
```

Choose new concrete run IDs and fresh destinations before optional reproduction;
never overwrite the published source bundles. These commands make no new QA
calls, but do repeat the owner analysis and plotting, unlike the index-only
verification above. No reproduction was necessary or executed in this handoff.

## Remaining work and ownership

The requested available-evidence indexing is complete. In the parent's supplied
September 14 01:59 CEST snapshot, replacement processor replay 1705996 and
supervised restart 1705995 are queued without result data. This is a dated
handoff, not a fresh scheduler query or guaranteed ETA. Neither is included as
completed evidence. Final variable efficiency and any supervised analysis await
their own validated immutable owner bundles. Root owns main/wiki integration;
WP5 publishes only the focused index branch and a private status report. No main
merge, wrapper pin change, frozen checkout mutation, new experiment or LaTeX.
