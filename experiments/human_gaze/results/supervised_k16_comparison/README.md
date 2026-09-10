# S1 supervised human-gaze K16 comparison

Status: **analysis preregistration and tooling complete; supervised results are
not yet available.** Training array `1702710` was admitted as `0-5%1`, one A40,
five CPUs,32 GiB and24 hours per element, dependency `afterany:1690935`, with
requeue disabled. Its immutable execution source remains
`5a31685d56ec727b9a46db60598a0693fae7e20e` and its run ID is
`20260910-2128_supervised-k16-comparison_5a31685`. This analysis preparation is
a child branch; it does not alter or duplicate the submitted job.
The final preparation check passed all 266 CPU tests, including 12 focused
analysis and scientific-invariant tests.

## Frozen off-center subgroup

`offcenter_validation_manifest.json` was generated before any supervised model
output existed. It uses only the frozen validation records and cached human
annotation mass. Within each STAViS source it ranks clips by mean human mass
outside Center-32, breaks exact score ties by ascending stable clip ID, and
selects `ceil(0.25 * source validation clips)`.

| Source | Validation clips | Eligible clips | Validation videos | Eligible videos |
|---|---:|---:|---:|---:|
| AVAD | 6 | 2 | 6 | 2 |
| Coutrot_db1 | 23 | 6 | 8 | 3 |
| Coutrot_db2 | 22 | 6 | 2 | 2 |
| DIEM | 214 | 54 | 13 | 11 |
| ETMD_av | 77 | 20 | 2 | 2 |
| SumMe | 85 | 22 | 3 | 3 |
| **Total** | **427** | **110** | **34 source/video pairs** | **23 source/video pairs** |

The frozen file SHA256 is
`69e6f3ba51c8dd3c4388dc082d4ccc96f44e4941090ace23be9e91c03638c7fb`;
its canonical selected-row SHA256 is
`6af272437fa4cc06610314e98f7630d9c2527e0d771ea2db1a2c2852c4123e3c`.
Aggregation is eligible frames to clip means, clip means to video means, then
equal source means. This quantitative group is separate from the existing 24
extreme examples used for visual inspection.

## Required action evidence

No reusable full-validation K16 action export exists for the six historical RL
endpoints. The existing validation histories contain aggregate coverage, the
center-collapse reports contain aggregates, and the fixed qualitative manifest
contains actions for only24 selected clips from seed440826. The HLVid tables are
downstream utility evidence, not STAViS action traces.

The minimal future extraction is therefore:

- six RL endpoint exports, one per base seed at fixed cumulative update20,000;
- thirty supervised exports, the six seeds at cumulative updates2,315,5,000,
  10,000,15,000 and20,000.

Each export covers all427 validation clips once and is reused for coverage,
SIM/CC, Center-16 overlap, static-frequency and same-source shuffled-video
controls, SL–RL/RL–RL agreement, the off-center subgroup and qualitative
rendering. It is not duplicate QA for each analysis.

After training and separate fixed-owner admission of the bounded extraction,
run one seed bundle inside its GPU allocation:

```bash
scripts/human_gaze/export_supervised_k16_seed_bundle.sh \
  440826 \
  /storage/user/latn/artifacts/autogaze-supervised-k16/20260910-2128_supervised-k16-comparison_5a31685 \
  /storage/user/latn/artifacts/autogaze-supervised-k16/20260910-2128_supervised-k16-comparison_5a31685/action_exports
```

Repeat for base seeds440827 through440831. The wrapper calls greedy label-free
exact-K16 inference with local fine actions69–264, records all clip/frame IDs,
hashes the checkpoint and action stream, and measures inference runtime and
peak memory. It refuses existing output directories and dirty source checkouts.
No extraction job is submitted by this branch.

## Complete CPU curation

The immutable analysis contract is
`../../configs/supervised_k16_analysis.yaml`. Once all36 action exports and six
resource receipts are present, run:

```bash
python scripts/human_gaze/curate_supervised_k16_comparison.py \
  --config experiments/human_gaze/configs/supervised_k16_analysis.yaml \
  --output-dir /storage/user/latn/artifacts/autogaze-supervised-k16/S1_CURATION_RUN_ID

python scripts/human_gaze/plot_supervised_k16_comparison.py \
  --metrics /storage/user/latn/artifacts/autogaze-supervised-k16/S1_CURATION_RUN_ID/metrics.json \
  --output-dir /storage/user/latn/artifacts/autogaze-supervised-k16/S1_CURATION_RUN_ID/figures
```

The curator validates hashes and exact action geometry, then reports all six
fixed endpoints; fixed supervised convergence points; source-balanced full and
off-center coverage; uniform-selected-density SIM/CC; dynamic, Center-16,
static-frequency, shuffled-video and train-source-prior controls; center overlap
and entropy; paired SL–RL agreement; and all15 unordered RL–RL pairs as
non-independent context. It includes descriptive six-seed 90% t intervals and
a paired source-stratified video bootstrap. Convergence is shown against base
clips, nominal trajectory action rows and measured single-GPU wall time, with
hardware labeled. Nominal rows are not FLOPs or a fourfold runtime claim.

The curator fails closed when any endpoint, fixed checkpoint, resource receipt,
input checksum or action stream is missing. Its `gate.json` can admit HLVid only
under the preregistered practical rule. If admitted, all six supervised endpoints
must use the frozen full HLVid protocol; the six existing RL K16 results are
reused rather than rerun.

## Qualitative videos and failures

The existing 24 human-selected clips remain the only candidate panel and use
paired seed440826. After curation, the role rule selects within each source:

- representative: the clip closest to that source's median four-frame
  supervised-minus-RL coverage delta, then clip ID;
- failure: the lowest delta, then clip ID.

This is a transparent post-result role assignment inside a frozen candidate
set, not manual example selection. Render full16-frame 3 Hz paired videos and
the fixed four-frame contact sheets with:

```bash
python scripts/human_gaze/render_supervised_k16_qualitative.py \
  --dataset-root /storage/user/zverev/datasets/av-gaze-stavis \
  --manifest /home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/d0_stavis_fold1_validated/clips.jsonl \
  --cell-mass /home/stud/latn/master-thesis/AutoGaze/outputs/human_gaze/d0_stavis_fold1_validated/cell_mass.npy \
  --curated-metrics /storage/user/latn/artifacts/autogaze-supervised-k16/S1_CURATION_RUN_ID/metrics.json \
  --fixed-qualitative-manifest experiments/human_gaze/results/r2c_qualitative_offcenter/qualitative_manifest.json \
  --supervised-actions /storage/user/latn/artifacts/autogaze-supervised-k16/20260910-2128_supervised-k16-comparison_5a31685/action_exports/supervised/base440826_train540826/step20000 \
  --rl-actions /storage/user/latn/artifacts/autogaze-supervised-k16/20260910-2128_supervised-k16-comparison_5a31685/action_exports/rl/base440826_train540826/step20000 \
  --output-dir /storage/user/latn/artifacts/autogaze-supervised-k16/S1_CURATION_RUN_ID/qualitative
```

## Interrupted-run recovery

The admitted launcher deliberately refuses an existing seed directory and has
automatic requeue disabled. Do not resubmit an interrupted array element or
silently delete its partial files.

1. Preserve the failed allocation and obtain a fixed-owner recovery admission.
2. Run the existing checkpoint verifier against the phase directory. Resume in
   place only if `checkpoint_latest_complete.json` and every recorded model,
   optimizer, task and teacher-RNG hash verify under the exact phase/seed/sampler
   contract.
3. If the latest bundle is valid, invoke the same phase config with the same
   seed, data, source and optimizer settings and set `trainer.resume` to that
   phase directory. This continues the same seed; it is not a seventh seed.
4. If no valid completion receipt exists, never use an unreceipted periodic or
   mixed latest bundle. Preserve the aborted directory, restart the affected
   phase in a named `recovery_attemptN` directory from its legitimate boundary
   (original pretrained model for stage one, or the verified same-seed stage-one
   endpoint with fresh Adam for stage two), and label it as the same seed.
5. Write `recovery_manifest.json` at the canonical seed root with schema version
   1, status `complete_same_seed_recovery`, the same base/training seeds, failure
   and recovery job-ID lists, exact source/data hashes, absolute or seed-relative
   `resolved_phase_directories.stage1` and `.stage2`, and hashed checkpoint
   verification receipts for both resolved phases. Concretely, the identity
   fields are `immutable_source_commit`, `input_sha256.manifest`,
   `input_sha256.cell_mass`, `failed_job_ids`, `recovery_job_ids`, and
   `checkpoint_verification_receipts.{stage1,stage2}.{path,sha256}`. Consolidate
   a verified `execution_manifest.json` and `final_sacct.json` at the canonical
   seed root.

The action-bundle wrapper accepts `SUPERVISED_STAGE1_DIR` and
`SUPERVISED_STAGE2_DIR` for an admitted recovery. The CPU curator reads the
canonical recovery manifest. Partial attempts remain accounted for in compute
and are never represented as independent seeds.

For curation, each `final_sacct.json` must contain schema version1, status
`complete`, base seed, terminal state `COMPLETED`, exit code `0:0`, one allocated
GPU, requested host memory `34359738368` bytes, and positive numeric elapsed,
MaxRSS and peak-GPU-memory bytes. This makes Route A fail closed until measured
training resources—not only nominal action rows—are complete.

For a recovered seed, `final_sacct.json` must additionally contain an
`attempts` list covering every job ID named in the recovery manifest. Every row
records its positive elapsed seconds and one allocated GPU; failed attempts
remain in this accounting instead of being hidden by the successful endpoint.
