# Supervised K16 checkpoint repair and same-seed restart

The six original array1702710 attempts reached 100 updates and failed before
writing any checkpoint. `failed_attempts.json` freezes their source/run/seed
identities and hashes the existing prechecks, stderr, partial GNU time and GPU
telemetry. Those heavy/raw paths are preserved unchanged. Final scheduler totals
must be obtained from the sole owner and retained in all-attempt cost accounting.

The metadata-only repair is commit
`13179be38351df168d702962d81ebdca85691a33`, based on reviewed integration
`9c089d183cdfd1e0755287e6fa9db67c73a3bb88`. Hydra passes an instantiated optimizer
as an explicit Trainer argument, consuming its original name. The repair passes
`optimizer_name` separately, checks it against the actual optimizer class, and
stores it in the supervised saved config. No optimizer, executable LR schedule,
objective, initialization, transform, budget, exposure cap, or seed is changed;
legacy RL does not receive a saved-config change.

Six CPU regressions instantiate the real Trainer with resolved stage-one and
stage-two configs and actual optimizer objects. They exercise save/reload of
model, Adam, scheduler, teacher RNG and input cursor; the fresh-Adam stage-two
boundary; and wrong/missing optimizer-name rejection. The original full suite
plus these regressions passed282 tests before restart-lineage scaffolding.

The disposable actual-model smoke from clean repair commit13179be passed on the
previously idle RTX A2000 12GB at 01:43CEST. It uses full16-frame/224px TRAIN
inputs, microbatch1, four-microbatch accumulation, one Adam update, verified
save/reload, a resumed backward pass, and label-free greedy exactK16. It is not a
production checkpoint or validation experiment. `checkpoint_smoke.json` pins its
full receipt: peak allocated CUDA441529856B, reserved509607936B, RSS2232098816B,
within10GiB/8GiB/two-CPU-thread/30-minute bounds.

## Six same-seed fresh restarts — sole-owner admission required

`../../configs/supervised_k16_restart_execution.yaml` inherits the exact original
execution recipe by SHA-256. Both original stage configs remain unchanged:
2315 stage-one updates then17685 stage-two updates, batch4 base clips, original
pretrained initialization, fresh Adam at the stage boundary, six base seeds
440826..440831 with continuation/teacher seeds540826..540831. Each completed
restart still has the fixed20k/80k-base-clip endpoint. The extra400 base-clip
presentations/102400 nominal action rows per failed attempt are separately
reported and included in total compute, not silently counted as new seeds or
omitted. Nominal rows are not physical decoder FLOPs or measured runtime.

Request one array0-5%1, one A40/5CPU/32GiB/24h per seed, Requeue0, no node pin.
The sole fixed owner01a05a0a-3c7e-7053-aa24-5069bff7fa96 must choose admission
timing/dependency against all user jobs, preserving existing priorities and the
3GPU/384GiB global cap. No job is submitted by this source. The inherited
launcher refuses existing directories; the versioned restart run/root must be
new. The exact admitted execution SHA and run ID are frozen in the separately
published admission/analysis manifest, avoiding a self-referential Git SHA.

Inside the owner-admitted allocation, from its clean detached execution SHA:

```bash
AUTOGAZE_WORKTREE=/storage/user/latn/worktrees/autogaze-supervised-k16-checkpoint-recovery \
AUTOGAZE_EXECUTION_COMMIT=ADMITTED_EXECUTION_SHA \
SUPERVISED_K16_RUN_ID=VERSIONED_RESTART_RUN_ID \
bash experiments/human_gaze/slurm/run_supervised_k16_restart_array.sbatch
```

This is the task payload, not an independent submission command. Precheck
validates the tested repair's exact training-code blobs, allowed metadata-only
source chain, actual successful smoke receipt, and preserved same-seed failed
evidence. Completed execution manifests bind both phase receipts and `restart_of`.

## Analysis, timing and next evidence

The immutable execution SHA is `ec320a5435275c6098c6d299b11d98d18d9b1afb`; the
new run is `20260914-0152_supervised-k16-comparison_ec320a5`. All six actual
prechecks passed from a clean detached copy of that SHA, with live data,
initialization, smoke and failed-attempt hashes verified. Final CPU suite:286
passed. `admission_request.json` is the concrete sole-owner handoff; it is not a
submission receipt. The separately published analysis layer pins this SHA and
the new root without altering the execution SHA or original inventory.

`runtime_estimate.json` projects the actual A40 warm-update/validation timings:
about3.4--6.8h per seed (median3.8h), or20--41h for the serialized six-seed lane
(median23h) after it begins. This assumes later/stage-two throughput is similar
and excludes queue delay and additional startup/checkpoint/I/O overhead. It is
a planning range, not a measured full-run runtime or completion guarantee.

After owner-admitted training and separate action-extraction admission, run the
seed bundle once per base seed from a clean analysis checkout:

```bash
SUPERVISED_ANALYSIS_CONFIG=experiments/human_gaze/configs/supervised_k16_restart_analysis.yaml \
bash scripts/human_gaze/export_supervised_k16_seed_bundle.sh 440826 \
  /storage/user/latn/artifacts/autogaze-supervised-k16/20260914-0152_supervised-k16-comparison_ec320a5 \
  /storage/user/latn/artifacts/autogaze-supervised-k16/20260914-0152_supervised-k16-comparison_ec320a5/action_exports

python scripts/human_gaze/curate_supervised_k16_comparison.py \
  --config experiments/human_gaze/configs/supervised_k16_restart_analysis.yaml \
  --output-dir /storage/user/latn/artifacts/autogaze-supervised-k16/NEW_CURATION_RUN_ID
```

The updated authoritative restart inventory/config must pin the admitted
execution SHA and new canonical root. Original inventory/source5a31685 is not
overwritten. Export and curation independently verify actual source/seed/cursor,
completion/model hashes and the failed-attempt lineage. A final `final_sacct.json`
for each seed must have an `attempts` list covering the original1702710 element
and the completed restart element, plus any subsequent recovery attempts. The
CPU resource validator fails closed if failed allocation time is omitted.

Fresh-restart process logs do not reconstruct prior failed time; wall-time
convergence curves are withheld. Nominal exposure curves remain available and
final scheduler/GPU totals include all attempts. The full-validation analysis is
still30 SL exports plus six frozen RL exports; no extraction is launched here.
Frozen human-only off-center subgroup, Center16/static/shuffle controls,
SL-RL/RL-RL agreement, qualitative seed440826 panel and practical HLVid gate are
unchanged. HLVid remains closed until complete all-six validation/resource
evidence passes the descriptive practical gate; then evaluate all six SL20k
endpoints and reuse the six existing RL HLVid endpoints.
