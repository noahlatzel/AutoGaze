# Necessary software-repair admission — 2026-09-14

Sole Slurm owner: fixed-K execution task `01a05a0a-3c7e-7053-aa24-5069bff7fa96`.
This admits only user-authorized processor replay and six same-seed supervised
restarts after verified software failures. It does not admit classification
full training, extra seeds or any new NVILA QA. Existing completed jobs remain
unchanged. New job IDs/effective fields are appended after exactly-once submission.

## Resource/prerequisite proof

- All-user `squeue -u latn -r` is empty at 2026-09-14 01:56 CEST. Original
  R2f array1690935, controls1701485 and A40 pilot1703027 are terminal COMPLETED;
  replay1700681 and supervised1702710 are terminal FAILED, not held/requeued.
- The completed six-element R2f population is a verified replay prerequisite.
  It no longer reserves two array lanes. Controller still resolves completed
  original array records; replay retains its full-array afterok1690935 gate,
  already satisfied and removed from effective Dependency. New jobs have no dependency on failed1700681 or
  failed1702710. This avoids unsatisfiable afterok dependencies.
- One replay lane: 1 A40 / 5 CPU / 128 GiB / 24h. One supervised array0–5%1:
  1 A40 / 5 CPU / 32 GiB / 24h per element, no requeue. Reserve both pending
  potential lanes: **2 GPUs / 160 GiB**, below the standing **3 / 384 GiB**
  cap. Seven counted submissions (six serialized seeds plus replay) are below
  the task cap10 and last-verified all-user QOS submitted cap20. No other
  active user jobs were observed. Future work needs a separate exact permit.
- NORMAL partition UP, MaxTime14days, no partition maximum memory. The last
  verified association group memory ceiling512GiB is above this envelope;
  the orchestration cap is not a claim about actual QOS. No speculative QOS
  change or GPU-class substitution is made. A40 node availability is dynamic.

## Processor-only replay

Prepared handoff lives in AutoGaze branch `codex/r2f-hlvid-replay-path-repair`
commit `03cd606257b9f3f28ac295e62533453b2e2b2612`, under
`results/human_gaze/20260914-0202_r2f-hlvid-complete-paired-accuracy_6b4eef7/replay_recovery_handoff.md`.
Execution source remains exact tested repair
`d3b8d5a4abc448e78021f7085563c0f930113618`, parent failed source
`ecd72f151e4ca5066ce48b9ecb0032af1cb06645`.

Clean detached checkout:
`/home/stud/latn/.codex/worktrees/r2f-replay-d3b8d5a-20260914`.
Launcher `experiments/human_gaze/slurm/run_r2d_hlvid_allocation_replay.sbatch`,
SHA256 `3b68577423a894eecc0cb5f98a003fef69875b38febe1f581fccf1709200df37`.
Runner SHA256 `db77f5f7261f5ab6f3e35fad947854534c07b875ca40eb28bd360bb45d273e1c`.
New, previously absent run ID:
`20260914-0153_r2f-r2d-hlvid-allocation-replay_d3b8d5a`.
Root: `/storage/user/latn/artifacts/autogaze-r2f/<RUN_ID>/`.
Environment: unchanged healthy `vila-autogaze-eval` conda environment.

Only two filesystem Path arguments become strings in processor initialization;
all input/selector settings remain128frames/64thumbnails/MTV48 and original
calibrated VARIABLE-EOS. CPU owner reports44 passing focused tests. Actual
constructor smoke is pinned to clean d3b8d5a and processing source
`2b9dbf36a9c69954af7e7cc8478a980e406e91c2ba27b7f2f164167da70dbc9a`;
heavy weight loading/device movement was stubbed, so this is not allocation
certification. Accepted shared audit hashes/runtime supplement were independently
loaded successfully:77videos, no current requested decode failures; no historical
certification asserted. Launcher first validates against the original
uninterrupted VARIABLE seed440826 observation before extrapolating implementation
validity. It serially replays three checkpoints, zero NVILA generation calls.

Exact submission (from the detached checkout):

```bash
sbatch --parsable --dependency=afterok:1690935 --export=ALL,R2F_REPLAY_WORKTREE=/home/stud/latn/.codex/worktrees/r2f-replay-d3b8d5a-20260914,R2F_REPLAY_CODE_COMMIT=d3b8d5a4abc448e78021f7085563c0f930113618,R2F_REPLAY_RUN_ID=20260914-0153_r2f-r2d-hlvid-allocation-replay_d3b8d5a,R2F_PROTOCOL_AUDIT_DIR=/home/stud/latn/.codex/worktrees/r2f-hlvid-curation/outputs/hlvid/wp0_protocol_audit/20260909-0211_wp0-hlvid-protocol-audit_b98970d experiments/human_gaze/slurm/run_r2d_hlvid_allocation_replay.sbatch
```

An attempted empty `--nodelist=` override was rejected at submission, creating
no job. The successful command above retains the immutable launcher; while
PENDING, only ReqNodeList was cleared using the installed individual-ID syntax.
The before/after machine receipt verifies unchanged resources/source/dependency.
One test-only query with the empty override and completed original afterok gate returned
`allocation failure: Requested node configuration is not available`; it created
no job. This is availability evidence, not an inference/protocol failure or
authorization to duplicate submission. A subsequent original-launcher test-only
query succeeded, projecting node17 September16 09:44:50CEST. Its reported
test-only ID1705994 is not a submitted job. Successful replay1705996 retains
the satisfied original dependency, and its node-pin relaxation improves the
current projection without changing A40 class or resource demand.

## Six supervised same-seed fresh restarts

Published branch `codex/supervised-k16-checkpoint-recovery`, immutable execution
commit `ec320a5435275c6098c6d299b11d98d18d9b1afb`, training metadata repair
`13179be38351df168d702962d81ebdca85691a33`. Clean detached checkout:
`/storage/user/latn/worktrees/autogaze-supervised-k16-checkpoint-recovery`.
New, previously absent run ID:
`20260914-0158_supervised-k16-comparison_ec320a5`.
Root: `/storage/user/latn/artifacts/autogaze-supervised-k16/<RUN_ID>/`.
Environment: unchanged healthy `autogaze` conda environment.

Restart launcher SHA256
`f2f2948c8153bbc12000d87d45d645d84009919cc7387289e04b75077f9f4f82`;
inherited phase launcher SHA256
`fc8ead4131ecc1647a2458a059432734df451fef9bafb0959d3f7241a72b2b48`;
restart execution YAML SHA256
`9334c9660cd2a42ce0e016c1e4d5d062ee3095dac6b6729abdf4dc99a8e85b99`.
Original recipe YAML remains
`e0ee7948b695a7f43ec40d4209437bdf8bcd138d1776b751195004eadac02050`.

The repair retains explicit optimizer recipe identity after Hydra injected the
actual Adam object. It does not change optimizer/LR/objective/inputs/budget/seed.
Actual-model disposable smoke verifies save, exact model/Adam/scheduler/teacher
reload, next-input cursor/order, resumed backward and greedy exactK16; full
receipt SHA256 `3c9fbcdd41e8039963e92aa975a80ff37ebfaa09fd03f0ed37767ea9963f08c6`.
CPU admission on the final execution source: **9 passed in11.99s** (six actual
Trainer save/reload regressions plus three restart-lineage regressions).
XML: `/storage/user/latn/artifacts/autogaze-supervised-k16-repair-smoke/20260914-0155_restart-source-precheck_ec320a5/focused_admission_tests.xml`,
SHA256 `5127e942754c64c8337839eeffee3c0d1ad6540141e94e13c5d6f48eba77e568`.
Detached live precheck passed, validating source blobs, successful smoke,
original failed-attempt hashes/no checkpoint, dataset/initialization identity.
Precheck SHA256 `91f17449491a602af83180b9dfe5a65de9e82348f47cf8edfc13b966c9815713`.

Exact submission (from the detached checkout):

```bash
sbatch --parsable --export=ALL,AUTOGAZE_WORKTREE=/storage/user/latn/worktrees/autogaze-supervised-k16-checkpoint-recovery,AUTOGAZE_EXECUTION_COMMIT=ec320a5435275c6098c6d299b11d98d18d9b1afb,SUPERVISED_K16_RUN_ID=20260914-0158_supervised-k16-comparison_ec320a5 experiments/human_gaze/slurm/run_supervised_k16_restart_array.sbatch
```

Six paired base440826–831 / continuation540826–831 chains remain stage1=2315
then stage2=17685 updates, fresh Adam at the original phase boundary, fixed20k
endpoint. Initial/restart launcher refuses existing seed paths, with no
automatic checkpoint recovery. Failed100-step attempts had no recoverable
checkpoint; this is explicit versioned fresh restart, not six additional seeds.
Each original failed attempt's100updates/400base-clip exposures and scheduler
cost must remain in final all-attempt accounting. Original six allocation times
sum15:08; process GNU-time770.41s is a different boundary. HLVid remains closed
until all-six supervised validation/analysis qualifies its conditional gate.

## Actual submission and current state

At **2026-09-14 01:59:05 CEST**, all seven user elements are PENDING(Priority),
with no allocated node and zero execution/output yet:

| Family | Actual ID | Source | Live projected first start (CEST) |
|---|---|---|---|
| Corrected processor-only replay |1705996|d3b8d5a|Sep15 23:39:52|
| Six serialized same-seed training restarts |1705995_0–5|ec320a5|Sep16 23:40:00|

Both effective Dependency fields are null (replay's original full-array afterok
was immediately satisfied); both ReqNodeList fields are null. No old job was
cancelled/requeued, no completed result altered and no duplicate pilot launched.
Exact raw before/after job fields are in [the machine receipt](2026-09-14-repair-admission.json).
The six equal array projections do NOT imply simultaneous starts: %1 remains.
These queue estimates are dynamic; there is no guaranteed finish date.

### Bounded runtime/finish guidance

Independent read of all six original `stage1/training_metrics.jsonl` files
confirms warm update0–99 slopes **0.404–1.014s/update**, excluding initial
validation. Step100 validation takes **19.49–19.92s**. Projecting20k updates
plus203 declared validation passes gives **3.4–6.8h per seed, 20–41 serialized
active hours for all six** (≈23h median-rate planning value). Whole failed
allocations1:45–4:31 include startup/failure and are a different boundary. This
is not a measured full-duration/stage-two rate. If the current September16 late
start held with uninterrupted back-to-back execution, conditional training
completion would be approximatelySeptember17–18; queueing/preemption can extend it.
Re-estimate after actual first-save/steady stage progression; no checkpoint
recovery/autorequeue is assumed. Validation analysis/conditional HLVid is extra.

Source raw timing paths are the preserved original root above, each
`seed<BASE>_train<BASE+100000>/stage1/{training,validation}_metrics.jsonl`.
An independent per-seed timing calculation gives3.344–6.758h and20.066–40.548h
sixfold low/high envelope; the owner's published warm-rate projection in
`supervised_k16_restart/runtime_estimate.json` differs only by using a common
validation duration. Neither is a confidence interval or guaranteed forecast.

### Analysis handoff synchronization (no execution/source change)

Owner's subsequently published analysis branch tip
`7c3d53984257a8e7ffa1556f30a678b04590ed7d` correctly pins executionec320a5,
but its proposed run root ends **0152** and its job ID is still null. Actual
admitted root ends **0158**, job1705995. Root/analysis owner must update only
the unexecuted analysis config/inventory/admission fields to this receipt before
extraction/curation. Do not modify frozen executionec320a5, resubmit or create
results under the unused0152 root. This is a metadata handoff, not a training
blocker or an extra experiment.

Replay has no fresh allocation measurement yet. For scale only, completed
pretrained control processor averages 156.9s/QA versus 24.5s generation, with
14.3s decode. Applying that question-weighted processor rate to 3×77 video
calls gives an order-of-magnitude ≈11 active hours including decode, not a
validated VARIABLE/unweighted-video rate. The 24h request is the bound; a
**rough 8–16h after allocation** is provisional and must be replaced by actual
early replay progress. No guaranteed completion date or full QA timing claim.
Action classification continuation remains unadmitted pending its bounded
performance/cost decision; actual A40 pilot lower bound 95.31h/head is not a
launch forecast.
