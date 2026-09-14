# One corrected processor-only replay: readiness, not admission

Status: **prepared; not submitted by this task**. The fixed-K owner remains the
sole Slurm admission owner. Root authorized this narrow recovery; no NVILA QA
regeneration is needed or allowed to recover costs.

- AutoGaze branch: `codex/r2f-hlvid-replay-path-repair`.
- Exact replay execution commit: `d3b8d5a4abc448e78021f7085563c0f930113618`.
- Parent/original failed source: `ecd72f151e4ca5066ce48b9ecb0032af1cb06645`.
- Clean detached checkout: `/home/stud/latn/.codex/worktrees/r2f-replay-d3b8d5a-20260914`.
- Launcher: `experiments/human_gaze/slurm/run_r2d_hlvid_allocation_replay.sbatch`.
- Launcher SHA256: `3b68577423a894eecc0cb5f98a003fef69875b38febe1f581fccf1709200df37`.
- Replay Python SHA256: `db77f5f7261f5ab6f3e35fad947854534c07b875ca40eb28bd360bb45d273e1c`.
- Proposed new run ID: `20260914-0153_r2f-r2d-hlvid-allocation-replay_d3b8d5a`.
- Resources for owner review: one A40, 5 CPUs, 128 GiB host RAM, 24 h,
  serial replay for three VARIABLE checkpoints, zero NVILA generations.
- Interpreter: existing `vila-autogaze-eval` conda environment; no mutation.
- Audit directory: `/home/stud/latn/.codex/worktrees/r2f-hlvid-curation/outputs/hlvid/wp0_protocol_audit/20260909-0211_wp0-hlvid-protocol-audit_b98970d`.

The launcher still requests node17. The owner must check the full live ledger
before issuing an exact permit and record any same-A40 node override. Existing
array1690935 is complete, so its original full-array afterok gate is satisfied;
do not resubmit QA or revive failed replay1700681.

The repair converts only the two filesystem arguments to strings in the
existing `AutoProcessor.from_pretrained` call. Its real CPU image-processor
regression and processor/tokenizer constructor smoke pass. The latter replaces
only heavy AutoGaze loading/device movement with a fixture: no CUDA work or
allocation certification. Targeted CPU tests total 44 passing; existing
matplotlib packaging warnings do not affect the 2D evidence figures.

Keep frozen 128 uniform frames / 64 full thumbnails / `max_tiles_video=48`,
exact three checkpoint/calibration hashes, base440826–828→train740826–828,
EOS biases 2.3046875 / 2.421875 / 2.3828125. Replay seed440826 must reproduce the
**original uninterrupted VARIABLE** one-question allocation reference exactly:
`/storage/user/latn/artifacts/autogaze-r2f/20260901-0156_r2f-r2d-hlvid-secondary_ecf1535/preflight/seed440826_variable/summary.json`.
Only then may other seeds use its implementation validation. A forced-K16
segment alone cannot validate EOS semantics. Full durable replay needs all 77
videos and 268 stable QA keys per variable seed, with valid masks rather than
padded maxima, source/frame/processor/checkpoint identity and complete raw sums.

If accepted replay completes, run the strict curator without `--accuracy-only`
into a NEW joint result run ID with `--allocation-replay-root=<new-run>/replays`.
Retain this complete accuracy-only stage and the failed original replay in the
source/compute chain. Never repeat NVILA generation to recover missing counters.
