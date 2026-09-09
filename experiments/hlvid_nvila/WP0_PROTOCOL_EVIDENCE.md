# HLVid protocol and restart-safe evidence

This additive utility package supports the matched pretrained fine-only K16 and
Center16 controls, and allocation-only recovery for variable-budget evaluation.
It does not change an evaluator, rerun QA, or infer unavailable historical costs.
The older manuscript does not define the scope of this work.

## Frozen input contract

- Dataset: `/storage/slurm/latn/data/AutoGaze/HLVid/data/test-00000-of-00001.parquet`,
  SHA256 `ed2a1a47603fd19f5d7fae0db4f9792369f5aff6548f35e0225c3b4bd35e112a`;
  268 questions, 77 videos.
- Resolve each video literally as `dataset_root / "videos" / row["video_path"]`.
- Legacy runner: `/home/stud/latn/AutoGaze/scripts/runners/evaluate_hlvid_nvila.py`,
  SHA256 `f7281533b556e508b6c82fc419e66a138d93e151be68932a7a632c8585c6de1e`.
  Source was read on the VM at reported checkout
  `1eb4d9243f3306cd3a759521d11281ac3c5bb190`; that commit was not resolvable through
  GitHub during local implementation. The script hash is the executable identity.
- Shared row loader `hlvid_common.py`: SHA256
  `d390fd54b8fca681e2d530c3fb53ee15fdddcb06c379ef86d7720a914949e0b4`.
- Preserve 128 uniform frames, 64 thumbnails, max_tiles_video=48, tile_len=16,
  greedy generation, max_new_tokens=16, and no input truncation.
- NVILA snapshot revision `7a5670e20da435d98b0efdc49f9a536f73985152`;
  NVILA uses bfloat16. This statement does not specify the separate AutoGaze
  parameter dtype or override its existing autocast behavior.
- Preserve the established evaluator's scoring. R2e uses micro QA accuracy as
  primary, with macro-video sensitivity; R2f retains its own registered primary.

## CPU decode audit

Run from the AutoGaze checkout with a fresh run directory:

```bash
uv run python scripts/runners/audit_hlvid_protocol.py \
  --output-dir outputs/hlvid/wp0_protocol_audit/<run-id>
```

The script imports NumPy, OpenCV, Pillow and PyArrow, but does not import torch,
load either model, or retain a video in memory. One decoded frame and its
conversion are live at a time. Output metadata covers only paths, source-index
maps and statuses, never images. Runtime duration depends on video seeking;
run on a functioning CPU execution surface, not a GPU allocation merely to
access the files.

The audit performs the same last-frame `grab()` validation, rounded `linspace`
and seek/read sequence as the literal loader. Failed duplicate indices are
retried until one succeeds. Only successful BGR-to-RGB and PIL conversions enter
the successful-index cache. Conversion exceptions abort that video's legacy
output. Failed reads disappear during chronological reconstruction; deficits
are filled by repeating the last successfully reconstructed frame.

Outputs are `decode_audit.jsonl`, `summary.json` and
`protocol_runtime_manifest.json`. Existing output directories are refused.
Default parquet and legacy source hashes are checked before decoding. Alternate
manifests may pass explicit expected hashes; their identity remains visible.
The frozen audit CLI rejects sample counts other than 128. Its lower-level
sampling and decoder helpers remain general for focused tests.

Exit status 0 means the audit completed, not that control admission passed.
The control owner must inspect `summary.json`, especially `reuse_qualification`
and `affected_question_rows`, before admitting expensive QA runs. Recovered
retries require interpretation and do not automatically require rerunning QA.

The runtime manifest records the known frozen model/configuration contract;
it does not freshly read or verify model weights, processor files or live model
settings. `live_model_files_verified=false` makes that boundary explicit. The
fixed/runtime owners must supplement it with their live model, configuration
and execution-runtime manifest. The dataset and legacy-loader hashes are checked
by this audit itself.

Keep these findings separate:

- `metadata_count_corrected`: the reported terminal frame was invalid. The
  legacy loader already corrected this; the decrement alone does not invalidate QA.
- `sampled_read_failure`: a requested read failed, including a recovered retry.
- `legacy_output_substituted`: the reconstructed output index map changed or
  required tail padding.
- `decode_usable=false`: the historical loader would not return a usable clip.

No current failures supports reuse, subject to the explicit limitation that a
present-day audit cannot certify every historical decode. Failures identify
the affected videos/questions for paired-input inspection; they do not condemn
all historical endpoints. A retry can recover the exact intended sequence.

## Shared public API

Import `scripts.runners.hlvid_evidence` from the repository root, or import
`hlvid_evidence` when `scripts/runners` is on `sys.path`. This module uses only
the Python standard library and accepts lists or tensor-like `.tolist()` values.

```python
stable_example_key(question_id, split)
stable_video_key(video_path, split)  # use the raw manifest-relative path
legacy_decode_map(intended_indices, successfully_decoded_indices)
resume_compatibility_key(identity)
load_resume_state(path, expected_compatibility_key)
counter_summary(values, expected_observations=None, unit="count", source="observed")
unavailable_counter(reason, expected_observations=None, unit="count")
merge_counter_summaries(summaries)
summarize_gaze_slots(lengths, padding, gazing_pos=None,
                     actions_per_frame=None, fine_action_offset=0)
write_evidence_record(directory, record)
export_evidence_jsonl(directory, output_path, expected_compatibility_key,
                      completed_only=False)
```

Arguments after `*` in the function definitions are keyword-only. The signatures
above omit `*` for readability; follow the definitions when calling positionally.

An example key hashes the typed question ID and split. If the dataset lacks a
unique question ID, use its immutable original row index and preserve it across
`--start-index`, subset evaluation and resume. A video key hashes the raw path
and split. The compatibility identity must include the dataset/row identity,
checkpoint hashes, policy/adapter, actual input sampling and preprocessing,
NVILA model/processor identity, generation settings and scoring implementation.
Do not include timestamps or instrumentation-only revisions in that identity;
record those separately. JSON NaN and Infinity are rejected.

## Durable records and QA success

Use one records directory per run/arm, with a single writer. Keep run manifests
outside this directory because its `*.json` files are all evidence records.
Each record has `schema_version=1`, stable example/video keys, `attempt_id`,
`compatibility_key`, and a state: `attempt_started`, `completed_answer`, or
`failed`. The schema is `schemas/per_example_evidence.schema.json`.

`completed_answer` embeds the answer and its measurements in the same durable
record. A started or failed attempt never authorizes skipping QA. A failed
attempt is terminal; a retry uses a new attempt ID. Duplicate completions,
conflicting terminal events and incompatible identities are rejected.

The writer fsyncs a temporary file, atomically publishes it using a same-directory
hard link without replacing an existing event, then fsyncs the directory on
POSIX. The filesystem must support hard links. Incomplete `.pending-*` files
are ignored. JSONL is a rebuildable export, not the authoritative transaction.
An interrupted JSONL tail is rejected on read, rather than silently ignored.

For existing evaluators with a separate QA JSONL, there is no atomic transaction
across independent files. Preserve their durable answers, reconcile by stable
identity, and retain their existing resume behavior. If QA was durably written
but its measurement write was interrupted, mark measurements unavailable or
recover them by a separately labeled deterministic processor replay. Do not
repeat QA merely because a sidecar is missing. New integrations may derive the
QA export from embedded completed records to eliminate this gap entirely.

## Measurements

Capture raw decoder actions before resolution adaptation and retained patches
after it. `num_gazing_each_frame` contains shared padded segment lengths. Use
`if_padded_gazing` for actual valid counts in every batch/frame. When counting
spatial actions, supply `gazing_pos`, `actions_per_frame=265` and
`fine_action_offset=69`; this excludes EOS even if the first EOS is marked valid.
For post-adaptation positions, use that output's vocabulary size, not 265.
`frame_observations` counts processor tile/frame observations, not unique source
frames. Keep decoded source-frame counts separate.

Suggested record fields are `raw_decoder_calls`, `post_adaptation_calls`,
`decode`, `counters`, `context`, `cache_state`, `timing_boundary` and
`measurement_origin`. Record actual expanded visual/context lengths at the
model boundary. Tokenizer placeholder lengths are a different quantity.

For each counter, `observed_sum` and `observed_count` describe only available
measurements. Set `expected_observations` when completeness matters. A resumed
tail remains `partial`; unavailable historical measurements have null totals,
never fabricated zeros. `merge_counter_summaries` weights means by observations
and preserves partial/unavailable status. Merge only disjoint observations with
the same unit and meaning, normally the completed records selected by resume.
Resource use of failed attempts can be reported separately.

Timing records should state whether they measure decode, selection, processor,
generation or the full pipeline, and disclose cache state. Allocation-only
replays can recover actions/patches/tokens; their timing does not reconstruct
historical end-to-end inference latency. Observational instrumentation and
missing historical cost metadata do not invalidate compatible QA answers.

## Validation

```bash
python -m pytest tests/test_hlvid_evidence.py -q
```

Tests cover identity changes, interrupted records and JSONL, duplicate completions,
QA-success state handling, variable valid-versus-padded counts, EOS exclusion,
weighted counter merges with missing prefixes, rounded short-video sampling,
last-frame correction, failed duplicate retry, and fatal RGB conversion.
