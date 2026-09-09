# HLVid present-day decode audit

This compact bundle records the single shared WP0 audit used to qualify the
existing R2e/R2f HLVid inputs and new matched controls. It ran under the same
`vila-autogaze-eval` interpreter as the healthy benchmark launchers and checked
the frozen 128-frame legacy-loader behavior for all 77 unique videos covering
268 questions.

Result: no current requested-frame read failure, metadata-count correction,
legacy substitution, unusable video, or affected question row was observed.
The exact parquet and legacy-loader hashes passed. This supports reuse and new
control admission under the frozen input contract, but it is not retrospective
proof of every historical decode (`historical_certification=false`).

Files:

- `protocol_runtime_manifest.json`: frozen protocol, source, dataset, and audit
  execution identity;
- `summary.json`: compact counts and reuse qualification;
- `decode_audit.jsonl`: one source-index/decode record per unique video, with no
  pixels retained;
- `live_runtime_supplement.json`: environment versions, hostname, clean Git
  state, output hashes, and stable keys for any affected rows (empty here).

The ignored execution directory remains at
`/home/stud/latn/.codex/worktrees/r2f-hlvid-curation/outputs/hlvid/wp0_protocol_audit/20260909-0211_wp0-hlvid-protocol-audit_b98970d`.
