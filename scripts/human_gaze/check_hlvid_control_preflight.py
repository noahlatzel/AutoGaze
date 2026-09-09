#!/usr/bin/env python3
"""Fail closed unless a one-example K16 control preflight proves the frozen contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from autogaze.human_gaze.coverage import center_order
from scripts.runners.hlvid_evidence import fingerprint, load_resume_state


CENTER16_ACTION_IDS = [69 + int(cell) for cell in center_order(14)[:16]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=("pretrained", "center16"), required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--admission-record", type=Path, required=True)
    return parser.parse_args()


def validate(
    summary: dict,
    records: dict,
    policy_kind: str,
    *,
    admission_record_sha256: str | None = None,
) -> dict:
    compatibility_key = summary.get("compatibility_key")
    if not compatibility_key:
        raise ValueError("Preflight summary lacks a compatibility key")
    completed = [
        record
        for record in records["completed"].values()
        if int(record.get("question_id", -1)) == 0
    ]
    if len(completed) != 1:
        raise ValueError("Preflight evidence must contain one durable completion for question 0")
    record = completed[0]
    if record["compatibility_key"] != compatibility_key:
        raise ValueError("Summary and evidence compatibility keys differ")
    if record.get("measurement_origin") != "original_forward":
        raise ValueError("Preflight must be measured during the original forward pass")
    decode = record.get("decode", {})
    if (
        not decode.get("decode_usable")
        or decode.get("sampled_read_failure")
        or decode.get("legacy_output_substituted")
        or decode.get("reported_index_mismatches")
        or decode.get("tail_padding_count")
    ):
        raise ValueError("Preflight decode is not exact")
    context = record.get("context", {})
    expanded_context = context.get("expanded_context_length")
    expanded_visual = context.get("expanded_visual_tokens")
    if not isinstance(expanded_context, int) or expanded_context <= 0:
        raise ValueError("Missing actual expanded context length")
    if not isinstance(expanded_visual, int) or expanded_visual <= 0:
        raise ValueError("Missing actual expanded visual-token count")
    if context.get("context_truncated") is not False:
        raise ValueError("Preflight does not certify the no-truncation contract")
    counters = record.get("counters", {})
    raw = counters.get("raw_decoder_spatial_actions_per_tile_frame", {})
    if (
        raw.get("availability") != "complete"
        or raw.get("observed_count", 0) <= 0
        or raw.get("min") != 16
        or raw.get("max") != 16
        or raw.get("mean") != 16
    ):
        raise ValueError("Preflight did not observe exact K16 actions for every tile-frame")
    retained = counters.get("post_adaptation_valid_patches_per_tile_frame", {})
    if retained.get("availability") != "complete" or retained.get("min", 0) <= 0:
        raise ValueError("Preflight lacks valid post-adaptation patch counts")
    action_batches = [
        actions
        for call in record.get("raw_decoder_calls", [])
        for batch in call.get("decoder_action_ids", [])
        for actions in batch
    ]
    if not action_batches:
        raise ValueError("Preflight lacks raw decoder action traces")
    if policy_kind == "center16" and any(
        actions != CENTER16_ACTION_IDS for actions in action_batches
    ):
        raise ValueError("Center16 actions differ from the frozen STAViS tile-local order")
    return {
        "status": "pass",
        "policy_kind": policy_kind,
        "compatibility_key": compatibility_key,
        "example_key": record["example_key"],
        "question_id": 0,
        "source_completed_record_sha256": fingerprint(record),
        "admission_record_sha256": admission_record_sha256,
        "expanded_context_length": expanded_context,
        "expanded_visual_tokens": expanded_visual,
        "raw_action_observations": raw["observed_count"],
        "retained_patch_observations": retained["observed_count"],
    }


def persist_or_validate_attestation(path: Path, result: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError("Persisted preflight attestation differs from question-0 evidence")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-preflight-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    compatibility_key = summary.get("compatibility_key")
    if not compatibility_key:
        raise ValueError("Summary lacks compatibility_key")
    records = load_resume_state(args.evidence_dir, compatibility_key)
    admission_record_sha256 = hashlib.sha256(args.admission_record.read_bytes()).hexdigest()
    admission = json.loads(args.admission_record.read_text())
    if admission.get("status") != "pass":
        raise ValueError("Control admission record did not pass")
    result = validate(
        summary,
        records,
        args.policy_kind,
        admission_record_sha256=admission_record_sha256,
    )
    persist_or_validate_attestation(args.attestation, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
