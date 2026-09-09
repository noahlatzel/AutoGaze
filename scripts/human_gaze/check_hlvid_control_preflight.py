#!/usr/bin/env python3
"""Fail closed unless a one-example K16 control preflight proves the frozen contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autogaze.human_gaze.coverage import center_order
from scripts.runners.hlvid_evidence import load_resume_state


CENTER16_ACTION_IDS = [69 + int(cell) for cell in center_order(14)[:16]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=("pretrained", "center16"), required=True)
    return parser.parse_args()


def validate(summary: dict, records: dict, policy_kind: str) -> dict:
    if summary.get("num_examples") != 1:
        raise ValueError("Preflight must contain exactly one completed answer")
    compatibility_key = summary.get("compatibility_key")
    completed = list(records["completed"].values())
    if len(completed) != 1 or not compatibility_key:
        raise ValueError("Preflight evidence must contain exactly one durable completion")
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
        "expanded_context_length": expanded_context,
        "expanded_visual_tokens": expanded_visual,
        "raw_action_observations": raw["observed_count"],
        "retained_patch_observations": retained["observed_count"],
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    compatibility_key = summary.get("compatibility_key")
    if not compatibility_key:
        raise ValueError("Summary lacks compatibility_key")
    records = load_resume_state(args.evidence_dir, compatibility_key)
    print(json.dumps(validate(summary, records, args.policy_kind), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
