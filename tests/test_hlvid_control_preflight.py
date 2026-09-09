import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "human_gaze"
    / "check_hlvid_control_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("hlvid_control_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixture(policy_kind="center16"):
    actions = (
        list(MODULE.CENTER16_ACTION_IDS)
        if policy_kind == "center16"
        else list(range(69, 85))
    )
    record = {
        "schema_version": 1,
        "example_key": "question-zero-key",
        "question_id": 0,
        "compatibility_key": "identity",
        "measurement_origin": "original_forward",
        "decode": {
            "decode_usable": True,
            "sampled_read_failure": False,
            "legacy_output_substituted": False,
            "reported_index_mismatches": [],
            "tail_padding_count": 0,
        },
        "context": {
            "expanded_context_length": 48_478,
            "expanded_visual_tokens": 48_384,
            "context_truncated": False,
        },
        "counters": {
            "raw_decoder_spatial_actions_per_tile_frame": {
                "availability": "complete",
                "observed_count": 1,
                "min": 16,
                "max": 16,
                "mean": 16,
            },
            "post_adaptation_valid_patches_per_tile_frame": {
                "availability": "complete",
                "observed_count": 1,
                "min": 64,
            },
        },
        "raw_decoder_calls": [{"decoder_action_ids": [[actions]]}],
    }
    return (
        {"num_examples": 1, "compatibility_key": "identity"},
        {"completed": {"example": record}},
    )


def test_center16_preflight_accepts_exact_tile_local_actions():
    summary, records = fixture()
    result = MODULE.validate(
        summary, records, "center16", admission_record_sha256="admission"
    )
    assert result["status"] == "pass"
    assert result["expanded_context_length"] == 48_478
    assert result["question_id"] == 0


def test_center16_preflight_rejects_global_or_reordered_static_selection():
    summary, records = fixture()
    records["completed"]["example"]["raw_decoder_calls"][0]["decoder_action_ids"][0][0][0], \
        records["completed"]["example"]["raw_decoder_calls"][0]["decoder_action_ids"][0][0][1] = (
            records["completed"]["example"]["raw_decoder_calls"][0]["decoder_action_ids"][0][0][1],
            records["completed"]["example"]["raw_decoder_calls"][0]["decoder_action_ids"][0][0][0],
        )
    with pytest.raises(ValueError, match="frozen STAViS"):
        MODULE.validate(summary, records, "center16")


def test_preflight_selects_question_zero_from_a_larger_resumed_run(tmp_path):
    summary, records = fixture()
    records["completed"]["later"] = {
        **records["completed"]["example"],
        "example_key": "later-key",
        "question_id": 7,
    }
    summary["num_examples"] = 8
    result = MODULE.validate(summary, records, "center16")
    attestation = tmp_path / "preflight_pass.json"
    MODULE.persist_or_validate_attestation(attestation, result)
    MODULE.persist_or_validate_attestation(attestation, result)
    assert json.loads(attestation.read_text()) == result

    changed = {**result, "expanded_context_length": result["expanded_context_length"] + 1}
    with pytest.raises(ValueError, match="differs"):
        MODULE.persist_or_validate_attestation(attestation, changed)
