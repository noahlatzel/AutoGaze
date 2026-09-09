import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "human_gaze" / "replay_r2d_hlvid_allocation.py"
SPEC = importlib.util.spec_from_file_location("replay_r2d_hlvid_allocation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def completed(video, identity="frozen"):
    return {
        "video_path": video,
        "compatibility_key": identity,
        "state": "completed_processor_observation",
    }


def test_resume_accepts_only_unique_completed_records_with_same_identity():
    records = [completed("a.mp4"), completed("b.mp4")]
    assert set(MODULE.completed_replay_by_video(records, "frozen")) == {"a.mp4", "b.mp4"}


def test_resume_rejects_duplicate_completion():
    with pytest.raises(ValueError, match="Duplicate completed"):
        MODULE.completed_replay_by_video([completed("a.mp4"), completed("a.mp4")], "frozen")


def test_resume_rejects_identity_mismatch():
    with pytest.raises(ValueError, match="identity mismatch"):
        MODULE.completed_replay_by_video([completed("a.mp4", "other")], "frozen")


def test_started_attempt_is_not_a_completed_replay_record():
    record = completed("a.mp4")
    record["state"] = "attempt_started"
    with pytest.raises(ValueError, match="Non-complete"):
        MODULE.completed_replay_by_video([record], "frozen")


def test_shared_counter_schema_keeps_observed_count_and_sum():
    counter = MODULE.raw_counter(
        observed_count=2,
        observed_sum=24,
        minimum=8,
        maximum=16,
        unit="actions_per_frame_observation",
    )
    assert counter == {
        "availability": "complete",
        "observed_count": 2,
        "expected_observations": 2,
        "observed_sum": 24,
        "mean": 12,
        "min": 8,
        "max": 16,
        "unit": "actions_per_frame_observation",
        "source": "processor_replay",
    }


def write_protocol_audit(directory, *, qualification="no_current_requested_decode_failures"):
    directory.mkdir()
    summary = {
        "video_count": 1,
        "affected_question_rows": [],
        "reuse_qualification": qualification,
        "historical_certification": False,
    }
    manifest = {
        "dataset_sha256": MODULE.EXPECTED_DATASET_SHA256,
        "legacy_loader_sha256": MODULE.EXPECTED_RUNNER_SHA256,
        "protocol": {"num_video_frames": 128},
        "live_model_files_verified": False,
    }
    record = {"manifest_path": "a.mp4", "video_key": "video", "decode_usable": True}
    paths = {
        "summary": directory / "summary.json",
        "protocol_runtime_manifest": directory / "protocol_runtime_manifest.json",
        "decode_audit": directory / "decode_audit.jsonl",
    }
    paths["summary"].write_text(json.dumps(summary) + "\n")
    paths["protocol_runtime_manifest"].write_text(json.dumps(manifest) + "\n")
    paths["decode_audit"].write_text(json.dumps(record) + "\n")
    supplement = {
        "artifact_hashes": {
            name: {"sha256": MODULE.sha256_file(path)} for name, path in paths.items()
        },
        "reuse_qualification": qualification,
        "runtime": {"numpy_version": "tested"},
        "git": {"commit": "audit", "dirty": False},
    }
    (directory / "audit_supplement.json").write_text(json.dumps(supplement) + "\n")


def test_protocol_audit_requires_supplement_hashes_and_clean_decode_qualification(tmp_path):
    accepted = tmp_path / "accepted"
    write_protocol_audit(accepted)
    summary, records, provenance = MODULE.load_protocol_audit(accepted)
    assert summary["reuse_qualification"] == "no_current_requested_decode_failures"
    assert set(records) == {"a.mp4"}
    assert provenance["supplement_sha256"] == MODULE.sha256_file(
        accepted / "audit_supplement.json"
    )

    rejected = tmp_path / "rejected"
    write_protocol_audit(rejected, qualification="inspect_affected_subset")
    with pytest.raises(ValueError, match="affected-subset"):
        MODULE.load_protocol_audit(rejected)
