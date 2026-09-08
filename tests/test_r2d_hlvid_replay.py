import importlib.util
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
        "identity_sha256": identity,
        "attempt_state": "completed_processor_observation",
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
    record["attempt_state"] = "started"
    with pytest.raises(ValueError, match="Non-complete"):
        MODULE.completed_replay_by_video([record], "frozen")
