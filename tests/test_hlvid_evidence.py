import json

import pytest

from scripts.runners.hlvid_evidence import (
    counter_summary,
    export_evidence_jsonl,
    legacy_decode_map,
    load_resume_state,
    merge_counter_summaries,
    resume_compatibility_key,
    stable_example_key,
    stable_video_key,
    summarize_gaze_slots,
    unavailable_counter,
    write_evidence_record,
)
from scripts.runners.audit_hlvid_protocol import audit_video, parse_args, requested_indices, summarize_audit


def record(state="attempt_started", *, question=3, attempt="attempt-1", compatibility="policy-a"):
    result = {
        "schema_version": 1, "example_key": stable_example_key(question, "test"),
        "video_key": stable_video_key("videos/example.mp4", "test"),
        "attempt_id": attempt, "compatibility_key": compatibility, "state": state,
    }
    if state == "completed_answer":
        result["answer"] = {"prediction": "B", "question_id": question}
    return result


def test_identity_is_stable_but_split_policy_and_question_sensitive():
    assert stable_example_key(3, "test") == stable_example_key(3, "test")
    assert stable_example_key(3, "test") != stable_example_key(3, "train")
    assert stable_example_key(3, "test") != stable_example_key(4, "test")
    assert stable_video_key("videos\\a.mp4", "test") == stable_video_key("videos/a.mp4", "test")
    identity = {"policy": "pretrained-k16", "checkpoint_sha256": "a", "protocol": {"frames": 128}}
    assert resume_compatibility_key(identity) == resume_compatibility_key(dict(reversed(list(identity.items()))))
    assert resume_compatibility_key(identity) != resume_compatibility_key({**identity, "policy": "center16"})


def test_started_and_failed_attempts_do_not_authorize_qa_skip(tmp_path):
    write_evidence_record(tmp_path, record())
    write_evidence_record(tmp_path, record("failed"))
    state = load_resume_state(tmp_path, "policy-a")
    assert state["completed"] == {}
    assert len(state["records"]) == 2
    write_evidence_record(tmp_path, record("completed_answer", attempt="attempt-2"))
    state = load_resume_state(tmp_path, "policy-a")
    assert len(state["completed"]) == 1
    assert next(iter(state["completed"].values()))["answer"]["prediction"] == "B"


def test_failed_attempt_requires_new_attempt_identity_for_completion(tmp_path):
    write_evidence_record(tmp_path, record("failed"))
    with pytest.raises(ValueError, match="terminal state"):
        write_evidence_record(tmp_path, record("completed_answer"))


def test_derived_jsonl_preserves_answer_identity_and_can_exclude_attempts(tmp_path):
    records = tmp_path / "records"
    write_evidence_record(records, record())
    write_evidence_record(records, record("completed_answer"))
    output = export_evidence_jsonl(records, tmp_path / "completed.jsonl", "policy-a", completed_only=True)
    state = load_resume_state(output, "policy-a")
    assert len(state["records"]) == 1
    assert len(state["completed"]) == 1


def test_incomplete_temporary_write_is_not_a_completed_record(tmp_path):
    (tmp_path / ".pending-interrupted").write_text('{"state":"completed_answer"')
    assert not load_resume_state(tmp_path, "policy-a")["completed"]
    write_evidence_record(tmp_path, record("completed_answer"))
    assert len(load_resume_state(tmp_path, "policy-a")["completed"]) == 1


def test_incompatible_resume_and_duplicate_completion_are_rejected(tmp_path):
    write_evidence_record(tmp_path, record("completed_answer"))
    with pytest.raises(ValueError, match="incompatible"):
        load_resume_state(tmp_path, "policy-b")
    with pytest.raises(ValueError, match="already has"):
        write_evidence_record(tmp_path, record("completed_answer", attempt="attempt-2"))


def test_jsonl_rejects_duplicate_completions_and_truncated_tail(tmp_path):
    path = tmp_path / "evidence.jsonl"
    first = record("completed_answer")
    second = record("completed_answer", attempt="attempt-2")
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="duplicate completed"):
        load_resume_state(path, "policy-a")
    path.write_text(json.dumps(first) + '\n{"state":')
    with pytest.raises(ValueError, match="line 2"):
        load_resume_state(path, "policy-a")


def test_answer_and_measurements_share_one_durable_record(tmp_path):
    row = record("completed_answer")
    row["counters"] = {"retained_patches": counter_summary([64, 60], expected_observations=2)}
    path = write_evidence_record(tmp_path, row)
    assert json.loads(path.read_text()) == row
    incomplete = record("completed_answer", question=4)
    del incomplete["answer"]
    with pytest.raises(ValueError, match="answer payload"):
        write_evidence_record(tmp_path, incomplete)


def test_variable_valid_spatial_counts_exclude_padding_and_eos():
    counts = summarize_gaze_slots(
        [3, 4],
        [[False, False, False, False, False, True, True],
         [False, True, True, False, False, False, False]],
        gazing_pos=[[69, 70, 265, 334, 530, 530, 530],
                    [71, 265, 265, 335, 336, 337, 530]],
        actions_per_frame=265, fine_action_offset=69,
    )
    assert counts["padded_slot_sum"] == 14
    assert counts["valid_counts_per_sample_frame"] == [[3, 2], [1, 4]]
    assert counts["spatial_counts_per_sample_frame"] == [[2, 1], [1, 3]]
    assert counts["spatial_entry_sum"] == 7
    assert counts["frame_observations"] == 4


def test_post_adaptation_valid_counts_do_not_use_batch_maximum():
    counts = summarize_gaze_slots([4, 3], [[False] * 7, [False, False, True, True, False, True, True]])
    assert counts["valid_counts_per_sample_frame"] == [[4, 3], [2, 1]]
    assert counts["valid_entry_sum"] == 10
    assert counts["padded_slot_sum"] == 14


@pytest.mark.parametrize("lengths,padding", [([2], [[False]]), ([2], [[0, 1]]), ([-1], [[]])])
def test_invalid_gaze_shapes_fail_instead_of_emitting_plausible_counts(lengths, padding):
    with pytest.raises(ValueError):
        summarize_gaze_slots(lengths, padding)


def test_resumed_counter_tail_remains_partial_and_merges_by_observation_count():
    tail = counter_summary([12, 20], expected_observations=8)
    assert tail["availability"] == "partial"
    missing = unavailable_counter("Killed prefix did not persist counters", expected_observations=6)
    merged = merge_counter_summaries([missing, counter_summary([12, 20], expected_observations=2)])
    assert merged["availability"] == "partial"
    assert merged["expected_observations"] == 8
    assert merged["observed_count"] == 2
    assert merged["observed_sum"] == 32
    completed = merge_counter_summaries([counter_summary([10, 20]), counter_summary([60])])
    assert completed["mean"] == 30


def test_counter_unavailable_is_not_zero_and_mixed_units_fail():
    missing = merge_counter_summaries([unavailable_counter("No historical metadata")])
    assert missing["observed_sum"] is None
    with pytest.raises(ValueError, match="different units"):
        merge_counter_summaries([counter_summary([1], unit="seconds"), counter_summary([1], unit="patches")])
    with pytest.raises(ValueError, match="finite"):
        counter_summary([float("nan")])


def test_decode_map_compacts_failed_slots_before_tail_padding():
    result = legacy_decode_map([0, 1, 2, 3, 4], [0, 2, 3, 4])
    assert result["effective_indices"] == [0, 2, 3, 4, 4]
    assert result["tail_padding_count"] == 1
    assert result["failed_unique_indices"] == [1]
    assert [entry["slot"] for entry in result["substitutions"]] == [1, 2, 3, 4]


def test_decode_map_handles_repeated_requests_and_all_failed():
    result = legacy_decode_map([0, 0, 1, 1, 2], [0, 2])
    assert result["effective_indices"] == [0, 0, 2, 2, 2]
    assert result["tail_padding_count"] == 2
    missing = legacy_decode_map([0, 1], [])
    assert not missing["decode_usable"]
    assert missing["effective_indices"] == []


def test_streaming_audit_validates_tail_reads_unique_indices_and_releases_reader():
    class Reader:
        reported_count = 7

        def __init__(self, _):
            self.reads = []
            self.closed = False

        def can_grab(self, index):
            return index <= 4

        def read(self, index):
            self.reads.append(index)
            return {"read_success": index != 1, "rgb_conversion_success": index != 1,
                    "image_conversion_success": index != 1}

        def close(self):
            self.closed = True

    reader = Reader(None)
    result = audit_video("unused", num_frames=9, reader_factory=lambda _: reader)
    assert result["validated_frame_count"] == 5
    assert [probe["index"] for probe in result["tail_probes"]] == [6, 5, 4]
    assert reader.reads == [0, 1, 2, 3, 4]
    assert len(result["effective_indices"]) == 9
    assert reader.closed


def test_failed_duplicate_request_is_retried_and_reconstructed_without_substitution():
    class Reader:
        reported_count = 2

        def __init__(self):
            self.reads = []

        def can_grab(self, index):
            return True

        def read(self, index):
            self.reads.append(index)
            return {"read_success": len(self.reads) != 1,
                    "rgb_conversion_success": len(self.reads) != 1,
                    "image_conversion_success": len(self.reads) != 1}

        def close(self):
            pass

    reader = Reader()
    result = audit_video("unused", num_frames=3, reader_factory=lambda _: reader)
    assert result["intended_indices"] == [0, 0, 1]
    assert reader.reads == [0, 0, 1]
    assert result["effective_indices"] == [0, 0, 1]
    assert result["substitutions"] == []
    assert result["sampled_read_failure"]
    assert not result["legacy_output_substituted"]


def test_rgb_conversion_failure_is_fatal_and_is_not_silently_padded():
    class Reader:
        reported_count = 3

        def can_grab(self, index):
            return True

        def read(self, index):
            return {"read_success": True, "rgb_conversion_success": False,
                    "image_conversion_success": None, "fatal_error": "RGB conversion failed"}

        def close(self):
            pass

    result = audit_video("unused", num_frames=3, reader_factory=lambda _: Reader())
    assert not result["decode_usable"]
    assert result["effective_indices"] == []
    assert len(result["read_results"]) == 1
    assert not result["sampled_read_failure"]
    assert result["error"] == "RGB conversion failed"


def test_metadata_count_correction_alone_does_not_flag_qa_reuse():
    record = {"decode_usable": True, "metadata_count_corrected": True,
              "sampled_read_failure": False, "legacy_output_substituted": False,
              "question_rows": [0, 1]}
    summary = summarize_audit([record], 2, True)
    assert summary["affected_question_rows"] == []
    assert summary["reuse_qualification"] == "no_current_requested_decode_failures"
    assert not summary["historical_certification"]


def test_requested_grid_uses_rounding_and_preserves_short_video_duplicates():
    assert requested_indices(3, 9) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert requested_indices(1, 128) == [0] * 128


def test_frozen_audit_cli_rejects_shorter_sampling_before_loading_inputs(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["audit_hlvid_protocol.py", "--output-dir", "unused", "--num-frames", "16"])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
    assert "requires --num-frames=128" in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", ["audit_hlvid_protocol.py", "--output-dir", "unused"])
    assert parse_args().num_frames == 128
