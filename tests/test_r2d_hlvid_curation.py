import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "human_gaze" / "curate_r2d_hlvid_secondary.py"
SPEC = importlib.util.spec_from_file_location("curate_r2d_hlvid_secondary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_legacy_tail_only_counter_summary_is_not_accepted_as_full_coverage():
    adapter = {
        "allocation_statistics": {
            "decoder": {
                "frame_observations": 322_560,
                "spatial_actions_per_frame_mean": 12.5,
            }
        }
    }
    assert MODULE.process_counter_scope_complete(adapter) is False


def test_explicit_uninterrupted_counter_coverage_is_accepted():
    adapter = {
        "observation_coverage": {
            "qa_examples_in_summary": 268,
            "qa_examples_observed_by_process_counters": 268,
            "resume_prefix_examples": 0,
            "complete": True,
        }
    }
    assert MODULE.process_counter_scope_complete(adapter) is True


def test_resumed_summary_is_rejected_even_if_qa_file_is_complete():
    adapter = {
        "observation_coverage": {
            "qa_examples_in_summary": 268,
            "qa_examples_observed_by_process_counters": 14,
            "resume_prefix_examples": 254,
            "complete": False,
        }
    }
    assert MODULE.process_counter_scope_complete(adapter) is False


def test_complete_validated_replay_covers_all_qa_keys(tmp_path):
    qa_rows = []
    replay_rows = []
    next_question = 0
    for video_index in range(77):
        count = 4 if video_index < 37 else 3
        question_ids = list(range(next_question, next_question + count))
        next_question += count
        video = f"video_{video_index:03d}.mp4"
        qa_rows.extend({"question_id": qid, "video_path": video} for qid in question_ids)
        replay_rows.append(
            {
                "video_path": video,
                "question_ids": question_ids,
                "question_keys": [MODULE.stable_example_key(qid, "test") for qid in question_ids],
                "question_count": count,
                "state": "completed_processor_observation",
                "compatibility_key": "identity",
                "video_key": MODULE.stable_video_key(video, "test"),
                "allocation_raw": {
                    "schema_version": 1,
                    "model_calls": 1,
                    "decoder": {
                        "calls": 1,
                        "frame_observations": 1,
                        "spatial_actions_sum": 4,
                        "padded_position_slots": 5,
                        "valid_action_tokens": 5,
                        "spatial_actions_min": 4,
                        "spatial_actions_max": 4,
                        "eos_actions": 1,
                        "spatial_action_histogram": {"4": 1},
                    },
                    "post_resolution_adaptation": {
                        "frame_observations": 1,
                        "retained_patches_sum": 16,
                        "padded_position_slots": 16,
                        "retained_patches_min": 16,
                        "retained_patches_max": 16,
                    },
                },
            }
        )
    assert next_question == 268
    replay_dir = tmp_path / "seed440826_variable"
    replay_dir.mkdir()
    results = replay_dir / "results.jsonl"
    results.write_text("".join(json.dumps(row) + "\n" for row in replay_rows))
    evidence = replay_dir / "evidence.jsonl"
    complete_counter = {
        "availability": "complete",
        "observed_count": 1,
        "expected_observations": 1,
        "observed_sum": 1,
        "mean": 1,
        "min": 1,
        "max": 1,
        "unit": "count",
        "source": "processor_replay",
    }
    evidence_rows = []
    for row in qa_rows:
        evidence_rows.append(
            {
                "schema_version": 1,
                "example_key": MODULE.stable_example_key(row["question_id"], "test"),
                "video_key": MODULE.stable_video_key(row["video_path"], "test"),
                "attempt_id": f"attempt-{row['question_id']}",
                "compatibility_key": "identity",
                "state": "completed_answer",
                "measurement_origin": "processor_replay",
                "answer": {"source_row": row},
                "counters": {
                    name: dict(complete_counter)
                    for name in (
                        "decoder_spatial_actions",
                        "retained_patches",
                        "expanded_visual_tokens",
                        "expanded_context_tokens",
                    )
                },
            }
        )
    evidence.write_text("".join(json.dumps(row) + "\n" for row in evidence_rows))
    audit_dir = tmp_path / "protocol_audit"
    audit_dir.mkdir()
    audit_files = {
        "summary_sha256": audit_dir / "summary.json",
        "manifest_sha256": audit_dir / "protocol_runtime_manifest.json",
        "decode_audit_sha256": audit_dir / "decode_audit.jsonl",
        "supplement_sha256": audit_dir / "live_runtime_supplement.json",
    }
    for path in audit_files.values():
        path.write_text("{}\n")
    weighted = MODULE.merge_raw_allocations(
        (row["allocation_raw"], row["question_count"]) for row in replay_rows
    )
    summary = {
        "schema_version": 1,
        "base_seed": 440826,
        "training_seed": 740826,
        "mode": "variable",
        "record_type": "r2d_hlvid_allocation_replay_summary",
        "status": "complete",
        "nvila_generation_calls": 0,
        "compatibility_key": "identity",
        "checkpoint": {"model_sha256": "model", "config_sha256": "config"},
        "calibration": {"sha256": "calibration"},
        "protocol": dict(MODULE.EXPECTED_PROTOCOL),
        "coverage": {
            "unique_videos_observed": 77,
            "unique_videos_expected": 77,
            "question_keys_observed": 268,
            "question_keys_expected": 268,
            "complete": True,
        },
        "validation": {
            "status": "pass",
            "criterion": "exact_allocation_statistics_match",
        },
        "weighted_allocation_raw": weighted,
        "weighted_allocation_statistics": MODULE.summarize_raw_allocation(weighted),
        "counters": {
            name: dict(complete_counter)
            for name in (
                "decoder_spatial_actions",
                "retained_patches",
                "expanded_visual_tokens",
                "expanded_context_tokens",
            )
        },
        "results_jsonl": str(results),
        "results_jsonl_sha256": MODULE.sha256_file(results),
        "evidence_jsonl": str(evidence),
        "evidence_jsonl_sha256": MODULE.sha256_file(evidence),
        "protocol_audit": {
            "directory": str(audit_dir),
            **{name: MODULE.sha256_file(path) for name, path in audit_files.items()},
        },
    }
    (replay_dir / "summary.json").write_text(json.dumps(summary))
    adapter = {
        "training_seed": 740826,
        "checkpoint": {"model_sha256": "model", "config_sha256": "config"},
        "eos_calibration": {"sha256": "calibration"},
    }

    loaded, paths = MODULE.load_variable_replay(
        tmp_path,
        seed=440826,
        qa_rows=qa_rows,
        qa_adapter=adapter,
    )

    assert loaded["weighted_allocation_statistics"]["decoder"][
        "spatial_actions_per_frame_mean"
    ] == 4.0
    assert paths == [
        replay_dir / "summary.json",
        results,
        evidence,
        audit_dir / "summary.json",
        audit_dir / "protocol_runtime_manifest.json",
        audit_dir / "decode_audit.jsonl",
        audit_dir / "live_runtime_supplement.json",
    ]
