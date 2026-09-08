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
                "question_count": count,
                "attempt_state": "completed_processor_observation",
                "identity_sha256": "identity",
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
    attempts = replay_dir / "attempts.jsonl"
    attempts.write_text(json.dumps({"attempt_state": "completed_processor_observation"}) + "\n")
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
        "identity_sha256": "identity",
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
        "results_jsonl": str(results),
        "results_jsonl_sha256": MODULE.sha256_file(results),
        "attempt_log": str(attempts),
        "attempt_log_sha256": MODULE.sha256_file(attempts),
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
    assert paths == [replay_dir / "summary.json", results, attempts]
