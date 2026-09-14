import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "human_gaze" / "curate_r2d_hlvid_secondary.py"
SPEC = importlib.util.spec_from_file_location("curate_r2d_hlvid_secondary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def qa_fixture():
    return [{"question_id": qid, "prediction": "A", "prediction_letter": "A", "answer": "A", "is_correct": True} for qid in range(268)]


def test_complete_accuracy_requires_all_unique_answers_and_exact_types():
    MODULE.validate_qa_rows(qa_fixture())
    for corrupt in ("duplicate", "missing", "bool_id", "string_correct", "wrong_parser"):
        rows = qa_fixture()
        if corrupt == "duplicate":
            rows[-1] = dict(rows[0])
        elif corrupt == "missing":
            rows.pop()
        elif corrupt == "bool_id":
            rows[0]["question_id"] = False
        elif corrupt == "string_correct":
            rows[0]["is_correct"] = "false"
        else:
            rows[0]["prediction_letter"] = "B"
        with pytest.raises(ValueError):
            MODULE.validate_qa_rows(rows)


def test_missing_variable_cost_is_not_imputed_or_averaged_over_partial_seeds():
    allocation = MODULE.unavailable_variable_allocation(False)
    assert allocation["statistics"]["decoder"]["spatial_actions_per_frame_mean"] is None
    assert allocation["mean_visual_tokens"] is None
    assert MODULE.complete_mean([16.0, None, 12.0]) is None
    assert MODULE.complete_mean([16.0, 16.0, 16.0]) == 16.0


def test_micro_bootstrap_clusters_keep_question_weights_and_pairing():
    videos = ["short", "long"]
    per_video = {(seed, mode): {"short": 1.0 if mode == "variable" else 0.0, "long": 0.0 if mode == "variable" else 1.0} for seed in MODULE.SEEDS for mode in MODULE.MODES}
    macro = MODULE.bootstrap_delta(per_video, videos, replicates=100)
    micro = MODULE.bootstrap_delta(per_video, videos, {"short": 1, "long": 9}, replicates=100)
    assert micro["estimand"] == "question_micro" and micro["seed_resampling"] is False
    assert macro["point_estimate"] == 0.0
    assert micro["point_estimate"] == pytest.approx(-0.8)
    # Degenerate perfect within-video pairing stays deterministic, irrespective of seed.
    identical = {(seed, mode): {video: 0.5 for video in videos} for seed in MODULE.SEEDS for mode in MODULE.MODES}
    assert MODULE.bootstrap_delta(identical, videos, {"short": 1, "long": 9}, replicates=100)["interval"] == [0.0, 0.0]
    assert macro["replicates"] == micro["replicates"] == 100


def test_accuracy_only_cli_keeps_complete_pairs_and_null_costs(tmp_path, monkeypatch):
    config_path = SCRIPT.parents[2] / "experiments/human_gaze/configs/r2f_r2d_hlvid_secondary.yaml"
    config = yaml.safe_load(config_path.read_text())
    qa = qa_fixture()
    for row in qa:
        row.update(video_path=f"v{row['question_id'] % 77}", category="av")
    for arm in config["arms"]:
        seed = arm["base_seed"]
        for mode in MODULE.MODES:
            directory = tmp_path / "source" / "rollouts" / f"seed{seed}_{mode}"
            directory.mkdir(parents=True)
            adapter = {
                "mode": mode, "base_seed": seed, "training_seed": arm["training_seed"],
                "execution": {"code_commit": "qa-source", "code_dirty": False},
                "checkpoint": {"model_sha256": arm["checkpoint_model_sha256"], "config_sha256": arm["checkpoint_config_sha256"]},
                "eos_calibration": {"sha256": arm["calibration_sha256"], "split": "train_source_balanced_fixed_subset", "target_mean_tokens": 16.0, "selected_mean_tokens": arm["calibration_mean_tokens"]},
                "action_contract": {"fine_action_ids": [69, 264], "eos_action_id": 265, "variable_min_spatial_actions": 4, "variable_max_generation_tokens": 36, "forced_spatial_actions": 16, "calibrated_eos_logit_bias": arm["eos_logit_bias"]},
                "benchmark_protocol": {**MODULE.EXPECTED_PROTOCOL, "dataset_split": "test", "dataset_parquet_sha256": config["dataset"]["parquet_sha256"]},
            }
            summary = {"num_examples": 268, "accuracy": 1.0, "model_path": config["models"]["nvila"], "r2d_hlvid_adapter": adapter}
            for key in ("max_batch_size_autogaze", "max_batch_size_siglip", "torch_dtype"):
                summary[key] = config["established_protocol"][key]
            (directory / "summary.json").write_text(json.dumps(summary))
            (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in qa))
    monkeypatch.setattr(MODULE.subprocess, "check_output", lambda cmd, **kwargs: "" if "status" in cmd else "curator-source")
    bootstrap = MODULE.bootstrap_delta
    monkeypatch.setattr(MODULE, "bootstrap_delta", lambda *args: bootstrap(*args, replicates=20))
    argv = [str(SCRIPT), "--run-id", "source", "--artifact-root", str(tmp_path), "--config", str(config_path), "--code-commit", "qa-source", "--output-dir", str(tmp_path / "result"), "--accuracy-only"]
    monkeypatch.setattr(MODULE.sys if hasattr(MODULE, "sys") else __import__("sys"), "argv", argv)
    MODULE.main()
    metrics = json.loads((tmp_path / "result/metrics.json").read_text())
    assert metrics["accuracy_status"] == "complete_three_seed_pairs"
    assert metrics["aggregate"]["variable"]["mean_decoder_spatial_actions_over_seeds"] is None
    assert metrics["aggregate"]["forced_k16"]["mean_decoder_spatial_actions_over_seeds"] == 16.0
    assert len((tmp_path / "result/paired_questions.csv").read_text().splitlines()) == 805
    # Removing a single counterpart must stop even explicit accuracy-only curation.
    (tmp_path / "source/rollouts/seed440828_forced_k16/results.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete HLVid arm"):
        MODULE.main()


def test_default_curator_still_requires_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__("sys"), "argv", [str(SCRIPT), "--run-id", "source", "--artifact-root", str(tmp_path), "--config", "config.yaml", "--code-commit", "source", "--output-dir", str(tmp_path / "result")])
    with pytest.raises(SystemExit) as error:
        MODULE.main()
    assert error.value.code == 2
    assert not (tmp_path / "result").exists()


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
