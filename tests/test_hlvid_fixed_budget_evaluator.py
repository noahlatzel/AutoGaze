import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


SCRIPT = Path(__file__).parents[1] / "scripts" / "human_gaze" / "evaluate_hlvid_nvila_fixed_budget.py"
SPEC = importlib.util.spec_from_file_location("hlvid_fixed_budget_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_letter_requires_standalone_option():
    assert MODULE.extract_letter("Answer: c") == "C"
    assert MODULE.extract_letter("The answer is (D).") == "D"
    assert MODULE.extract_letter("CAB") == ""
    assert MODULE.extract_letter("") == ""


def test_summarize_reports_official_micro_and_video_macro():
    records = [
        {"video_path": "v1.mp4", "category": "av", "prediction_letter": "A", "is_correct": True},
        {"video_path": "v1.mp4", "category": "av", "prediction_letter": "", "is_correct": False},
        {"video_path": "v2.mp4", "category": "household", "prediction_letter": "B", "is_correct": True},
    ]
    result = MODULE.summarize(records)
    assert result["num_examples"] == 3
    assert result["num_correct"] == 2
    assert result["accuracy"] == 2 / 3
    assert result["macro_video_accuracy"] == 0.75
    assert result["num_videos"] == 2
    assert result["invalid_prediction_count"] == 1
    assert result["by_category"]["av"]["accuracy"] == 0.5
    assert result["by_category"]["household"]["accuracy"] == 1.0


AGG_SCRIPT = Path(__file__).parents[1] / "scripts" / "human_gaze" / "aggregate_hlvid_fixed_budget_all_seeds.py"
AGG_SPEC = importlib.util.spec_from_file_location("hlvid_fixed_budget_aggregate", AGG_SCRIPT)
AGG = importlib.util.module_from_spec(AGG_SPEC)
assert AGG_SPEC.loader is not None
AGG_SPEC.loader.exec_module(AGG)


def test_seed_interval_uses_frozen_t_interval():
    low, high = AGG.seed_interval([0.4, 0.5, 0.6])
    assert low < 0.4
    assert high > 0.6
    assert abs((low + high) / 2 - 0.5) < 1e-12


def test_cluster_bootstrap_difference_preserves_pairing():
    rows_a = [[
        {"question_id": 1, "video_path": "v1", "is_correct": True},
        {"question_id": 2, "video_path": "v2", "is_correct": True},
    ]] * 3
    rows_b = [[
        {"question_id": 1, "video_path": "v1", "is_correct": False},
        {"question_id": 2, "video_path": "v2", "is_correct": True},
    ]] * 3
    result = AGG.cluster_bootstrap_difference(rows_a, rows_b, iterations=200, seed=7)
    assert result["difference"] == 0.5
    assert result["ci90_low"] <= result["difference"] <= result["ci90_high"]
    assert result["num_video_clusters"] == 2


def test_checkpoint_compatibility_accepts_only_known_zero_buffer(tmp_path):
    bias_name = "gazing_model.gaze_decoder.output_token_logit_bias"
    checkpoint = tmp_path / "model.safetensors"
    save_file({"saved.weight": torch.ones(2)}, checkpoint)

    class FakeModel:
        def __init__(self, bias):
            self.bias = bias

        def state_dict(self):
            return {"saved.weight": torch.ones(2), bias_name: self.bias}

    result = MODULE.verify_loaded_checkpoint(FakeModel(torch.zeros(3)), checkpoint)
    assert result["accepted_runtime_only_keys"] == [bias_name]
    assert result["max_absolute_value"] == 0.0
    with pytest.raises(ValueError, match="compatibility buffer is not zero"):
        MODULE.verify_loaded_checkpoint(FakeModel(torch.ones(3)), checkpoint)


def test_attempt_record_resume_recovers_only_completed_answers(tmp_path):
    records = tmp_path / "records"
    compatibility_key = "abc123"

    def evidence_record(question_id, state, attempt_id, **extra):
        return {
            "schema_version": 1,
            "example_key": MODULE.stable_example_key(question_id, "test"),
            "video_key": MODULE.stable_video_key("video.mp4", "test"),
            "attempt_id": attempt_id,
            "compatibility_key": compatibility_key,
            "state": state,
            "protocol_id": MODULE.PROTOCOL_ID,
            "question_id": question_id,
            **extra,
        }

    MODULE.write_evidence_record(
        records, evidence_record(1, "attempt_started", "attempt-1")
    )
    MODULE.write_evidence_record(
        records,
        evidence_record(
            1,
            "completed_answer",
            "attempt-1",
            answer={"question_id": 1, "prediction_letter": "A"},
            context={"expanded_context_length": 101},
        ),
    )
    MODULE.write_evidence_record(
        records, evidence_record(2, "attempt_started", "attempt-2")
    )

    answers, telemetry = MODULE.load_completed_evidence(
        records, compatibility_key=compatibility_key
    )
    assert answers == {1: {"question_id": 1, "prediction_letter": "A"}}
    assert telemetry[1]["context"]["expanded_context_length"] == 101


def test_stale_compact_results_are_recoverable_from_durable_evidence():
    completed = {
        1: {"question_id": 1, "prediction_letter": "A"},
        2: {"question_id": 2, "prediction_letter": "B"},
    }
    MODULE.validate_compact_results({1: completed[1]}, completed)


def test_compact_results_cannot_outrun_or_disagree_with_durable_evidence():
    completed = {1: {"question_id": 1, "prediction_letter": "A"}}
    with pytest.raises(ValueError, match="without durable evidence"):
        MODULE.validate_compact_results(
            {2: {"question_id": 2, "prediction_letter": "B"}}, completed
        )
    with pytest.raises(ValueError, match="differs from durable evidence"):
        MODULE.validate_compact_results(
            {1: {"question_id": 1, "prediction_letter": "D"}}, completed
        )


def test_resume_manifest_rejects_protocol_or_checkpoint_drift(tmp_path):
    path = tmp_path / "run_manifest.json"
    identity = {"protocol_id": "v1", "checkpoint": "sha-a"}
    first_hash = MODULE.ensure_manifest(path, identity, resume=True)
    assert json.loads(path.read_text())["compatibility_key"] == first_hash
    assert MODULE.ensure_manifest(path, identity, resume=True) == first_hash

    with pytest.raises(ValueError, match="Resume-incompatible"):
        MODULE.ensure_manifest(
            path,
            {"protocol_id": "v1", "checkpoint": "sha-b"},
            resume=True,
        )
