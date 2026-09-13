import json
from pathlib import Path

import pytest

from autogaze.human_gaze.checkpoint_provenance import verify_supervised_checkpoint
from autogaze.human_gaze.supervised_restart import (
    FAILED_INVENTORY_SHA256, REPAIR_COMMIT, load_failed_attempts,
    restart_identity, verify_published_restart_source,
)
from autogaze.human_gaze.supervised_analysis import sha256_file
from scripts.human_gaze.curate_supervised_k16_comparison import supervised_training_wall, validate_resource_receipt
from scripts.human_gaze.verify_supervised_k16_execution import load_execution_config
from tests.test_supervised_k16_checkpoint_provenance import make_fixture, write_json, SEED


ROOT = Path(__file__).resolve().parents[1]


def restart_fixture(tmp_path):
    inventory, training_root, paths = make_fixture(tmp_path)
    run_id = "20260914-0150_supervised-k16-comparison_13179be"
    inventory["authoritative_sources"].update({
        "supervised_execution_kind": "same_seed_fresh_restart",
        "supervised_execution_commit": REPAIR_COMMIT, "supervised_run_id": run_id,
    })
    execution_path = paths["seed_root"] / "execution_manifest.json"
    execution = json.loads(execution_path.read_text())
    execution["run_id"] = run_id
    execution["source"] = {"execution_commit": REPAIR_COMMIT, "dirty": False,
                           "source_chain": verify_published_restart_source(ROOT, REPAIR_COMMIT)}
    execution["restart_of"] = restart_identity(ROOT, SEED, verify_live=False)
    execution["slurm"] = {"SLURM_ARRAY_JOB_ID": "1706000", "SLURM_ARRAY_TASK_ID": "0"}
    write_json(execution_path, execution)
    return inventory, training_root, paths


def test_restart_inherits_exact_original_recipe_and_all_six_seeds():
    original = load_execution_config(ROOT / "experiments/human_gaze/configs/supervised_k16_execution.yaml")
    restart = load_execution_config(ROOT / "experiments/human_gaze/configs/supervised_k16_restart_execution.yaml")
    for key in ("data", "initialization", "training", "base_seeds", "continuation_seeds", "admission"):
        assert restart[key] == original[key]
    assert restart["resources"]["host_memory_gib_per_task"] == 32
    failed = load_failed_attempts(ROOT, verify_live=False)
    assert sum(row["process_elapsed_seconds"] for row in failed["seeds"]) == pytest.approx(770.41)
    assert failed["checkpoint_present"] is False
    assert restart["restart"]["failed_attempt_inventory_sha256"] == FAILED_INVENTORY_SHA256


def test_all_five_restart_bundles_bind_source_seed_and_original_failed_attempt(tmp_path, monkeypatch):
    # Live Linux evidence verification is a separate precheck. CPU fixtures do
    # not depend on heavy remote paths.
    import autogaze.human_gaze.supervised_restart as lineage
    real_identity = lineage.restart_identity
    monkeypatch.setattr(lineage, "restart_identity", lambda root, seed: real_identity(root, seed, verify_live=False))
    inventory, training_root, paths = restart_fixture(tmp_path)
    for step in (2315, 5000, 10000, 15000, 20000):
        _checkpoint, provenance = verify_supervised_checkpoint(inventory, training_root=training_root, base_seed=SEED, cumulative_update=step)
        assert provenance["restart_of"]["failed_array_element"] == "1702710_0"
        assert provenance["restart_of"]["failed_completed_updates"] == 100
    assert supervised_training_wall(inventory, training_root, SEED)["points"] == {}
    execution_path = paths["seed_root"] / "execution_manifest.json"
    execution = json.loads(execution_path.read_text())
    execution["restart_of"]["base_seed"] = SEED + 1
    write_json(execution_path, execution)
    with pytest.raises(ValueError, match="same-seed"):
        verify_supervised_checkpoint(inventory, training_root=training_root, base_seed=SEED, cumulative_update=20000)


def test_restart_resource_receipt_cannot_omit_failed_allocation(tmp_path):
    inventory, training_root, paths = restart_fixture(tmp_path)
    root = paths["seed_root"]
    execution = json.loads((root / "execution_manifest.json").read_text())
    for name in ("stage1_time", "stage2_time", "gpu_telemetry"):
        path = root / f"{name}.txt"
        path.write_text(name)
        execution["resource_records"][name] = {"path": str(path), "sha256": sha256_file(path)}
    write_json(root / "execution_manifest.json", execution)
    sacct = {"schema_version": 1, "status": "complete", "base_seed": SEED,
             "terminal_state": "COMPLETED", "exit_code": "0:0", "allocated_gpu_count": 1,
             "requested_host_memory_bytes": 32 * 1024**3, "elapsed_seconds": 100,
             "max_rss_bytes": 1000, "gpu_memory_peak_bytes": 1000,
             "attempts": [{"job_id": "1706000_0", "elapsed_seconds": 100, "allocated_gpu_count": 1}]}
    write_json(root / "final_sacct.json", sacct)
    complete, detail = validate_resource_receipt(root, SEED)
    assert not complete and "recovery_attempt_job_ids" in detail["problems"]
    sacct["attempts"].append({"job_id": "1702710_0", "elapsed_seconds": 250, "allocated_gpu_count": 1})
    write_json(root / "final_sacct.json", sacct)
    complete, _detail = validate_resource_receipt(root, SEED)
    assert complete
