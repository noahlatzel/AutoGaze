import json
from pathlib import Path

import pytest
import torch

from autogaze.human_gaze.checkpoint_provenance import (
    CELL_MASS_SHA256,
    EXECUTION_COMMIT,
    MANIFEST_SHA256,
    RUN_ID,
    expected_resume_contract,
    load_checkpoint_inventory,
    verify_checkpoint_for_method,
    verify_rl_checkpoint,
    verify_supervised_checkpoint,
)
from autogaze.human_gaze.supervised_analysis import sha256_file
from autogaze.supervised_checkpoint import write_supervised_completion
from scripts.human_gaze.curate_supervised_k16_comparison import supervised_training_wall


SEED = 440826
TRAINING_SEED = 540826
FIXED = {
    2315: ("stage1", 2315, (5, 0), "latest", True),
    5000: ("stage2", 2685, (5, 1480), "periodic", False),
    10000: ("stage2", 7685, (16, 1108), "periodic", False),
    15000: ("stage2", 12685, (27, 736), "periodic", False),
    20000: ("stage2", 17685, (38, 364), "latest", True),
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_model(path: Path, payload: bytes) -> None:
    path.mkdir(parents=True)
    write_json(
        path / "config.json",
        {
            "gaze_model_config": {
                "num_vision_tokens_each_frame": 265,
                "gaze_decoder_config": {
                    "vocab_size": 266,
                    "eos_token_id": 265,
                    "num_multi_token_pred": 10,
                },
            }
        },
    )
    (path / "model.safetensors").write_bytes(payload)
    write_json(path / "preprocessor_config.json", {"size": 224})


def train_state(stage: str, phase_step: int, cursor: tuple[int, int], seed: int) -> dict:
    config = {
        "seed": seed,
        "batch_size": 4,
        "per_gpu_max_batch_size": 1,
        "optimizer": "adam",
        "freeze_gaze_vision": True,
        "freeze_gaze_connector": True,
        "max_train_steps": 2315 if stage == "stage1" else 17685,
        "lr_schedule": "linear_w_warmup" if stage == "stage1" else "constant",
        "lr": 1e-5 if stage == "stage1" else 3e-6,
    }
    if stage == "stage2":
        config["save_train_steps"] = [2685, 7685, 12685]
    return {
        "epoch": cursor[0],
        "iteration": cursor[1],
        "train_step": phase_step,
        "val_step": 1,
        "config": config,
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "supervised_algorithm_state": {
            "schema_version": 1,
            "contract": expected_resume_contract(stage, seed)["action_contract"],
            "teacher_seed": seed,
            "teacher_rng_state": torch.Generator().manual_seed(seed).get_state(),
        },
    }


def write_endpoint(
    phase_dir: Path,
    receipt_path: Path,
    *,
    stage: str,
    step: int,
    cursor: tuple[int, int],
) -> dict:
    phase_seed = SEED if stage == "stage1" else TRAINING_SEED
    write_model(phase_dir / "checkpoint_latest_gaze", f"{stage}-latest".encode())
    torch.save(
        train_state(stage, step, cursor, phase_seed),
        phase_dir / "checkpoint_latest_train.pt",
    )
    torch.save({}, phase_dir / "checkpoint_latest_task.pt")
    write_supervised_completion(
        phase_dir,
        train_step=step,
        contract=expected_resume_contract(stage, phase_seed),
    )
    completion = json.loads(
        (phase_dir / "checkpoint_latest_complete.json").read_text(encoding="utf-8")
    )
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "verified_utc": "2026-09-11T00:00:00Z",
        "stage": stage,
        "base_seed": SEED,
        "continuation_seed": TRAINING_SEED,
        "phase_seed": phase_seed,
        "train_step": step,
        "cumulative_train_step": step if stage == "stage1" else 20000,
        "next_input_cursor": list(cursor),
        "run_directory": str(phase_dir),
        "completion_marker_sha256": sha256_file(
            phase_dir / "checkpoint_latest_complete.json"
        ),
        "checkpoint_files_sha256": completion["files_sha256"],
        "resume_contract": expected_resume_contract(stage, phase_seed),
    }
    write_json(receipt_path, receipt)
    return receipt


def make_fixture(tmp_path: Path, *, recovery: bool = False) -> tuple[dict, Path, dict]:
    training_root = tmp_path / "training"
    seed_root = training_root / f"seed{SEED}_train{TRAINING_SEED}"
    seed_root.mkdir(parents=True)
    phase_root = seed_root / "recovery_attempt1" if recovery else seed_root
    stage1_dir = phase_root / "stage1"
    stage2_dir = phase_root / "stage2"
    stage1_dir.mkdir(parents=True)
    stage2_dir.mkdir(parents=True)
    stage1_receipt_path = seed_root / "stage1_completion_verification.json"
    stage2_receipt_path = seed_root / "stage2_completion_verification.json"
    stage1_receipt = write_endpoint(
        stage1_dir,
        stage1_receipt_path,
        stage="stage1",
        step=2315,
        cursor=(5, 0),
    )
    stage2_receipt = write_endpoint(
        stage2_dir,
        stage2_receipt_path,
        stage="stage2",
        step=17685,
        cursor=(38, 364),
    )
    for cumulative in (5000, 10000, 15000):
        _stage, step, cursor, _storage, _marker = FIXED[cumulative]
        periodic = stage2_dir / f"checkpoint_ep{cursor[0]}_iter{cursor[1]}"
        periodic.mkdir()
        write_model(periodic / "checkpoint_gaze", f"periodic-{step}".encode())
        torch.save(
            train_state("stage2", step, cursor, TRAINING_SEED),
            periodic / "checkpoint_train.pt",
        )
    execution = {
        "schema_version": 1,
        "status": "training_complete_pending_validation_and_final_sacct",
        "experiment_id": "supervised_k16_comparison",
        "run_id": RUN_ID,
        "base_seed": SEED,
        "continuation_seed": TRAINING_SEED,
        "source": {"execution_commit": EXECUTION_COMMIT, "dirty": False},
        "input_sha256": {
            "manifest": MANIFEST_SHA256,
            "cell_mass": CELL_MASS_SHA256,
        },
        "fixed_endpoint": {
            "cumulative_updates": 20000,
            "base_clip_presentations": 80000,
            "nominal_action_rows": 20480000,
        },
        "checkpoint_records": {
            "stage1": stage1_receipt,
            "stage2": stage2_receipt,
        },
        "resource_records": {
            "stage1_completion_verification": {
                "path": str(stage1_receipt_path),
                "sha256": sha256_file(stage1_receipt_path),
            },
            "stage2_completion_verification": {
                "path": str(stage2_receipt_path),
                "sha256": sha256_file(stage2_receipt_path),
            },
        },
    }
    write_json(seed_root / "execution_manifest.json", execution)
    if recovery:
        recovery_manifest = {
            "schema_version": 1,
            "status": "complete_same_seed_recovery",
            "base_seed": SEED,
            "training_seed": TRAINING_SEED,
            "immutable_source_commit": EXECUTION_COMMIT,
            "input_sha256": {
                "manifest": MANIFEST_SHA256,
                "cell_mass": CELL_MASS_SHA256,
            },
            "failed_job_ids": ["1702710_0"],
            "recovery_job_ids": ["1703000"],
            "resolved_phase_directories": {
                "stage1": str(stage1_dir),
                "stage2": str(stage2_dir),
            },
            "checkpoint_verification_receipts": {
                "stage1": {
                    "path": str(stage1_receipt_path),
                    "sha256": sha256_file(stage1_receipt_path),
                },
                "stage2": {
                    "path": str(stage2_receipt_path),
                    "sha256": sha256_file(stage2_receipt_path),
                },
            },
        }
        write_json(seed_root / "recovery_manifest.json", recovery_manifest)

    rl_dir = tmp_path / "rl" / "checkpoint_latest_gaze"
    write_model(rl_dir, b"known-frozen-rl")
    inventory = {
        "supervised_training_root": str(training_root),
        "supervised_seed_root_template": "seed{base_seed}_train{training_seed}",
        "supervised_checkpoints": [
            {
                "cumulative_update": cumulative,
                "phase": values[0],
                "phase_train_step": values[1],
                "cursor": list(values[2]),
                "storage": values[3],
                "completion_marker_required": values[4],
            }
            for cumulative, values in FIXED.items()
        ],
        "rl_endpoints": [
            {
                "base_seed": SEED,
                "continuation_seed": TRAINING_SEED,
                "path": str(rl_dir),
                "config_sha256": sha256_file(rl_dir / "config.json"),
                "model_safetensors_sha256": sha256_file(
                    rl_dir / "model.safetensors"
                ),
                "preprocessor_config_sha256": sha256_file(
                    rl_dir / "preprocessor_config.json"
                ),
            }
        ],
        "authoritative_sources": {
            "rl_preflight": {"path": "preflight", "sha256": "a" * 64},
            "rl_manifest": {"path": "manifest", "sha256": "b" * 64},
        },
    }
    return inventory, training_root, {
        "seed_root": seed_root,
        "stage1": stage1_dir,
        "stage2": stage2_dir,
        "stage1_receipt": stage1_receipt_path,
        "stage2_receipt": stage2_receipt_path,
        "rl": rl_dir,
    }


def test_valid_supervised_intermediate_and_endpoint_are_authoritatively_bound(
    tmp_path: Path,
) -> None:
    inventory, training_root, paths = make_fixture(tmp_path)
    intermediate, intermediate_provenance = verify_supervised_checkpoint(
        inventory,
        training_root=training_root,
        base_seed=SEED,
        cumulative_update=5000,
        supplied_run_dir=paths["stage2"],
    )
    endpoint, endpoint_provenance = verify_supervised_checkpoint(
        inventory,
        training_root=training_root,
        base_seed=SEED,
        cumulative_update=20000,
        supplied_run_dir=paths["stage2"],
    )
    assert intermediate.name == "checkpoint_gaze"
    assert intermediate_provenance["phase_train_step"] == 2685
    assert intermediate_provenance["completion_marker_sha256"] is None
    assert endpoint.name == "checkpoint_latest_gaze"
    assert endpoint_provenance["completion_marker_sha256"]


def test_tracked_inventory_pins_all_six_published_rl_endpoints() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/human_gaze/results/supervised_k16_comparison/checkpoint_provenance_inventory.json"
    inventory = load_checkpoint_inventory(
        path,
        expected_sha256="9c824436ef21eb0b00a538eb8b9b65ca82ddc1571b28615458c787050ff042a5",
        repository_root=root,
    )
    assert [row["base_seed"] for row in inventory["rl_endpoints"]] == list(
        range(440826, 440832)
    )
    assert inventory["rl_endpoints"][0]["model_safetensors_sha256"] == (
        "d008a458e62a15c69a236e4585a248b185343e82c2b6ece205d6638d4578b9ff"
    )


def test_wrong_seed_and_wrong_step_periodic_states_are_rejected(tmp_path: Path) -> None:
    inventory, training_root, paths = make_fixture(tmp_path)
    state_path = paths["stage2"] / "checkpoint_ep5_iter1480/checkpoint_train.pt"
    wrong_seed = train_state("stage2", 2685, (5, 1480), 540831)
    torch.save(wrong_seed, state_path)
    with pytest.raises(ValueError, match="config mismatch for seed"):
        verify_supervised_checkpoint(
            inventory,
            training_root=training_root,
            base_seed=SEED,
            cumulative_update=5000,
        )

    torch.save(train_state("stage2", 2684, (5, 1480), TRAINING_SEED), state_path)
    with pytest.raises(ValueError, match="wrong fixed step"):
        verify_supervised_checkpoint(
            inventory,
            training_root=training_root,
            base_seed=SEED,
            cumulative_update=5000,
        )


@pytest.mark.parametrize("mutation", ("missing_marker", "tampered_model"))
def test_endpoint_requires_valid_completion_marker_and_model(
    tmp_path: Path, mutation: str
) -> None:
    inventory, training_root, paths = make_fixture(tmp_path)
    if mutation == "missing_marker":
        (paths["stage2"] / "checkpoint_latest_complete.json").unlink()
    else:
        with (paths["stage2"] / "checkpoint_latest_gaze/model.safetensors").open(
            "ab"
        ) as handle:
            handle.write(b"tampered")
    with pytest.raises(ValueError, match="completion marker|hash mismatch"):
        verify_supervised_checkpoint(
            inventory,
            training_root=training_root,
            base_seed=SEED,
            cumulative_update=20000,
        )


def test_method_and_direct_checkpoint_confusion_are_rejected(tmp_path: Path) -> None:
    inventory, training_root, paths = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="Direct supervised"):
        verify_checkpoint_for_method(
            "supervised",
            inventory,
            training_root=training_root,
            base_seed=SEED,
            cumulative_update=20000,
            supplied_checkpoint=paths["stage2"] / "checkpoint_latest_gaze",
            supplied_run_dir=paths["stage2"],
        )
    with pytest.raises(ValueError, match="canonical frozen endpoint"):
        verify_checkpoint_for_method(
            "rl",
            inventory,
            base_seed=SEED,
            cumulative_update=20000,
            supplied_checkpoint=paths["stage2"] / "checkpoint_latest_gaze",
        )
    with pytest.raises(ValueError, match="cannot use a supervised"):
        verify_checkpoint_for_method(
            "rl",
            inventory,
            base_seed=SEED,
            cumulative_update=20000,
            supplied_checkpoint=paths["rl"],
            supplied_run_dir=paths["stage2"],
        )


def test_valid_known_rl_and_tampered_rl_model(tmp_path: Path) -> None:
    inventory, _training_root, paths = make_fixture(tmp_path)
    checkpoint, provenance = verify_rl_checkpoint(
        inventory,
        base_seed=SEED,
        cumulative_update=20000,
        supplied_checkpoint=paths["rl"],
    )
    assert checkpoint == paths["rl"].resolve()
    assert provenance["authority"] == "published_r2e_frozen_rl_inventory"
    with (paths["rl"] / "model.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_rl_checkpoint(
            inventory,
            base_seed=SEED,
            cumulative_update=20000,
        )


def test_valid_recovery_and_self_attested_empty_receipt_rejected(tmp_path: Path) -> None:
    inventory, training_root, paths = make_fixture(tmp_path, recovery=True)
    checkpoint, provenance = verify_supervised_checkpoint(
        inventory,
        training_root=training_root,
        base_seed=SEED,
        cumulative_update=20000,
        supplied_run_dir=paths["stage2"],
    )
    assert checkpoint == (paths["stage2"] / "checkpoint_latest_gaze").resolve()
    assert provenance["recovery_manifest"] is not None
    timing = supervised_training_wall(inventory, training_root, SEED)
    assert timing["status"] == "withheld_recovery_attempt_timing_not_reconstructable"
    assert timing["points"] == {}

    write_json(paths["stage2_receipt"], {})
    recovery_path = paths["seed_root"] / "recovery_manifest.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery["checkpoint_verification_receipts"]["stage2"]["sha256"] = sha256_file(
        paths["stage2_receipt"]
    )
    write_json(recovery_path, recovery)
    execution_path = paths["seed_root"] / "execution_manifest.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["checkpoint_records"]["stage2"] = {}
    write_json(execution_path, execution)
    with pytest.raises(ValueError, match="receipt mismatch"):
        verify_supervised_checkpoint(
            inventory,
            training_root=training_root,
            base_seed=SEED,
            cumulative_update=20000,
        )
