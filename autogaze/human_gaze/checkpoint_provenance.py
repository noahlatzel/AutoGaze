"""Authoritative checkpoint binding for the supervised K16 comparison.

Export labels are never treated as checkpoint provenance.  Supervised models
are bound to the canonical execution manifest, phase receipts, completion
markers and saved train state.  The genuine legacy RL endpoints are bound to
their already-published model/config/processor hashes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from autogaze.human_gaze.supervised_analysis import sha256_file, sha256_tree
from autogaze.supervised_checkpoint import verify_supervised_completion


BASE_SEEDS = tuple(range(440826, 440832))
EXECUTION_COMMIT = "5a31685d56ec727b9a46db60598a0693fae7e20e"
RUN_ID = "20260910-2128_supervised-k16-comparison_5a31685"
MANIFEST_SHA256 = "0ef9fa17881517ea69d9edbaf57f8571c5c58d1da517370746fbe264fe33ff10"
CELL_MASS_SHA256 = "fed5a6552ab94eeaa02a807959d4c3cda6de9dbadeadc35ee248e46ebc16dfb8"
MODEL_FILENAMES = ("config.json", "model.safetensors", "preprocessor_config.json")


def _load_json(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Required provenance JSON is missing or symbolic: {candidate}")
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError(f"Required provenance JSON is unreadable: {candidate}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Required provenance JSON is not an object: {candidate}")
    return value


def _resolve_record_path(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def load_checkpoint_inventory(
    path: Path,
    *,
    expected_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    """Load the tracked inventory and validate its published source artifacts."""
    inventory_path = Path(path).resolve(strict=True)
    if sha256_file(inventory_path) != expected_sha256:
        raise ValueError("Checkpoint provenance inventory hash drift")
    inventory = _load_json(inventory_path)
    if (
        inventory.get("schema_version") != 1
        or inventory.get("status") != "frozen_authoritative_checkpoint_inventory"
        or inventory.get("experiment_id") != "supervised_k16_comparison"
    ):
        raise ValueError("Unsupported checkpoint provenance inventory")
    authority = inventory.get("authoritative_sources", {})
    if (
        authority.get("supervised_execution_commit") != EXECUTION_COMMIT
        or authority.get("supervised_run_id") != RUN_ID
        or inventory.get("data_sha256")
        != {"manifest": MANIFEST_SHA256, "cell_mass": CELL_MASS_SHA256}
    ):
        raise ValueError("Checkpoint inventory changes execution or data identity")
    for source in ("rl_preflight", "rl_manifest"):
        record = authority.get(source, {})
        source_path = _resolve_record_path(record.get("path", ""), repository_root)
        if sha256_file(source_path) != record.get("sha256"):
            raise ValueError(f"Published {source} provenance hash drift")
    sl_records = inventory.get("supervised_checkpoints")
    expected_sl = {
        2315: ("stage1", 2315, (5, 0), "latest", True),
        5000: ("stage2", 2685, (5, 1480), "periodic", False),
        10000: ("stage2", 7685, (16, 1108), "periodic", False),
        15000: ("stage2", 12685, (27, 736), "periodic", False),
        20000: ("stage2", 17685, (38, 364), "latest", True),
    }
    if not isinstance(sl_records, list) or len(sl_records) != len(expected_sl):
        raise ValueError("Checkpoint inventory has the wrong supervised checkpoint count")
    for record in sl_records:
        step = record.get("cumulative_update")
        actual = (
            record.get("phase"),
            record.get("phase_train_step"),
            tuple(record.get("cursor", ())),
            record.get("storage"),
            record.get("completion_marker_required"),
        )
        if step not in expected_sl or actual != expected_sl[step]:
            raise ValueError("Checkpoint inventory changes a supervised checkpoint contract")
    rl_records = inventory.get("rl_endpoints")
    if not isinstance(rl_records, list) or [row.get("base_seed") for row in rl_records] != list(BASE_SEEDS):
        raise ValueError("Checkpoint inventory has the wrong RL endpoint matrix")
    for record in rl_records:
        seed = record["base_seed"]
        if record.get("continuation_seed") != seed + 100000:
            raise ValueError("Checkpoint inventory has a mismatched RL continuation seed")
        for key in (
            "model_safetensors_sha256",
            "config_sha256",
            "preprocessor_config_sha256",
        ):
            value = record.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Checkpoint inventory has an invalid RL {key}")
    return inventory


def expected_resume_contract(stage: str, phase_seed: int) -> dict[str, Any]:
    if stage == "stage1":
        n_epochs, schedule, learning_rate = 5, "linear_w_warmup", 1e-5
    elif stage == "stage2":
        n_epochs, schedule, learning_rate = 40, "constant", 3e-6
    else:
        raise ValueError(f"Unknown supervised phase: {stage}")
    return {
        "trainer_seed": phase_seed,
        "sampler_seed": phase_seed,
        "teacher_seed": phase_seed,
        "configured_batch_size": 4,
        "configured_per_gpu_max_batch_size": 1,
        "loader_batch_size": 1,
        "loader_batches": 1852,
        "dataset_items": 1854,
        "gradient_accumulation_steps": 4,
        "n_epochs": n_epochs,
        "optimizer": "adam",
        "lr_schedule": schedule,
        "learning_rate": learning_rate,
        "sampler_type": "autogaze.datasets.av_gaze_stavis.BalancedSourceSampler",
        "action_contract": {
            "clip_len": 16,
            "actions_per_frame": 265,
            "fine_action_offset": 69,
            "exact_budget": 16,
        },
    }


def _model_identity(checkpoint: Path) -> dict[str, Any]:
    root = Path(checkpoint).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Checkpoint model directory is missing or symbolic")
    observed = sorted(path.name for path in root.iterdir() if path.is_file())
    if observed != sorted(MODEL_FILENAMES) or any(path.is_symlink() for path in root.iterdir()):
        raise ValueError("Checkpoint model directory has unexpected, missing or symbolic entries")
    config = _load_json(root / "config.json")
    decoder = config.get("gaze_model_config", {}).get("gaze_decoder_config", {})
    if (
        config.get("gaze_model_config", {}).get("num_vision_tokens_each_frame") != 265
        or decoder.get("vocab_size") != 266
        or decoder.get("eos_token_id") != 265
        or decoder.get("num_multi_token_pred") != 10
    ):
        raise ValueError("Checkpoint model config violates the fixed action geometry")
    files = {name: sha256_file(root / name) for name in MODEL_FILENAMES}
    return {
        "path": str(root),
        "tree_sha256": sha256_tree(root),
        "files_sha256": files,
    }


def _load_and_validate_train_state(
    path: Path,
    *,
    stage: str,
    phase_seed: int,
    phase_step: int,
    cursor: tuple[int, int],
) -> tuple[dict[str, Any], str]:
    state_path = Path(path).resolve(strict=True)
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("Supervised checkpoint train state is missing or symbolic")
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("Supervised checkpoint train state is unreadable") from error
    if not isinstance(state, Mapping):
        raise ValueError("Supervised checkpoint train state is not a mapping")
    if state.get("train_step") != phase_step or (state.get("epoch"), state.get("iteration")) != cursor:
        raise ValueError("Supervised checkpoint has the wrong fixed step or sampler cursor")
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Supervised checkpoint lacks its trainer config")
    expected = {
        "seed": phase_seed,
        "batch_size": 4,
        "per_gpu_max_batch_size": 1,
        "optimizer": "adam",
        "freeze_gaze_vision": True,
        "freeze_gaze_connector": True,
        "max_train_steps": 2315 if stage == "stage1" else 17685,
        "lr_schedule": "linear_w_warmup" if stage == "stage1" else "constant",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Supervised checkpoint trainer config mismatch for {key}")
    learning_rate = config.get("lr")
    expected_lr = 1e-5 if stage == "stage1" else 3e-6
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isclose(float(learning_rate), expected_lr, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("Supervised checkpoint trainer config has the wrong learning rate")
    if stage == "stage2" and list(config.get("save_train_steps", ())) != [2685, 7685, 12685]:
        raise ValueError("Supervised stage-two checkpoint has the wrong fixed save steps")
    algorithm = state.get("supervised_algorithm_state")
    if not isinstance(algorithm, Mapping):
        raise ValueError("Supervised checkpoint lacks teacher state")
    if (
        algorithm.get("schema_version") != 1
        or algorithm.get("teacher_seed") != phase_seed
        or algorithm.get("contract") != expected_resume_contract(stage, phase_seed)["action_contract"]
    ):
        raise ValueError("Supervised checkpoint teacher seed or action contract mismatch")
    rng = algorithm.get("teacher_rng_state")
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError("Supervised checkpoint has an invalid teacher RNG state")
    return dict(state), sha256_file(state_path)


def _validate_phase_receipt(
    path: Path,
    *,
    stage: str,
    base_seed: int,
    phase_dir: Path,
    expected_step: int,
    expected_cursor: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = Path(path).resolve(strict=True)
    receipt = _load_json(receipt_path)
    phase_seed = base_seed if stage == "stage1" else base_seed + 100000
    cumulative = expected_step if stage == "stage1" else 20000
    expected_fields = {
        "schema_version": 1,
        "status": "pass",
        "stage": stage,
        "base_seed": base_seed,
        "continuation_seed": base_seed + 100000,
        "phase_seed": phase_seed,
        "train_step": expected_step,
        "cumulative_train_step": cumulative,
        "next_input_cursor": list(expected_cursor),
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"Supervised phase receipt mismatch for {stage}.{key}")
    if Path(receipt.get("run_directory", "")).resolve(strict=True) != phase_dir.resolve(strict=True):
        raise ValueError("Supervised phase receipt points to a different run directory")
    contract = expected_resume_contract(stage, phase_seed)
    if receipt.get("resume_contract") != contract:
        raise ValueError("Supervised phase receipt has the wrong resume contract")
    completion = verify_supervised_completion(phase_dir, expected_contract=contract)
    if completion.get("train_step") != expected_step:
        raise ValueError("Supervised completion marker has the wrong phase step")
    marker = phase_dir / "checkpoint_latest_complete.json"
    if receipt.get("completion_marker_sha256") != sha256_file(marker):
        raise ValueError("Supervised phase receipt does not bind the completion marker")
    if receipt.get("checkpoint_files_sha256") != completion.get("files_sha256"):
        raise ValueError("Supervised phase receipt does not bind the completed checkpoint files")
    _load_and_validate_train_state(
        phase_dir / "checkpoint_latest_train.pt",
        stage=stage,
        phase_seed=phase_seed,
        phase_step=expected_step,
        cursor=expected_cursor,
    )
    model = _model_identity(phase_dir / "checkpoint_latest_gaze")
    return receipt, {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "completion_marker_sha256": sha256_file(marker),
        "model": model,
    }


def _canonical_supervised_execution(
    inventory: Mapping[str, Any],
    *,
    training_root: Path,
    base_seed: int,
) -> dict[str, Any]:
    if base_seed not in BASE_SEEDS:
        raise ValueError("Base seed is outside the frozen supervised matrix")
    training_seed = base_seed + 100000
    seed_root = (
        Path(training_root)
        / inventory["supervised_seed_root_template"].format(
            base_seed=base_seed, training_seed=training_seed
        )
    ).resolve(strict=True)
    execution_path = seed_root / "execution_manifest.json"
    execution = _load_json(execution_path)
    if (
        execution.get("schema_version") != 1
        or not str(execution.get("status", "")).startswith("training_complete")
        or execution.get("experiment_id") != "supervised_k16_comparison"
        or execution.get("run_id") != RUN_ID
        or execution.get("base_seed") != base_seed
        or execution.get("continuation_seed") != training_seed
    ):
        raise ValueError("Canonical supervised execution manifest identity mismatch")
    source = execution.get("source", {})
    if source.get("execution_commit") != EXECUTION_COMMIT or source.get("dirty") is not False:
        raise ValueError("Canonical supervised execution source identity mismatch")
    inputs = execution.get("input_sha256", {})
    if inputs.get("manifest") != MANIFEST_SHA256 or inputs.get("cell_mass") != CELL_MASS_SHA256:
        raise ValueError("Canonical supervised execution data identity mismatch")
    fixed = execution.get("fixed_endpoint", {})
    if (
        fixed.get("cumulative_updates") != 20000
        or fixed.get("base_clip_presentations") != 80000
        or fixed.get("nominal_action_rows") != 20480000
    ):
        raise ValueError("Canonical supervised execution exposure mismatch")

    phase_dirs = {"stage1": seed_root / "stage1", "stage2": seed_root / "stage2"}
    receipt_paths: dict[str, Path] = {}
    recovery_path = seed_root / "recovery_manifest.json"
    recovery = None
    if recovery_path.is_file():
        recovery = _load_json(recovery_path)
        if (
            recovery.get("schema_version") != 1
            or recovery.get("status") != "complete_same_seed_recovery"
            or recovery.get("base_seed") != base_seed
            or recovery.get("training_seed") != training_seed
            or recovery.get("immutable_source_commit") != EXECUTION_COMMIT
            or recovery.get("input_sha256")
            != {"manifest": MANIFEST_SHA256, "cell_mass": CELL_MASS_SHA256}
        ):
            raise ValueError("Same-seed recovery manifest identity mismatch")
        for key in ("failed_job_ids", "recovery_job_ids"):
            if not isinstance(recovery.get(key), list) or not recovery[key]:
                raise ValueError(f"Same-seed recovery manifest lacks {key}")
        for stage in phase_dirs:
            phase_dirs[stage] = _resolve_record_path(
                recovery.get("resolved_phase_directories", {}).get(stage, ""), seed_root
            )
            receipt_record = recovery.get("checkpoint_verification_receipts", {}).get(stage, {})
            receipt_path = _resolve_record_path(receipt_record.get("path", ""), seed_root)
            if sha256_file(receipt_path) != receipt_record.get("sha256"):
                raise ValueError(f"Same-seed recovery {stage} receipt hash mismatch")
            receipt_paths[stage] = receipt_path
    else:
        for stage in phase_dirs:
            phase_dirs[stage] = phase_dirs[stage].resolve(strict=True)
            name = f"{stage}_completion_verification"
            resource = execution.get("resource_records", {}).get(name, {})
            receipt_path = _resolve_record_path(resource.get("path", ""), seed_root)
            if sha256_file(receipt_path) != resource.get("sha256"):
                raise ValueError(f"Canonical supervised {stage} receipt hash mismatch")
            receipt_paths[stage] = receipt_path

    endpoints = {}
    endpoint_contracts = {
        "stage1": (2315, (5, 0)),
        "stage2": (17685, (38, 364)),
    }
    for stage, (step, cursor) in endpoint_contracts.items():
        embedded = execution.get("checkpoint_records", {}).get(stage)
        actual = _load_json(receipt_paths[stage])
        if embedded != actual:
            raise ValueError(f"Execution manifest does not bind the actual {stage} receipt")
        receipt, identity = _validate_phase_receipt(
            receipt_paths[stage],
            stage=stage,
            base_seed=base_seed,
            phase_dir=phase_dirs[stage],
            expected_step=step,
            expected_cursor=cursor,
        )
        endpoints[stage] = {"receipt": receipt, **identity}
    return {
        "seed_root": seed_root,
        "execution_manifest_path": execution_path,
        "execution_manifest_sha256": sha256_file(execution_path),
        "phase_dirs": phase_dirs,
        "endpoints": endpoints,
        "recovery_manifest_path": recovery_path if recovery is not None else None,
        "recovery_manifest_sha256": sha256_file(recovery_path) if recovery is not None else None,
    }


def verify_supervised_checkpoint(
    inventory: Mapping[str, Any],
    *,
    training_root: Path,
    base_seed: int,
    cumulative_update: int,
    supplied_run_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    canonical_root = Path(inventory["supervised_training_root"]).resolve(strict=True)
    if Path(training_root).resolve(strict=True) != canonical_root:
        raise ValueError("Supervised training root is not the canonical admitted execution")
    records = {
        int(record["cumulative_update"]): record
        for record in inventory["supervised_checkpoints"]
    }
    if cumulative_update not in records:
        raise ValueError("Supervised checkpoint is not a fixed convergence point")
    record = records[cumulative_update]
    execution = _canonical_supervised_execution(
        inventory, training_root=canonical_root, base_seed=base_seed
    )
    phase = record["phase"]
    phase_dir = execution["phase_dirs"][phase]
    if supplied_run_dir is not None and supplied_run_dir.resolve(strict=True) != phase_dir:
        raise ValueError("Supplied supervised run directory is not the canonical seed phase")
    phase_seed = base_seed if phase == "stage1" else base_seed + 100000
    phase_step = int(record["phase_train_step"])
    cursor = tuple(record["cursor"])
    if record["storage"] == "latest":
        checkpoint = phase_dir / "checkpoint_latest_gaze"
        train_state = phase_dir / "checkpoint_latest_train.pt"
        marker_sha256 = execution["endpoints"][phase]["completion_marker_sha256"]
    else:
        periodic = phase_dir / f"checkpoint_ep{cursor[0]}_iter{cursor[1]}"
        if periodic.is_symlink() or not periodic.is_dir():
            raise ValueError("Fixed supervised periodic checkpoint directory is missing")
        checkpoint = periodic / "checkpoint_gaze"
        train_state = periodic / "checkpoint_train.pt"
        marker_sha256 = None
    _, train_state_sha256 = _load_and_validate_train_state(
        train_state,
        stage=phase,
        phase_seed=phase_seed,
        phase_step=phase_step,
        cursor=cursor,
    )
    model = _model_identity(checkpoint)
    provenance = {
        "schema_version": 1,
        "authority": "canonical_supervised_execution",
        "method": "supervised",
        "base_seed": base_seed,
        "training_seed": base_seed + 100000,
        "cumulative_update": cumulative_update,
        "phase": phase,
        "phase_train_step": phase_step,
        "cursor": list(cursor),
        "storage": record["storage"],
        "checkpoint": model,
        "checkpoint_train_state": str(train_state.resolve(strict=True)),
        "checkpoint_train_state_sha256": train_state_sha256,
        "completion_marker_sha256": marker_sha256,
        "execution_manifest": str(execution["execution_manifest_path"]),
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "recovery_manifest": (
            str(execution["recovery_manifest_path"])
            if execution["recovery_manifest_path"] is not None
            else None
        ),
        "recovery_manifest_sha256": execution["recovery_manifest_sha256"],
    }
    return checkpoint.resolve(strict=True), provenance


def verify_rl_checkpoint(
    inventory: Mapping[str, Any],
    *,
    base_seed: int,
    cumulative_update: int,
    supplied_checkpoint: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if cumulative_update != 20000:
        raise ValueError("Only the frozen RL K16 20k endpoints are authoritative")
    matches = [row for row in inventory["rl_endpoints"] if row["base_seed"] == base_seed]
    if len(matches) != 1:
        raise ValueError("RL base seed is outside the frozen endpoint inventory")
    record = matches[0]
    checkpoint = Path(record["path"]).resolve(strict=True)
    if supplied_checkpoint is not None and supplied_checkpoint.resolve(strict=True) != checkpoint:
        raise ValueError("Supplied RL checkpoint is not the canonical frozen endpoint")
    model = _model_identity(checkpoint)
    expected = {
        "config.json": record["config_sha256"],
        "model.safetensors": record["model_safetensors_sha256"],
        "preprocessor_config.json": record["preprocessor_config_sha256"],
    }
    if model["files_sha256"] != expected:
        raise ValueError("Frozen RL endpoint model/config/processor hash mismatch")
    provenance = {
        "schema_version": 1,
        "authority": "published_r2e_frozen_rl_inventory",
        "method": "rl",
        "base_seed": base_seed,
        "training_seed": base_seed + 100000,
        "cumulative_update": 20000,
        "checkpoint": model,
        "published_preflight": inventory["authoritative_sources"]["rl_preflight"],
        "published_manifest": inventory["authoritative_sources"]["rl_manifest"],
    }
    return checkpoint, provenance


def verify_checkpoint_for_method(
    method: str,
    inventory: Mapping[str, Any],
    *,
    base_seed: int,
    cumulative_update: int,
    training_root: Path | None = None,
    supplied_checkpoint: Path | None = None,
    supplied_run_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Dispatch without allowing a path from one method to impersonate another."""
    if method == "supervised":
        if supplied_checkpoint is not None:
            raise ValueError("Direct supervised checkpoints are forbidden")
        if training_root is None or supplied_run_dir is None:
            raise ValueError("Supervised export requires its canonical training root and phase")
        return verify_supervised_checkpoint(
            inventory,
            training_root=training_root,
            base_seed=base_seed,
            cumulative_update=cumulative_update,
            supplied_run_dir=supplied_run_dir,
        )
    if method == "rl":
        if supplied_run_dir is not None:
            raise ValueError("RL export cannot use a supervised run directory")
        if supplied_checkpoint is None:
            raise ValueError("RL export requires its canonical frozen checkpoint")
        return verify_rl_checkpoint(
            inventory,
            base_seed=base_seed,
            cumulative_update=cumulative_update,
            supplied_checkpoint=supplied_checkpoint,
        )
    raise ValueError(f"Unsupported checkpoint method: {method}")
