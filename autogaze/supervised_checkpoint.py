"""Completion receipts for the latest supervised checkpoint bundle only.

The receipt protects model, optimizer and teacher-state consistency after an
interrupted write. Dataset and execution-source identities remain the external
immutable run manifest's responsibility. Periodic snapshots are not automatic
resume targets and do not receive a receipt from this helper.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from omegaconf import OmegaConf


COMPLETION_MARKER = "checkpoint_latest_complete.json"


def plain_supervised_checkpoint_config(config):
    """Keep Hydra containers out of checkpoints loaded with weights_only=True."""
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True, enum_to_str=True)


def supervised_resume_contract(trainer):
    """Capture the phase and sampler settings needed to reproduce next inputs."""
    loader = trainer.train_loader
    sampler = loader.sampler
    config = trainer.config
    integers = {
        "trainer_seed": config.get("seed"),
        "sampler_seed": getattr(sampler, "seed", None),
        "teacher_seed": trainer.algorithm.teacher_seed,
        "configured_batch_size": config.get("batch_size"),
        "configured_per_gpu_max_batch_size": config.get("per_gpu_max_batch_size"),
        "loader_batch_size": loader.batch_size,
        "loader_batches": len(loader),
        "dataset_items": len(loader.dataset),
        "gradient_accumulation_steps": trainer.grad_acc_steps,
        "n_epochs": trainer.n_epochs,
    }
    for key, value in integers.items():
        lower = 0 if key.endswith("seed") else 1
        if type(value) is not int or value < lower:
            raise ValueError(f"Supervised checkpoint contract requires a valid {key}")
    strings = {name: config.get(name) for name in ("optimizer", "lr_schedule")}
    if any(not isinstance(value, str) or not value for value in strings.values()):
        raise ValueError("Supervised checkpoint contract requires optimizer and lr_schedule")
    learning_rate = config.get("lr")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Supervised checkpoint contract requires a positive finite learning rate")
    return {
        **integers,
        **strings,
        "learning_rate": float(learning_rate),
        "sampler_type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
        "action_contract": trainer.algorithm._contract(),
    }


def _checkpoint_files(directory):
    root = Path(directory)
    gaze = root / "checkpoint_latest_gaze"
    if gaze.is_symlink() or not gaze.is_dir():
        raise ValueError("Supervised checkpoint model directory is missing or symbolic")
    files = []
    for path in gaze.rglob("*"):
        if path.is_symlink():
            raise ValueError("Supervised checkpoint files must not be symbolic links")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError("Supervised checkpoint contains a non-regular entry")
    if not files:
        raise ValueError("Supervised checkpoint model directory is empty")
    for name in ("checkpoint_latest_task.pt", "checkpoint_latest_train.pt"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Supervised checkpoint is missing regular file {name}")
        files.append(path)
    return {path.relative_to(root).as_posix(): path for path in sorted(files)}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_supervised_completion(directory, *, train_step, contract):
    """Publish an atomic receipt only after all checkpoint writes have finished."""
    if type(train_step) is not int or train_step < 0:
        raise ValueError("Supervised completion marker requires a nonnegative train_step")
    root = Path(directory)
    hashes = {name: _sha256(path) for name, path in _checkpoint_files(root).items()}
    receipt = {
        "schema_version": 1,
        "train_step": train_step,
        "resume_contract": contract,
        "files_sha256": hashes,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix=".supervised-complete-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / COMPLETION_MARKER)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def verify_supervised_completion(directory, *, expected_contract):
    """Verify the complete bundle before deserialization or model mutation."""
    marker = Path(directory) / COMPLETION_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Supervised checkpoint completion marker is missing; refusing an unverified resume")
    try:
        with marker.open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("Supervised checkpoint completion marker is unreadable") from error
    keys = {"schema_version", "train_step", "resume_contract", "files_sha256"}
    if not isinstance(receipt, dict) or set(receipt) != keys or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("Supervised checkpoint completion marker has an unsupported schema")
    if type(receipt["train_step"]) is not int or receipt["train_step"] < 0:
        raise ValueError("Supervised checkpoint completion marker has an invalid train_step")
    if receipt["resume_contract"] != expected_contract:
        raise ValueError("Supervised checkpoint resume contract mismatch: phase, seed, sampler or batch settings changed")
    files = _checkpoint_files(directory)
    hashes = receipt["files_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(files):
        raise ValueError("Supervised checkpoint completion marker has unexpected or missing checkpoint files")
    for name, path in files.items():
        expected = hashes[name]
        if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("Supervised checkpoint completion marker contains an invalid SHA256")
        if _sha256(path) != expected:
            raise ValueError(f"Supervised checkpoint hash mismatch for {name}; refusing an incomplete or mixed bundle")
    return receipt
