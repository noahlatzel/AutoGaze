"""A supervised resume must reject incomplete or mixed checkpoint bundles."""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from autogaze.algorithms import HumanGazeNTP
from autogaze.supervised_checkpoint import (
    COMPLETION_MARKER,
    supervised_resume_contract,
    verify_supervised_completion,
)
from autogaze.trainer import Trainer


class ReceiptModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25]))

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "weights.pt")
        (directory / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")

    @classmethod
    def from_pretrained(cls, directory):
        model = cls()
        model.load_state_dict(torch.load(Path(directory) / "weights.pt", weights_only=True))
        return model


def receipt_trainer(directory):
    trainer = Trainer.__new__(Trainer)
    trainer.gaze_model = ReceiptModel()
    trainer.task = torch.nn.Identity()
    trainer.algorithm = HumanGazeNTP(clip_len=1, actions_per_frame=5, fine_action_offset=1, exact_budget=2, teacher_seed=5)
    trainer.human_supervision = True
    trainer.config = {
        "seed": 5, "batch_size": 2, "per_gpu_max_batch_size": 1,
        "optimizer": "adam", "lr": 0.01, "lr_schedule": "constant",
    }
    trainer.train_loader = torch.utils.data.DataLoader(torch.arange(8), batch_size=1)
    trainer.train_loader.sampler.seed = 5
    trainer.grad_acc_steps = 2
    trainer.n_epochs = 3
    trainer.optimizer = torch.optim.Adam(trainer.gaze_model.parameters(), lr=0.01)
    trainer.scheduler = torch.optim.lr_scheduler.ConstantLR(trainer.optimizer, factor=1.0)
    trainer.train_step = trainer.val_step = 0
    trainer.save_dir = str(directory)
    trainer.gaze_processor = None
    trainer.max_periodic_checkpoints = None
    return trainer


def test_receipt_covers_only_sorted_relative_checkpoint_files(tmp_path):
    trainer = receipt_trainer(tmp_path)
    (tmp_path / "training_metrics.jsonl").write_text("unrelated log\n", encoding="utf-8")
    (tmp_path / "unrelated_artifact.bin").write_bytes(b"not a checkpoint")
    trainer.save_checkpoint(0, 0)
    receipt = verify_supervised_completion(tmp_path, expected_contract=supervised_resume_contract(trainer))
    assert list(receipt["files_sha256"]) == [
        "checkpoint_latest_gaze/config.json", "checkpoint_latest_gaze/weights.pt",
        "checkpoint_latest_task.pt", "checkpoint_latest_train.pt",
    ]
    assert receipt["train_step"] == 0
    assert receipt["resume_contract"]["sampler_seed"] == 5
    assert not (tmp_path / "checkpoint_ep0_iter0" / COMPLETION_MARKER).exists()
    assert not list(tmp_path.glob(".supervised-complete-*.tmp"))


@pytest.mark.parametrize("mutation", [
    "model_bytes", "task_bytes", "train_bytes", "missing_marker", "unexpected_model_file",
    "missing_model_config", "trainer_seed", "sampler_seed", "teacher_seed", "batch_size",
    "gradient_accumulation", "loader_length", "phase_epochs",
])
def test_invalid_bundle_or_resume_contract_fails_before_torch_load_or_mutation(tmp_path, monkeypatch, mutation):
    trainer = receipt_trainer(tmp_path)
    trainer.save_checkpoint(0, 0)
    if mutation.endswith("_bytes"):
        paths = {
            "model_bytes": tmp_path / "checkpoint_latest_gaze" / "weights.pt",
            "task_bytes": tmp_path / "checkpoint_latest_task.pt",
            "train_bytes": tmp_path / "checkpoint_latest_train.pt",
        }
        with paths[mutation].open("ab") as handle:
            handle.write(b"corrupted-or-mixed-state")
    elif mutation == "missing_marker":
        (tmp_path / COMPLETION_MARKER).unlink()
    elif mutation == "unexpected_model_file":
        (tmp_path / "checkpoint_latest_gaze" / "extra.bin").write_bytes(b"unexpected")
    elif mutation == "missing_model_config":
        (tmp_path / "checkpoint_latest_gaze" / "config.json").unlink()
    elif mutation == "trainer_seed":
        trainer.config["seed"] = 6
    elif mutation == "sampler_seed":
        trainer.train_loader.sampler.seed = 6
    elif mutation == "teacher_seed":
        trainer.algorithm.teacher_seed = 6
    elif mutation == "batch_size":
        trainer.config["batch_size"] = 4
    elif mutation == "gradient_accumulation":
        trainer.grad_acc_steps = 4
    elif mutation == "loader_length":
        trainer.train_loader = torch.utils.data.DataLoader(torch.arange(12), batch_size=1)
        trainer.train_loader.sampler.seed = 5
    elif mutation == "phase_epochs":
        trainer.n_epochs = 4

    with torch.no_grad():
        trainer.gaze_model.weight.fill_(99)
    rng_before = trainer.algorithm.state_dict()["teacher_rng_state"]

    def forbidden_load(*args, **kwargs):
        pytest.fail("Checkpoint deserialization occurred before receipt verification")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="Supervised checkpoint"):
        trainer.load_checkpoint(resume=True)
    assert trainer.gaze_model.weight.item() == 99
    assert torch.equal(rng_before, trainer.algorithm.state_dict()["teacher_rng_state"])


def test_interruption_after_new_model_write_cannot_resume_with_old_optimizer_receipt(tmp_path, monkeypatch):
    trainer = receipt_trainer(tmp_path)
    trainer.save_checkpoint(0, 0)
    old_receipt = (tmp_path / COMPLETION_MARKER).read_bytes()
    original_save = trainer.gaze_model.save_pretrained

    def interrupted_save(directory):
        original_save(directory)
        raise RuntimeError("simulated preemption before optimizer and teacher state write")

    with torch.no_grad():
        trainer.gaze_model.weight.add_(1)
    trainer.train_step = 1
    monkeypatch.setattr(trainer.gaze_model, "save_pretrained", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        trainer.save_checkpoint(0, 2)
    assert (tmp_path / COMPLETION_MARKER).read_bytes() == old_receipt
    with pytest.raises(ValueError, match="hash mismatch"):
        trainer.load_checkpoint(resume=True)


def test_marker_step_disagreement_is_rejected_before_model_mutation(tmp_path):
    trainer = receipt_trainer(tmp_path)
    trainer.save_checkpoint(0, 0)
    marker = tmp_path / COMPLETION_MARKER
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    receipt["train_step"] = 99
    marker.write_text(json.dumps(receipt), encoding="utf-8")
    with torch.no_grad():
        trainer.gaze_model.weight.fill_(123)
    with pytest.raises(ValueError, match="training step disagrees"):
        trainer.load_checkpoint(resume=True)
    assert trainer.gaze_model.weight.item() == 123


def test_real_stage_two_hydra_config_round_trips_with_weights_only_checkpoint_loading(tmp_path):
    config_dir = str(Path(__file__).resolve().parents[1] / "autogaze" / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="av_gaze_stavis_supervised_k16_stage2",
            overrides=["trainer.gaze_weights=fixture-stage-one-checkpoint"],
        )
    trainer = receipt_trainer(tmp_path)
    trainer.config = {key: value for key, value in config.trainer.items() if key != "_target_"}
    trainer.n_epochs = config.trainer.n_epochs
    trainer.grad_acc_steps = config.trainer.batch_size
    trainer.train_loader.sampler.seed = config.trainer.seed
    trainer.algorithm = HumanGazeNTP(
        clip_len=1, actions_per_frame=5, fine_action_offset=1,
        exact_budget=2, teacher_seed=config.trainer.seed,
    )
    assert OmegaConf.is_config(trainer.config["save_train_steps"])
    trainer.save_checkpoint(0, 0)
    # This failed with ListConfig in torch 2.9's default weights-only loader.
    saved = torch.load(tmp_path / "checkpoint_latest_train.pt", weights_only=True)
    assert type(saved["config"]["save_train_steps"]) is list
    assert saved["config"]["save_train_steps"] == [2685, 7685, 12685]
    assert saved["config"]["seed"] == 540826
    assert OmegaConf.is_config(trainer.config["save_train_steps"])
    with torch.no_grad():
        trainer.gaze_model.weight.fill_(42)
    trainer.load_checkpoint(resume=True)
    assert trainer.gaze_model.weight.item() == 0.25
