"""Exercise the real Trainer constructor, not a hand-filled __new__ config."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import autogaze.trainer as trainer_module
from autogaze.supervised_checkpoint import supervised_resume_contract, verify_supervised_completion
from tests.test_supervised_checkpoint_completion import ReceiptModel


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(trainer_module.wandb, "define_metric", lambda *args, **kwargs: None)


def stage_config(stage):
    config_dir = str(Path(__file__).resolve().parents[1] / "autogaze/configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name=f"av_gaze_stavis_supervised_k16_{stage}",
            overrides=["trainer.gaze_weights=unused-fixture"],
        )


def make_trainer(cfg, directory, *, optimizer_name="adam", optimizer_type=torch.optim.Adam):
    directory.mkdir(parents=True, exist_ok=True)
    model = ReceiptModel()
    loader = torch.utils.data.DataLoader(torch.arange(8), batch_size=1)
    loader.sampler.seed = int(cfg.trainer.seed)
    optimizer = optimizer_type(model.parameters(), lr=float(cfg.trainer.lr))
    trainer = instantiate(
        cfg.trainer,
        gaze_model=model,
        task=torch.nn.Identity(),
        algorithm=instantiate(cfg.algorithm),
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        optimizer_name=optimizer_name,
        save_dir=str(directory),
        grad_acc_steps=4,
        gaze_processor=None,
        gaze_weights=None,
    )
    assert trainer.optimizer is optimizer
    return trainer


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_resolved_real_trainer_saves_reloads_optimizer_scheduler_teacher_and_cursor(tmp_path, runtime, stage):
    cfg = stage_config(stage)
    original_config = OmegaConf.to_container(cfg, resolve=True)
    trainer = make_trainer(cfg, tmp_path / "first")
    mass = {"cell_mass": torch.arange(1, 197).float().expand(1, 16, -1)}
    trainer.algorithm.preprocess_inputs(dict(mass))
    trainer.gaze_model.weight.square().sum().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    trainer.scheduler.step()
    trainer.train_step = 1
    trainer.save_checkpoint(epoch=0, iteration=4)
    expected_teacher = trainer.algorithm.preprocess_inputs(dict(mass))["gt_gazing_info"]["gazing_pos"]
    completion = verify_supervised_completion(trainer.save_dir, expected_contract=supervised_resume_contract(trainer))
    saved = torch.load(Path(trainer.save_dir) / "checkpoint_latest_train.pt", weights_only=True)
    assert saved["config"]["optimizer"] == completion["resume_contract"]["optimizer"] == "adam"
    assert saved["config"]["seed"] == cfg.trainer.seed
    assert saved["config"]["lr_schedule"] == cfg.trainer.lr_schedule
    assert saved["supervised_algorithm_state"]["teacher_seed"] == cfg.trainer.seed
    assert (saved["epoch"], saved["iteration"], saved["train_step"]) == (0, 4, 1)
    if stage == "stage2":
        assert saved["config"]["save_train_steps"] == [2685, 7685, 12685]

    restored = make_trainer(cfg, tmp_path / "second")
    restored.load_checkpoint(resume_path=trainer.save_dir, resume=True)
    torch.testing.assert_close(restored.gaze_model.weight, trainer.gaze_model.weight, rtol=0, atol=0)
    assert (restored.train_step, restored.start_epoch, restored.start_iteration) == (1, 0, 4)
    assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()
    for key, value in trainer.optimizer.state_dict()["state"][0].items():
        torch.testing.assert_close(restored.optimizer.state_dict()["state"][0][key], value, rtol=0, atol=0)
    actual_teacher = restored.algorithm.preprocess_inputs(dict(mass))["gt_gazing_info"]["gazing_pos"]
    assert torch.equal(actual_teacher, expected_teacher)
    assert OmegaConf.to_container(cfg, resolve=True) == original_config


def test_stage_two_boundary_has_same_model_fresh_adam_and_continuation_teacher_seed(tmp_path, runtime):
    first = make_trainer(stage_config("stage1"), tmp_path / "stage1")
    first.gaze_model.weight.square().sum().backward()
    first.optimizer.step()
    first.train_step = 1
    first.save_checkpoint(epoch=0, iteration=4)
    second = make_trainer(stage_config("stage2"), tmp_path / "stage2")
    second.load_checkpoint(gaze_model_path=str(Path(first.save_dir) / "checkpoint_latest_gaze"), resume=False)
    torch.testing.assert_close(second.gaze_model.weight, first.gaze_model.weight, rtol=0, atol=0)
    assert second.optimizer.state_dict()["state"] == {}
    assert second.config["optimizer"] == "adam" and second.config["lr"] == 3e-6
    assert second.algorithm.teacher_seed == second.config["seed"] == 540826


@pytest.mark.parametrize("name,optimizer_type", [(None, torch.optim.Adam), ("sgd", torch.optim.Adam), ("adam", torch.optim.SGD)])
def test_optimizer_identity_is_required_and_bound_to_actual_object(tmp_path, runtime, name, optimizer_type):
    with pytest.raises(Exception, match="recipe optimizer_name matching"):
        make_trainer(stage_config("stage1"), tmp_path, optimizer_name=name, optimizer_type=optimizer_type)
