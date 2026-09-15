from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from autogaze.algorithms.human_gaze_ntp import HumanGazeNTP
from autogaze.trainer import Trainer


class CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([3.0]))

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / 'weights.pt')

    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        model.load_state_dict(torch.load(Path(path) / 'weights.pt', weights_only=True))
        return model


def checkpoint_trainer(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.gaze_model = CheckpointModel()
    trainer.task = torch.nn.Linear(1, 1)
    trainer.algorithm = HumanGazeNTP(teacher_seed=440826)
    trainer.human_supervision = True
    trainer.optimizer = torch.optim.Adam(trainer.gaze_model.parameters(), lr=1e-5)
    trainer.scheduler = torch.optim.lr_scheduler.ConstantLR(trainer.optimizer, factor=1.0)
    trainer.train_step, trainer.val_step = 7, 2
    trainer.config = {
        'seed': 440826, 'batch_size': 4, 'per_gpu_max_batch_size': 1,
        'optimizer': 'adam', 'lr': 1e-5, 'lr_schedule': 'constant',
    }
    trainer.train_loader = torch.utils.data.DataLoader(torch.arange(64), batch_size=1)
    trainer.train_loader.sampler.seed = 440826
    trainer.grad_acc_steps = 4
    trainer.n_epochs = 5
    trainer.save_dir = str(tmp_path)
    trainer.gaze_processor = None
    trainer.max_periodic_checkpoints = 2
    return trainer


def test_resume_restores_model_and_next_teacher_sample(tmp_path):
    trainer = checkpoint_trainer(tmp_path)
    inputs = {'cell_mass': torch.arange(1, 197).float().expand(1, 16, -1)}
    trainer.algorithm.preprocess_inputs(dict(inputs))
    trainer.save_checkpoint(epoch=1, iteration=28)
    expected = trainer.algorithm.preprocess_inputs(dict(inputs))['gt_gazing_info']['gazing_pos']
    with torch.no_grad():
        trainer.gaze_model.weight.fill_(-10)
    trainer.train_step = 99
    trainer.load_checkpoint(resume=True)
    actual = trainer.algorithm.preprocess_inputs(dict(inputs))['gt_gazing_info']['gazing_pos']
    torch.testing.assert_close(trainer.gaze_model.weight, torch.tensor([3.0]))
    assert torch.equal(actual, expected)
    assert (trainer.train_step, trainer.val_step, trainer.start_epoch, trainer.start_iteration) == (7, 2, 1, 28)


def test_supervised_resume_fails_when_teacher_state_is_missing(tmp_path):
    trainer = checkpoint_trainer(tmp_path)
    trainer.save_checkpoint(epoch=1, iteration=28)
    checkpoint = tmp_path / 'checkpoint_latest_train.pt'
    state = torch.load(checkpoint, weights_only=True)
    del state['supervised_algorithm_state']
    torch.save(state, checkpoint)
    # Tampering with serialized teacher state now fails the bundle hash check
    # before deserialization, so the model and algorithm cannot be mutated.
    with pytest.raises(ValueError, match='hash mismatch'):
        trainer.load_checkpoint(resume=True)


def test_supervised_resume_never_silently_starts_fresh(tmp_path):
    trainer = checkpoint_trainer(tmp_path)
    with pytest.raises(FileNotFoundError, match='Supervised resume'):
        trainer.load_checkpoint(resume=True)


def test_supervised_configs_match_rl_data_model_and_training_contract(monkeypatch):
    monkeypatch.delenv('SUPERVISED_STAGE1_CHECKPOINT', raising=False)
    config_dir = str(Path(__file__).resolve().parents[1] / 'autogaze' / 'configs')
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        rl1 = compose(config_name='av_gaze_stavis_human_coverage_grpo_no_kl_fresh10k_stage1')
        sl1 = compose(config_name='av_gaze_stavis_supervised_k16_stage1')
        rl2 = compose(config_name='av_gaze_stavis_human_coverage_grpo_no_kl_fresh20k_stage3')
        sl2 = compose(config_name='av_gaze_stavis_supervised_k16_stage2')
    for rl, sl in [(rl1, sl1), (rl2, sl2)]:
        for component in ('dataset', 'model', 'task'):
            assert OmegaConf.to_container(rl[component], resolve=True) == OmegaConf.to_container(sl[component], resolve=True)
        for field in ('batch_size', 'per_gpu_max_batch_size', 'per_gpu_max_val_batch_size',
                      'optimizer', 'lr', 'lr_schedule', 'n_epochs', 'grad_norm',
                      'freeze_gaze_vision', 'freeze_gaze_connector', 'train_task', 'val_nsteps'):
            assert rl.trainer[field] == sl.trainer[field], field
    assert sl1.trainer.max_train_steps + sl2.trainer.max_train_steps == 20000
    assert sl1.trainer.seed == 440826 and sl2.trainer.seed == 540826
    assert sl1.algorithm.teacher_seed == sl1.trainer.seed
    assert 'group_size' not in sl1.algorithm
    with pytest.raises(InterpolationResolutionError):
        _ = sl2.trainer.gaze_weights
