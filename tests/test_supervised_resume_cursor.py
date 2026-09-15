"""Check resume continuity through the real trainer loop using CPU stand-ins."""

from pathlib import Path
import time

import pytest
import torch

import autogaze.trainer as trainer_module
from autogaze.algorithms import HumanGazeNTP
from autogaze.datasets.av_gaze_stavis import BalancedSourceSampler, STAVIS_SOURCES
from autogaze.trainer import Trainer


class TinyGaze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float64))

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "weights.pt")

    @classmethod
    def from_pretrained(cls, directory):
        model = cls()
        model.load_state_dict(torch.load(Path(directory) / "weights.pt", weights_only=True))
        return model


class TinyClips(torch.utils.data.Dataset):
    def __init__(self):
        self.records = [
            {"source": source, "video_id": f"video{video}"}
            for source in STAVIS_SOURCES for video in range(2)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {
            "clip_index": index,
            "cell_mass": torch.tensor([[1.0 + index, 2.0, 3.0, 4.0]], dtype=torch.float64),
        }


class QuietProgress:
    def __init__(self, **kwargs):
        pass

    def update(self, *args):
        pass

    def set_description(self, *args):
        pass


@pytest.fixture
def cpu_runtime(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(trainer_module, "move_inputs_to_cuda", lambda inputs: inputs)
    monkeypatch.setattr(trainer_module, "tqdm", QuietProgress)
    monkeypatch.setattr(trainer_module.wandb, "log", lambda values: None)


def tiny_trainer(directory, budget, *, epoch_inputs=8, human=True):
    directory.mkdir(parents=True, exist_ok=True)
    trainer = Trainer.__new__(Trainer)
    trainer.gaze_model = TinyGaze()
    trainer.task = torch.nn.Identity()
    trainer.algorithm = HumanGazeNTP(clip_len=1, actions_per_frame=5, fine_action_offset=1, exact_budget=2, teacher_seed=314)
    trainer.human_supervision = human
    trainer.train_w_ntp = True
    dataset = TinyClips()
    sampler = BalancedSourceSampler(dataset, num_samples=epoch_inputs, seed=1701)
    trainer.train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    trainer.optimizer = torch.optim.Adam(trainer.gaze_model.parameters(), lr=0.03)
    trainer.scheduler = torch.optim.lr_scheduler.ConstantLR(trainer.optimizer, factor=1.0)
    trainer.grad_acc_steps = 2
    trainer.n_epochs = 3
    trainer.train_step = trainer.val_step = trainer.start_epoch = trainer.start_iteration = 0
    trainer.max_train_steps = budget
    trainer.train_gaze, trainer.train_task = True, False
    trainer.truncate_grads = False
    trainer.val_nsteps = 10000
    trainer.save_nsteps = 2
    trainer.save_train_steps = set()
    trainer.validate_at_start = trainer.save_at_start = False
    trainer.val_only = False
    trainer.save_at_end = trainer.skip_final_validation = True
    trainer.save_dir = str(directory)
    trainer.gaze_processor = None
    trainer.max_periodic_checkpoints = None
    trainer.config = {
        "seed": 1701, "batch_size": 2, "per_gpu_max_batch_size": 1,
        "optimizer": "adam", "lr": 0.03, "lr_schedule": "constant",
    }
    trainer.started_at = time.time()
    trainer.temperature = 0.0
    trainer.seen = []
    trainer._recurrent_gradient_metrics = lambda: {}
    trainer.extract_metrics = lambda gaze, task, algorithm: algorithm["metrics"]

    def supervised_step(inputs):
        positions = inputs["gt_gazing_info"]["gazing_pos"]
        clip_id = int(inputs["clip_index"].item())
        trainer.seen.append((clip_id, positions.detach().clone()))
        # A deterministic differentiable stand-in exercises Adam state as well
        # as sampled inputs and teacher histories, without a vision model/GPU.
        target = positions.to(torch.float64).mean() / 5 + clip_id / 12
        loss = (trainer.gaze_model.weight - target).square()
        return inputs["gt_gazing_info"], None, {"loss": loss, "metrics": {"loss": loss.mean()}}

    trainer._one_step_ntp = supervised_step
    return trainer


@pytest.mark.parametrize("stop_updates,expected_cursor", [(3, (0, 6)), (4, (1, 0))])
def test_final_checkpoint_resume_matches_uninterrupted_inputs_teachers_and_adam(
    tmp_path, cpu_runtime, stop_updates, expected_cursor,
):
    uninterrupted = tiny_trainer(tmp_path / "uninterrupted", 7)
    uninterrupted.trainval()
    interrupted = tiny_trainer(tmp_path / "interrupted", stop_updates)
    interrupted.trainval()
    saved = torch.load(tmp_path / "interrupted" / "checkpoint_latest_train.pt", weights_only=True)
    assert (saved["epoch"], saved["iteration"]) == expected_cursor
    assert saved["train_step"] == stop_updates

    resumed = tiny_trainer(tmp_path / "resumed", 7)
    resumed.load_checkpoint(resume_path=str(tmp_path / "interrupted"), resume=True)
    resumed.trainval()
    combined = interrupted.seen + resumed.seen
    assert len(combined) == len(uninterrupted.seen) == 14
    assert [item[0] for item in combined] == [item[0] for item in uninterrupted.seen]
    assert all(torch.equal(left[1], right[1]) for left, right in zip(combined, uninterrupted.seen))
    torch.testing.assert_close(resumed.gaze_model.weight, uninterrupted.gaze_model.weight, rtol=0, atol=0)
    for key, value in uninterrupted.optimizer.state_dict()["state"][0].items():
        torch.testing.assert_close(resumed.optimizer.state_dict()["state"][0][key], value, rtol=0, atol=0)
    assert torch.equal(
        resumed.algorithm.state_dict()["teacher_rng_state"],
        uninterrupted.algorithm.state_dict()["teacher_rng_state"],
    )


def test_periodic_checkpoint_keeps_pre_input_cursor_and_teacher_state(tmp_path, cpu_runtime):
    trainer = tiny_trainer(tmp_path, 3)
    trainer.trainval()
    saved = torch.load(tmp_path / "checkpoint_ep0_iter4" / "checkpoint_train.pt", weights_only=True)
    assert (saved["epoch"], saved["iteration"], saved["train_step"]) == (0, 4, 2)
    restored_teacher = HumanGazeNTP(clip_len=1, actions_per_frame=5, fine_action_offset=1, exact_budget=2)
    restored_teacher.load_state_dict(saved["supervised_algorithm_state"])
    next_clip, expected_positions = trainer.seen[4]
    inputs = TinyClips()[next_clip]
    actual = restored_teacher.preprocess_inputs({"cell_mass": inputs["cell_mass"].unsqueeze(0)})
    assert torch.equal(actual["gt_gazing_info"]["gazing_pos"], expected_positions)


def test_supervised_rejects_partial_update_tail_and_mid_update_resume(tmp_path, cpu_runtime):
    trainer = tiny_trainer(tmp_path / "tail", 2, epoch_inputs=7)
    with pytest.raises(ValueError, match="complete gradient-accumulation groups"):
        trainer.trainval()
    trainer = tiny_trainer(tmp_path / "resume", 2)
    trainer.start_iteration = 1
    with pytest.raises(ValueError, match="gradient-accumulation boundary"):
        trainer.trainval()


def test_legacy_final_checkpoint_cursor_is_unchanged(tmp_path, cpu_runtime):
    trainer = tiny_trainer(tmp_path, 3, human=False)
    trainer.trainval()
    saved = torch.load(tmp_path / "checkpoint_latest_train.pt", weights_only=True)
    assert (saved["epoch"], saved["iteration"]) == (3, 0)
    assert "supervised_algorithm_state" not in saved
