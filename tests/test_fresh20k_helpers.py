from pathlib import Path

import torch

from scripts.human_gaze.diagnose_center_collapse import resolve_checkpoint


def test_resolve_checkpoint_direct_path() -> None:
    checkpoint = Path("checkpoint_latest_gaze")
    assert resolve_checkpoint(checkpoint, None, None) == checkpoint


def test_resolve_checkpoint_exact_train_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_ep4_iter8"
    gaze = checkpoint / "checkpoint_gaze"
    gaze.mkdir(parents=True)
    torch.save({"train_step": 12685}, checkpoint / "checkpoint_train.pt")
    assert resolve_checkpoint(None, tmp_path, 12685) == gaze


def test_resolve_checkpoint_rejects_ambiguous_arguments() -> None:
    checkpoint = Path("checkpoint_latest_gaze")
    try:
        resolve_checkpoint(checkpoint, Path("run"), 10)
    except ValueError as error:
        assert "not both" in str(error)
    else:
        raise AssertionError("Expected conflicting checkpoint arguments to fail")
