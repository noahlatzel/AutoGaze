import importlib.util
from pathlib import Path

import pytest
import torch

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES


SCRIPT = Path(__file__).parents[1] / "scripts" / "human_gaze" / "preflight_supervised_k16.py"
SPEC = importlib.util.spec_from_file_location("preflight_supervised_k16", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_first_stable_index_is_lexicographic_and_source_ordered() -> None:
    records = []
    for source in reversed(STAVIS_SOURCES):
        records.extend(
            [
                {"source": source, "clip_id": "z", "video_id": "a"},
                {"source": source, "clip_id": "a", "video_id": "z"},
                {"source": source, "clip_id": "a", "video_id": "a"},
            ]
        )
    indices = MODULE.first_stable_index_per_source(records)
    assert [records[index]["source"] for index in indices] == list(STAVIS_SOURCES)
    assert [(records[index]["clip_id"], records[index]["video_id"]) for index in indices] == [
        ("a", "a") for _ in STAVIS_SOURCES
    ]


def test_verify_exact_k_accepts_fine_unique_global_positions() -> None:
    local = torch.arange(MODULE.EXACT_K).repeat(MODULE.CLIP_LEN, 1) + MODULE.FINE_OFFSET
    offsets = torch.arange(MODULE.CLIP_LEN).reshape(-1, 1) * MODULE.ACTIONS_PER_FRAME
    positions = (local + offsets).reshape(1, -1)
    info = {
        "gazing_pos": positions,
        "num_gazing_each_frame": torch.full((MODULE.CLIP_LEN,), MODULE.EXACT_K),
        "if_padded_gazing": torch.zeros_like(positions, dtype=torch.bool),
    }
    fine = MODULE.verify_exact_k(info)
    assert fine.shape == (1, MODULE.CLIP_LEN, MODULE.EXACT_K)
    assert torch.equal(fine[0, 0], torch.arange(MODULE.EXACT_K))


def test_verify_exact_k_rejects_duplicate_fine_cells() -> None:
    local = torch.arange(MODULE.EXACT_K).repeat(MODULE.CLIP_LEN, 1) + MODULE.FINE_OFFSET
    local[:, 1] = local[:, 0]
    offsets = torch.arange(MODULE.CLIP_LEN).reshape(-1, 1) * MODULE.ACTIONS_PER_FRAME
    positions = (local + offsets).reshape(1, -1)
    info = {
        "gazing_pos": positions,
        "num_gazing_each_frame": torch.full((MODULE.CLIP_LEN,), MODULE.EXACT_K),
        "if_padded_gazing": torch.zeros_like(positions, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="repeat"):
        MODULE.verify_exact_k(info)
