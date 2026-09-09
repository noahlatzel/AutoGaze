import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "human_gaze" / "evaluate_hlvid_nvila_r2d.py"
SPEC = importlib.util.spec_from_file_location("evaluate_hlvid_nvila_r2d", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def established_args(**updates):
    values = {
        "frame_source": "uniform",
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "tile_len": 16,
        "delta_gap": 0,
        "max_new_tokens": 16,
        "start_index": 0,
        "limit": None,
        "delta_policy_checkpoint": None,
        "selected_indices": None,
        "max_source_tiles": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_established_protocol_keeps_mtv48_distinct_from_frame_counts():
    MODULE.validate_established_protocol(established_args(), preflight=False)
    assert MODULE.ESTABLISHED_PROTOCOL["max_tiles_video"] == 48
    assert MODULE.ESTABLISHED_PROTOCOL["num_video_frames"] == 128
    assert MODULE.ESTABLISHED_PROTOCOL["num_video_frames_thumbnail"] == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tiles_video", 36),
        ("num_video_frames", 48),
        ("num_video_frames_thumbnail", 48),
        ("frame_source", "dense"),
        ("tile_len", 8),
        ("max_new_tokens", 32),
    ],
)
def test_established_protocol_blocks_comparability_mismatches(field, value):
    with pytest.raises(ValueError, match="established-protocol mismatch"):
        MODULE.validate_established_protocol(established_args(**{field: value}), preflight=False)


def test_preflight_is_the_only_allowed_partial_run():
    MODULE.validate_established_protocol(established_args(limit=1), preflight=True)
    with pytest.raises(ValueError, match="Expected --limit"):
        MODULE.validate_established_protocol(established_args(limit=1), preflight=False)
