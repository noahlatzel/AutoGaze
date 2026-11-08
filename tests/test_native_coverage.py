from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from autogaze.datasets.av_gaze_stavis import heatmap_to_cell_mass
from autogaze.human_gaze.coverage import center_order
from autogaze.human_gaze.native_coverage import (
    LAYOUT, action_box, centre_area_coverage, fine_cells_to_actions, frame_coverage,
    native_actions_from_generation, normalise_heatmap, scale_counts, union_mask,
)
from autogaze.models.autogaze.autogaze import AutoGaze


def random_heat(seed=0):
    rng = np.random.default_rng(seed)
    return normalise_heatmap(rng.random((224, 224)) ** 4)


def test_layout_matches_autogaze_token_split():
    counts = [int(s**2 / sum(x**2 for x in (32, 64, 112, 224)) * 265) for s in (32, 64, 112, 224)]
    assert [b["count"] for b in LAYOUT] == counts == [4, 16, 49, 196]
    assert [b["offset"] for b in LAYOUT] == [0, 4, 20, 69]
    assert [b["grid"] for b in LAYOUT] == [2, 4, 7, 14]
    assert [b["side_px"] for b in LAYOUT] == [112, 56, 32, 16]


@pytest.mark.parametrize("action,expected", [
    (0, (0, 0, 112, 0, 112)), (1, (0, 0, 112, 112, 224)), (3, (0, 112, 224, 112, 224)),
    (4, (1, 0, 56, 0, 56)), (5, (1, 0, 56, 56, 112)), (8, (1, 56, 112, 0, 56)), (19, (1, 168, 224, 168, 224)),
    (20, (2, 0, 32, 0, 32)), (27, (2, 32, 64, 0, 32)), (68, (2, 192, 224, 192, 224)),
    (69, (3, 0, 16, 0, 16)), (70, (3, 0, 16, 16, 32)), (83, (3, 16, 32, 0, 16)), (264, (3, 208, 224, 208, 224)),
])
def test_action_box_corners_all_scales(action, expected):
    assert action_box(action) == expected


def test_out_of_vocabulary_rejected():
    with pytest.raises(ValueError):
        action_box(265)
    with pytest.raises(ValueError):
        union_mask([265])
    with pytest.raises(ValueError):
        union_mask([5, 5])


def _model_pixel_mask(actions):
    """Independent route: AutoGaze's own per-scale mask, row-major reshape, nearest upsampling."""
    fake = SimpleNamespace(num_vision_tokens_each_frame=265, frame_sampling_rate=1, scales=[32, 64, 112, 224],
                           num_vision_tokens_each_scale_each_frame=[4, 16, 49, 196])
    pos = torch.tensor([actions])
    masks = AutoGaze.get_mask_from_gazing_pos(fake, torch.zeros(1, 1, 3, 224, 224), pos,
                                              torch.zeros_like(pos, dtype=torch.bool), torch.tensor([len(actions)]))
    union = torch.zeros(224, 224, dtype=torch.bool)
    for mask, grid in zip(masks, (2, 4, 7, 14)):
        up = F.interpolate(mask[0, 0].reshape(1, 1, grid, grid), size=(224, 224), mode="nearest")
        union |= up[0, 0].bool()
    return union.numpy()


def test_union_mask_matches_autogaze_scale_masks_for_every_action_and_random_sets():
    for action in range(265):
        assert np.array_equal(union_mask([action]), _model_pixel_mask([action])), action
    rng = np.random.default_rng(3)
    for _ in range(50):
        actions = rng.choice(265, size=int(rng.integers(1, 60)), replace=False).tolist()
        assert np.array_equal(union_mask(actions), _model_pixel_mask(actions))


def test_union_counts_overlap_once():
    mask = union_mask([0, 69, 4])  # 2x2 top-left contains fine cell 0 and 4x4 cell 0
    assert mask.sum() == 112 * 112
    assert scale_counts([0, 69, 4]).tolist() == [1, 1, 0, 1]


def test_fine_only_pixel_coverage_equals_cell_mass():
    heat = random_heat(1)
    cells = heatmap_to_cell_mass(torch.from_numpy(heat[None])).double().numpy()[0]
    selected = [0, 13, 90, 105, 195, 182]
    cov, area = frame_coverage(heat, fine_cells_to_actions(selected))
    assert cov == pytest.approx(cells[selected].sum(), abs=1e-12)
    assert area == 6 * 256


def test_centre_at_area_reproduces_center16_and_is_exact_area():
    heat = random_heat(2)
    cells = heatmap_to_cell_mass(torch.from_numpy(heat[None])).double().numpy()[0]
    center16 = cells[center_order(14)[:16].numpy()].sum()
    assert centre_area_coverage(heat, 4096) == pytest.approx(center16, abs=1e-12)
    assert centre_area_coverage(heat, 224 * 224) == pytest.approx(1.0)
    inner, outer = heat[110:114, 110:114].sum(), heat[109:115, 109:115].sum()
    assert centre_area_coverage(heat, 16) == pytest.approx(inner)
    assert centre_area_coverage(heat, 26) == pytest.approx(inner + (10 / 20) * (outer - inner))
    values = [centre_area_coverage(heat, a) for a in range(1, 224 * 224, 997)]
    assert all(b >= a - 1e-15 for a, b in zip(values, values[1:]))


def test_native_actions_from_generation_drops_padding_and_offsets():
    # two clips, two frames; frame 0 slots 3, frame 1 slots 2; EOS=265 padding
    pos = np.array([[0, 70, 265, 265 + 5, 265 + 264],
                    [3, 265, 100, 265 + 7, 265 + 265]])
    padded = np.array([[False, False, True, False, False],
                       [False, True, False, False, True]])
    clips = native_actions_from_generation(pos, padded, [3, 2])
    assert clips[0] == [[0, 70], [5, 264]]
    assert clips[1] == [[3, 100], [7]]  # action kept after a stop-padded slot stays selected


def test_native_actions_reject_leaked_eos():
    pos = np.array([[0, 265]])
    with pytest.raises(ValueError):
        native_actions_from_generation(pos, np.zeros((1, 2), dtype=bool), [2])
