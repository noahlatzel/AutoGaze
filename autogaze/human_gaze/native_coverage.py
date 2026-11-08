# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-level human-attention coverage for native multi-scale AutoGaze actions (gate N1).

Geometry follows the AutoGaze code, not a hand-written table:

* ``AutoGaze.__init__`` sorts ``config.scales`` ("32+64+112+224") ascending and
  gives scale ``s`` ``int(s**2 / sum(s_i**2) * 265)`` tokens: 4, 16, 49, 196.
  Local action IDs are the concatenation of these blocks in ascending scale
  order (``get_mask_from_gazing_pos`` splits the 265-vector in that order):
  0-3 the 2x2 grid, 4-19 the 4x4 grid, 20-68 the 7x7 grid, 69-264 the 14x14 grid;
  265 is EOS.
* The downstream multi-scale ViT (``modeling_video_mae`` embeddings) resizes
  the 224-px frame to ``s x s`` and embeds 16-px patches row-major, so one
  token of scale ``s`` covers a square of ``224 * 16 / s`` pixels of the
  224-px frame: 112, 56, 32 and 16 px.  All edges are integers on the
  224 x 224 attention map, so the union of selected patches is an exact pixel set.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

IMAGE_SIZE = 224
PATCH_SIZE = 16
NATIVE_SCALES = (32, 64, 112, 224)
NUM_ACTIONS = 265
EOS_ACTION = 265


def scale_layout(
    scales: Sequence[int] = NATIVE_SCALES,
    patch_size: int = PATCH_SIZE,
    num_tokens: int = NUM_ACTIONS,
    image_size: int = IMAGE_SIZE,
) -> list[dict]:
    """Per-scale action-ID block, grid size and pixel side on the frame."""
    ordered = sorted(int(s) for s in scales)
    total = sum(s**2 for s in ordered)
    layout, offset = [], 0
    for scale in ordered:
        count = int(scale**2 / total * num_tokens)  # AutoGaze.__init__ formula
        grid = scale // patch_size
        if grid * grid != count:
            raise ValueError(f"Scale {scale} token count {count} is not a {grid}x{grid} grid")
        side = image_size * patch_size / scale
        if side != int(side):
            raise ValueError(f"Scale {scale} patch side {side} is not an integer pixel count")
        layout.append({"scale": scale, "grid": grid, "offset": offset, "count": count,
                       "side_px": int(side)})
        offset += count
    if offset != num_tokens:
        raise ValueError("Scale blocks do not cover the action vocabulary")
    return layout


LAYOUT = scale_layout()


def action_box(action: int, layout: Sequence[dict] = LAYOUT) -> tuple[int, int, int, int, int]:
    """Return ``(scale_index, y0, y1, x0, x1)`` pixel bounds (half-open) of one local action."""
    action = int(action)
    for index, block in enumerate(layout):
        if block["offset"] <= action < block["offset"] + block["count"]:
            row, col = divmod(action - block["offset"], block["grid"])
            side = block["side_px"]
            return index, row * side, (row + 1) * side, col * side, (col + 1) * side
    raise ValueError(f"Action {action} is outside the {NUM_ACTIONS}-action vocabulary")


_BOXES = np.asarray([action_box(a) for a in range(NUM_ACTIONS)], dtype=np.int64)


def validate_frame_actions(actions: Iterable[int], *, max_actions: int | None = None) -> np.ndarray:
    array = np.asarray(list(actions), dtype=np.int64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("A frame must select at least one action")
    if array.min() < 0 or array.max() >= NUM_ACTIONS:
        raise ValueError("Frame action outside 0..264 (EOS or padding leaked)")
    if np.unique(array).size != array.size:
        raise ValueError("Frame repeats an action")
    if max_actions is not None and array.size > max_actions:
        raise ValueError("Frame exceeds its action cap")
    return array


def union_mask(actions: Iterable[int]) -> np.ndarray:
    """Boolean 224x224 mask of the pixel union of the selected patches."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for action in validate_frame_actions(actions):
        _, y0, y1, x0, x1 = _BOXES[action]
        mask[y0:y1, x0:x1] = True
    return mask


def fine_cells_to_actions(cells: Iterable[int]) -> list[int]:
    """Map 14x14 fine-cell indices (0..195) to local action IDs (69..264)."""
    fine = LAYOUT[-1]
    values = [int(c) for c in cells]
    if any(c < 0 or c >= fine["count"] for c in values):
        raise ValueError("Fine cell index out of range")
    return [fine["offset"] + c for c in values]


def scale_counts(actions: Iterable[int]) -> np.ndarray:
    array = validate_frame_actions(actions)
    return np.bincount(_BOXES[array, 0], minlength=len(LAYOUT))


def normalise_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heat = np.asarray(heatmap, dtype=np.float64)
    if heat.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(heat).all() or (heat < 0).any():
        raise ValueError("Heatmap must be a finite nonnegative 224x224 array")
    total = heat.sum()
    if total <= 0:
        raise ValueError("Heatmap has no mass")
    return heat / total


def frame_coverage(heat: np.ndarray, actions: Iterable[int]) -> tuple[float, int]:
    """Return (attention mass inside the pixel union, union area in pixels)."""
    mask = union_mask(actions)
    return float(heat[mask].sum()), int(mask.sum())


def _centre_ring_masses(heat: np.ndarray) -> np.ndarray:
    """Cumulative mass of centred even-sided squares: entry k is side 2k (k=0..112)."""
    half = IMAGE_SIZE // 2
    return np.asarray([0.0] + [float(heat[half - k:half + k, half - k:half + k].sum())
                               for k in range(1, half + 1)])


def centre_area_coverage(heat: np.ndarray, area_px: float, cumulative: np.ndarray | None = None) -> float:
    """Exact-area centred square with a uniformly fractional boundary ring.

    The frame centre lies between pixels 111 and 112, so centred pixel squares
    have even sides 2k.  With k the largest integer such that (2k)^2 <= area,
    coverage = M(2k) + f * (M(2k+2) - M(2k)), f = (area - (2k)^2) / ((2k+2)^2 - (2k)^2),
    i.e. the next ring is credited with the fraction of its pixels needed to
    reach the exact area.  Area 4096 = 64^2 reproduces Center16 exactly.
    """
    area = float(area_px)
    if not 0 < area <= IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError("Area must lie in (0, 224*224]")
    masses = _centre_ring_masses(heat) if cumulative is None else cumulative
    k = int(np.floor(np.sqrt(area) / 2 + 1e-12))
    k = min(k, IMAGE_SIZE // 2)
    while (2 * k) ** 2 > area:
        k -= 1
    if (2 * k) ** 2 == area or k == IMAGE_SIZE // 2:
        return float(masses[k])
    inner, outer = (2 * k) ** 2, (2 * k + 2) ** 2
    fraction = (area - inner) / (outer - inner)
    return float(masses[k] + fraction * (masses[k + 1] - masses[k]))


def native_actions_from_generation(gazing_pos, if_padded, num_gazing_each_frame,
                                   *, actions_per_frame: int = NUM_ACTIONS) -> list[list[list[int]]]:
    """Split AutoGaze generate() output into per-clip, per-frame local action lists.

    ``gazing_pos`` holds global IDs ``local + t * 265``; padded slots (EOS
    emitted by task-loss stopping or batch padding) are flagged by
    ``if_padded`` and dropped.  Generation order is preserved.
    """
    positions = np.asarray(gazing_pos)
    padded = np.asarray(if_padded, dtype=bool)
    counts = [int(c) for c in np.asarray(num_gazing_each_frame).reshape(-1)]
    if positions.shape != padded.shape or positions.shape[1] != sum(counts):
        raise ValueError("Generation outputs have inconsistent shapes")
    clips: list[list[list[int]]] = [[] for _ in range(positions.shape[0])]
    offset = 0
    for t, count in enumerate(counts):
        block, block_padded = positions[:, offset:offset + count], padded[:, offset:offset + count]
        for b in range(positions.shape[0]):
            local = (block[b][~block_padded[b]] - t * actions_per_frame).tolist()
            clips[b].append(validate_frame_actions(local, max_actions=count).tolist())
        offset += count
    return clips
