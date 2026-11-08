# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Off-the-shelf video saliency baseline for human-attention coverage (gate SB1).

TASED-Net (DHF1K checkpoint) is applied with the inference protocol of the
upstream ``run_example.py`` (MichiganCOG/TASED-Net @ 0cf47b7), re-implemented
here without OpenCV/SciPy:

* every native frame is read as RGB and resized to 384 x 224 (W x H) with
  half-pixel bilinear interpolation without antialiasing (``cv2.INTER_LINEAR``),
  rounded to uint8, and scaled to [-1, 1] by ``(2x - 255) / 255``;
* the map of native frame ``j`` (0-based) is predicted from the 32-frame window
  ``j-31 .. j``; for ``j < 31`` the upstream code uses the time-reversed window
  ``j+31 .. j`` (it also ends at the target frame);
* the sigmoid output is truncated to 8 bit, blurred with a Gaussian (sigma 7,
  SciPy ``gaussian_filter`` defaults: reflect boundary, truncate 4), divided by
  its maximum and truncated to the saved uint8 PNG.

The PNG-equivalent map is resized to 224 x 224 with PIL bilinear (the
transform the D0 ground-truth maps received), normalised, and summed into the
14 x 14 fine-cell grid (cell = row * 14 + col).  The baseline selects the 16
cells with the largest predicted mass per frame, ties broken by cell index.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES, heatmap_to_cell_mass

LEN_TEMPORAL = 32
INPUT_HW = (224, 384)
GAUSSIAN_SIGMA = 7.0
GAUSSIAN_TRUNCATE = 4.0
GRID = 14
NUM_CELLS = GRID * GRID


def native_window(frame_number: int, len_temporal: int = LEN_TEMPORAL) -> list[int]:
    """1-based native frame numbers fed to TASED-Net for a 1-based target frame.

    ``run_example.py`` slides a window over 0-based frames: at step ``i`` the clip
    ``i-31..i`` predicts frame ``i``; for ``i < 62`` the flipped clip additionally
    predicts frame ``i-31`` (frames ``i..i-31``).  Both end at the target frame.
    """
    target = int(frame_number) - 1
    if target < 0:
        raise ValueError("Frame numbers are 1-based")
    if target >= len_temporal - 1:
        frames = range(target - len_temporal + 1, target + 1)
    else:
        frames = range(target + len_temporal - 1, target - 1, -1)
    return [f + 1 for f in frames]


def min_video_frames(len_temporal: int = LEN_TEMPORAL) -> int:
    """Upstream skips videos shorter than ``2 * T - 1`` frames."""
    return 2 * len_temporal - 1


def resize_like_cv2_linear(image: np.ndarray, size_hw: tuple[int, int] = INPUT_HW) -> np.ndarray:
    """uint8 HxWx3 -> uint8 resized with half-pixel bilinear, no antialias (cv2.INTER_LINEAR)."""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 RGB frame")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float()
    resized = F.interpolate(tensor, size=size_hw, mode="bilinear", align_corners=False, antialias=False)
    return resized[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def load_native_frame(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return resize_like_cv2_linear(np.asarray(image.convert("RGB"), dtype=np.uint8))


def to_model_input(frames_uint8: torch.Tensor) -> torch.Tensor:
    """[B, T, H, W, 3] uint8 -> [B, 3, T, H, W] float in [-1, 1] (upstream ``transform``)."""
    if frames_uint8.dtype != torch.uint8 or frames_uint8.ndim != 5 or frames_uint8.shape[-1] != 3:
        raise ValueError("Expected [B, T, H, W, 3] uint8 frames")
    x = frames_uint8.permute(0, 4, 1, 2, 3).float()
    return x.mul(2.0).sub(255.0).div(255.0)


def gaussian_kernel(sigma: float = GAUSSIAN_SIGMA, truncate: float = GAUSSIAN_TRUNCATE) -> np.ndarray:
    """SciPy ``gaussian_filter`` 1-D kernel (radius ``int(truncate * sigma + 0.5)``)."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 / (sigma * sigma) * x * x)
    return kernel / kernel.sum()


def gaussian_filter_reflect(image: np.ndarray, sigma: float = GAUSSIAN_SIGMA,
                            truncate: float = GAUSSIAN_TRUNCATE) -> np.ndarray:
    """Separable Gaussian blur with SciPy's default ``reflect`` (half-sample symmetric) boundary."""
    out = np.asarray(image, dtype=np.float64)
    kernel = gaussian_kernel(sigma, truncate)
    radius = (len(kernel) - 1) // 2
    for axis in range(out.ndim):
        if out.shape[axis] <= radius:
            raise ValueError("Image too small for the symmetric padding radius")
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="symmetric")
        windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=axis)
        out = windows @ kernel  # symmetric kernel: correlation == convolution
    return out


def postprocess_prediction(smap: np.ndarray) -> np.ndarray:
    """Model sigmoid map (H x W float32) -> the uint8 PNG ``run_example.process`` saves."""
    smap = np.asarray(smap, dtype=np.float32)
    if smap.ndim != 2 or not np.isfinite(smap).all():
        raise ValueError("Prediction must be a finite 2-D map")
    quantised = (smap * 255.0).astype(np.int64) / 255.0
    blurred = gaussian_filter_reflect(quantised)
    peak = blurred.max()
    if not peak > 0:
        raise ValueError("Saliency map has no positive mass")
    return (blurred / peak * 255.0).astype(np.uint8)


def png_to_cell_mass(png: np.ndarray, image_size: int = 224) -> np.ndarray:
    """uint8 PNG map -> PIL bilinear 224 x 224 -> normalised 14 x 14 fine-cell mass (196,)."""
    resized = Image.fromarray(np.asarray(png, dtype=np.float32), mode="F").resize(
        (image_size, image_size), resample=Image.Resampling.BILINEAR)
    heat = torch.from_numpy(np.clip(np.asarray(resized, dtype=np.float64), 0.0, None))
    return heatmap_to_cell_mass(heat, grid_size=GRID)[0].numpy()


def top_k_cells(cell_mass: np.ndarray, k: int = 16) -> np.ndarray:
    """Indices of the ``k`` largest masses along the last axis; ties broken by smaller cell index."""
    mass = np.asarray(cell_mass, dtype=np.float64)
    if mass.shape[-1] != NUM_CELLS or not np.isfinite(mass).all():
        raise ValueError("cell_mass must be finite with 196 cells on the last axis")
    order = np.argsort(-mass, axis=-1, kind="stable")  # stable: equal masses keep index order
    return order[..., :k].astype(np.int64)


def boundary_tie_count(cell_mass: np.ndarray, k: int = 16) -> int:
    """Number of frames whose k-th and (k+1)-th largest cell masses are exactly equal."""
    ordered = -np.sort(-np.asarray(cell_mass, dtype=np.float64), axis=-1)
    return int((ordered[..., k - 1] == ordered[..., k]).sum())


def load_tased_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Weights-only load of ``TASED_updated.pt`` with the ``module.`` (DataParallel) prefix removed."""
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping) or not all(isinstance(v, torch.Tensor) for v in raw.values()):
        raise ValueError("Checkpoint is not a plain tensor state dict")
    return {(".".join(k.split(".")[1:]) if "module" in k else k): v for k, v in raw.items()}


class LastWriteMaxUnpool3d(torch.nn.Module):
    """``nn.MaxUnpool3d`` with PyTorch's CPU semantics on any device.

    TASED-Net unpools with switches from overlapping 3x3/stride-2 poolings, so
    several inputs can target the same output cell.  The CUDA kernel resolves
    these collisions by a write race (run-to-run nondeterministic); the CPU
    kernel writes inputs in flat order, so the last input wins.  This module
    reproduces the CPU rule deterministically with a scatter-amax over input
    positions.
    """

    def __init__(self, reference: torch.nn.MaxUnpool3d):
        super().__init__()
        self.kernel_size, self.stride, self.padding = reference.kernel_size, reference.stride, reference.padding

    def _output_shape(self, values: torch.Tensor, output_size) -> tuple[int, int, int]:
        if output_size is not None:
            return tuple(int(s) for s in tuple(output_size)[-3:])
        return tuple((values.shape[2 + d] - 1) * self.stride[d] - 2 * self.padding[d] + self.kernel_size[d]
                     for d in range(3))

    def forward(self, values: torch.Tensor, indices: torch.Tensor, output_size=None) -> torch.Tensor:
        b, c = values.shape[:2]
        shape = self._output_shape(values, output_size)
        plane = shape[0] * shape[1] * shape[2]
        flat_values = values.reshape(b, c, -1)
        flat_indices = indices.reshape(b, c, -1)
        if flat_indices.min() < 0 or flat_indices.max() >= plane:
            raise ValueError("Unpool index outside the output plane")
        order = torch.arange(flat_values.shape[-1], device=values.device).expand_as(flat_indices)
        winner = torch.full((b, c, plane), -1, dtype=torch.long, device=values.device)
        winner.scatter_reduce_(2, flat_indices, order, reduce="amax", include_self=True)
        out = torch.gather(flat_values, 2, winner.clamp(min=0))
        out = torch.where(winner >= 0, out, torch.zeros((), dtype=out.dtype, device=out.device))
        return out.reshape(b, c, *shape)


def make_unpooling_deterministic(model: torch.nn.Module) -> list[str]:
    replaced = []
    for name, module in list(model.named_children()):
        if isinstance(module, torch.nn.MaxUnpool3d):
            setattr(model, name, LastWriteMaxUnpool3d(module))
            replaced.append(name)
    return replaced


def build_tased(weights_path: Path, *, deterministic_unpool: bool = True) -> torch.nn.Module:
    """TASED_v2 with every parameter/buffer loaded strictly (upstream copies by name and size)."""
    from autogaze.human_gaze.third_party.tased_net import TASED_v2

    model = TASED_v2()
    if deterministic_unpool and make_unpooling_deterministic(model) != ["unpool1", "unpool2", "unpool3"]:
        raise ValueError("Unexpected TASED-Net unpooling layers")
    state = load_tased_state_dict(weights_path)
    expected = model.state_dict()
    if set(state) != set(expected):
        missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
        raise ValueError(f"State dict keys differ: missing={missing[:5]} unexpected={unexpected[:5]}")
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"Shape mismatch for {name}")
    model.load_state_dict(state, strict=True)
    return model.eval()


def best_constant_cells(records: Sequence[Mapping], cell_mass: np.ndarray, *, k: int = 16,
                        split: str = "train", sources: Sequence[str] = STAVIS_SOURCES) -> tuple[list[int], np.ndarray]:
    """16 cells with the largest attention mass under the evaluation weights, fitted on one split.

    Weight per frame = 1 / (sources * videos_in_source * clips_in_video * frames_in_clip),
    i.e. equal sources, equal videos within a source, equal clips within a video,
    equal frames within a clip.  Returns (cells by rank, weighted 196-cell mass).
    """
    mass = np.asarray(cell_mass, dtype=np.float64)
    if mass.ndim != 3 or mass.shape[0] != len(records) or mass.shape[2] != NUM_CELLS:
        raise ValueError("cell_mass must have shape [records, frames, 196]")
    tree: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        if record["split"] == split:
            tree[str(record["source"])][str(record["video_id"])].append(index)
    if set(tree) != set(sources):
        raise ValueError(f"Split {split} does not cover every source")
    weighted = np.zeros(NUM_CELLS, dtype=np.float64)
    for source in sources:
        videos = tree[source]
        for clip_indices in videos.values():
            weight = 1.0 / (len(sources) * len(videos) * len(clip_indices))
            for index in clip_indices:
                weighted += weight * mass[index].mean(axis=0)
    cells = top_k_cells(weighted, k).tolist()
    return cells, weighted
