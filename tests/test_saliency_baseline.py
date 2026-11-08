import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from autogaze.datasets.av_gaze_stavis import STAVIS_SOURCES
from autogaze.human_gaze.saliency_baseline import (
    LastWriteMaxUnpool3d, best_constant_cells, boundary_tie_count, gaussian_filter_reflect, gaussian_kernel,
    make_unpooling_deterministic, native_window, png_to_cell_mass, postprocess_prediction,
    resize_like_cv2_linear, to_model_input, top_k_cells,
)
from autogaze.human_gaze.third_party.tased_net import TASED_v2

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/human_gaze/configs/sb1_saliency_baseline_v1.yaml"


@pytest.mark.parametrize("frame,expected", [
    (1, list(range(32, 0, -1))),      # flipped first window ends at frame 1
    (31, list(range(62, 30, -1))),
    (32, list(range(1, 33))),          # first regular window
    (100, list(range(69, 101))),
])
def test_native_window_matches_run_example(frame, expected):
    window = native_window(frame)
    assert window == expected and len(window) == 32 and window[-1] == frame


def test_native_window_reproduces_upstream_sliding_loop():
    """Simulate run_example.py's loop on 0-based frames and compare every predicted index."""
    n, T = 130, 32
    produced = {}
    snippet = []
    for i in range(n):
        snippet.append(i)
        if i >= T - 1:
            produced[i] = list(snippet)
            if i < 2 * T - 2:
                produced[i - T + 1] = list(reversed(snippet))
            del snippet[0]
    assert sorted(produced) == list(range(n))
    for j, frames in produced.items():
        assert native_window(j + 1) == [f + 1 for f in frames]


def test_gaussian_kernel_matches_scipy_definition():
    kernel = gaussian_kernel(7.0, 4.0)
    assert kernel.size == 2 * 28 + 1 and np.isclose(kernel.sum(), 1.0)
    x = np.arange(-28, 29)
    expected = np.exp(-0.5 * x**2 / 49.0)
    np.testing.assert_allclose(kernel, expected / expected.sum())


def test_gaussian_filter_reflect_against_bruteforce():
    rng = np.random.default_rng(0)
    image = rng.random((40, 64))
    kernel = gaussian_kernel(7.0, 4.0)
    radius = 28

    def reflect(i, n):  # SciPy 'reflect': d c b a | a b c d | d c b a
        while i < 0 or i >= n:
            i = -i - 1 if i < 0 else 2 * n - i - 1
        return i

    def blur1d(a, axis):
        out = np.zeros_like(a)
        n = a.shape[axis]
        for i in range(n):
            idx = [reflect(i + k, n) for k in range(-radius, radius + 1)]
            out.swapaxes(0, axis)[i] = np.tensordot(kernel, a.swapaxes(0, axis)[idx], axes=(0, 0))
        return out

    np.testing.assert_allclose(gaussian_filter_reflect(image), blur1d(blur1d(image, 0), 1), atol=1e-12)


def test_resize_like_cv2_linear_half_pixel_downscale():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(448, 768, 3), dtype=np.uint8)
    out = resize_like_cv2_linear(image)
    assert out.shape == (224, 384, 3) and out.dtype == np.uint8
    expected = image.reshape(224, 2, 384, 2, 3).astype(np.float64).mean(axis=(1, 3))
    assert np.abs(out.astype(np.float64) - expected).max() <= 0.5 + 1e-9
    const = np.full((300, 500, 3), 77, dtype=np.uint8)
    assert (resize_like_cv2_linear(const) == 77).all()


def test_model_input_scaling_and_layout():
    frames = torch.zeros(2, 32, 224, 384, 3, dtype=torch.uint8)
    frames[..., 0] = 255
    x = to_model_input(frames)
    assert x.shape == (2, 3, 32, 224, 384)
    assert torch.all(x[:, 0] == 1.0) and torch.all(x[:, 1:] == -1.0)


def test_postprocess_matches_run_example_steps():
    rng = np.random.default_rng(2)
    smap = rng.random((224, 384)).astype(np.float32)
    png = postprocess_prediction(smap)
    quant = (smap * 255.0).astype(np.int64) / 255.0
    blurred = gaussian_filter_reflect(quant)
    np.testing.assert_array_equal(png, (blurred / blurred.max() * 255.0).astype(np.uint8))
    assert png.dtype == np.uint8 and png.max() == 255
    with pytest.raises(ValueError):
        postprocess_prediction(np.zeros((224, 384), dtype=np.float32))


def test_png_to_cell_mass_row_major_and_normalised():
    png = np.zeros((224, 384), dtype=np.uint8)
    # 224 rows -> 14 rows of 16 px; 384 cols -> 224 -> col 5 spans x in [80, 96) of 224, i.e. [137, 165) of 384
    png[32:48, 140:162] = 255
    mass = png_to_cell_mass(png)
    assert mass.shape == (196,) and np.isclose(mass.sum(), 1.0)
    assert int(np.argmax(mass)) == 2 * 14 + 5


def test_top_k_ties_by_cell_index():
    mass = np.zeros(196)
    mass[[150, 3, 77]] = 1.0
    mass[[10, 20]] = 0.5
    cells = top_k_cells(mass, 16)
    assert cells[:5].tolist() == [3, 77, 150, 10, 20]
    assert cells[5:].tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12]
    assert boundary_tie_count(mass[None], 16) == 1


def test_best_constant_uses_equal_source_video_clip_weights():
    records, masses = [], []

    def frame(cell):
        m = np.zeros((16, 196))
        m[:, cell] = 1.0
        return m

    # Source 0: video a has 9 clips at cell 1, video b has 1 clip at cell 2 -> each video weight 1/2.
    for i in range(9):
        records.append({"split": "train", "source": STAVIS_SOURCES[0], "video_id": "a"})
        masses.append(frame(1))
    records.append({"split": "train", "source": STAVIS_SOURCES[0], "video_id": "b"})
    masses.append(frame(2))
    for s, source in enumerate(STAVIS_SOURCES[1:], start=1):
        records.append({"split": "train", "source": source, "video_id": "v"})
        masses.append(frame(100 + s))
    records.append({"split": "test", "source": STAVIS_SOURCES[0], "video_id": "t"})
    masses.append(frame(195))
    cells, weighted = best_constant_cells(records, np.stack(masses), k=16)
    n = len(STAVIS_SOURCES)
    assert np.isclose(weighted[1], 0.5 / n) and np.isclose(weighted[2], 0.5 / n)
    assert weighted[195] == 0.0  # test split never used
    assert np.isclose(weighted.sum(), 1.0)
    assert cells[:n - 1] == [100 + s for s in range(1, n)]


@pytest.mark.parametrize("kernel,stride,padding,size", [
    ((1, 3, 3), (1, 2, 2), (0, 1, 1), (2, 3, 2, 27, 45)),
    ((1, 2, 2), (1, 2, 2), (0, 0, 0), (2, 3, 1, 14, 24)),
])
def test_last_write_unpool_equals_cpu_max_unpool(kernel, stride, padding, size):
    torch.manual_seed(0)
    pool = torch.nn.MaxPool3d(kernel, stride, padding, return_indices=True)
    unpool = torch.nn.MaxUnpool3d(kernel, stride, padding)
    x = torch.relu(torch.randn(size))
    y, idx = pool(x)
    values = torch.randn_like(y)
    output_size = x.size() if padding != (0, 0, 0) else None
    reference = unpool(values, idx, output_size)
    ours = LastWriteMaxUnpool3d(unpool)(values, idx, output_size)
    assert torch.equal(reference, ours)
    if padding != (0, 0, 0):
        assert idx[0, 0].numel() > torch.unique(idx[0, 0]).numel()  # collisions actually occur


def test_tased_unpool_replacement_and_vendored_source():
    model = TASED_v2()
    keys = set(model.state_dict())
    assert make_unpooling_deterministic(model) == ["unpool1", "unpool2", "unpool3"]
    assert set(model.state_dict()) == keys
    cfg = yaml.safe_load(CONFIG.read_text())
    vendored = ROOT / "autogaze/human_gaze/third_party/tased_net/model.py"
    assert hashlib.sha256(vendored.read_bytes()).hexdigest() == cfg["tased_net"]["model_py_sha256"]
    assert (vendored.parent / "LICENSE").read_text().startswith("MIT License")


def test_tased_output_shape_small_input():
    model = TASED_v2().eval()
    make_unpooling_deterministic(model)
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 32, 64, 96))
    assert out.shape == (1, 64, 96)
