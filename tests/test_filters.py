"""Unit tests for filters/. Run from the project root:  python -m pytest -q
SciPy / OpenCV appear ONLY here, as independent references."""
import math

import numpy as np
import pytest

from filters import (denoise_bilateral, denoise_gaussian, denoise_median,
                     denoise_perona_malik)
from filters.core import convolve2d, correlate2d, gaussian_kernel1d

rng = np.random.RandomState(0)
NOISY = (np.clip(120 + 25 * rng.randn(64, 64), 0, 255)).astype(np.float32)

ALL = [
    (denoise_gaussian, dict(sigma=1.5)),
    (denoise_median, dict(size=5)),
    (denoise_bilateral, dict(sigma_s=2.0, sigma_r=20.0)),
    (denoise_perona_malik, dict(num_iter=10, lam=0.2, K=20.0)),
    (denoise_perona_malik, dict(num_iter=10, lam=0.2, K=20.0, conductance="exp")),
]


@pytest.mark.parametrize("fn,kw", ALL)
def test_constant_image_unchanged(fn, kw):
    img = np.full((32, 32), 77.0, dtype=np.float32)
    assert np.allclose(fn(img, **kw), 77.0, atol=1e-3)


@pytest.mark.parametrize("fn,kw", ALL)
def test_shape_and_range_preserved(fn, kw):
    out = fn(NOISY, **kw)
    assert out.shape == NOISY.shape
    assert out.min() >= NOISY.min() - 1e-3 and out.max() <= NOISY.max() + 1e-3  # convex combos
    rgb = np.stack([NOISY, NOISY[::-1], NOISY[:, ::-1]], axis=2)
    assert fn(rgb, **kw).shape == rgb.shape


@pytest.mark.parametrize("fn,kw", ALL)
def test_reduces_noise_on_flat_region(fn, kw):
    assert fn(NOISY, **kw).std() < NOISY.std()


def test_correlate_vs_convolve_flip():
    a = NOISY[:, :, None]
    k = rng.rand(3, 3).astype(np.float32)
    assert np.allclose(convolve2d(a, k), correlate2d(a, k[::-1, ::-1]))


def test_gaussian_kernel_normalized_and_symmetric():
    k = gaussian_kernel1d(1.7)
    assert abs(k.sum() - 1) < 1e-6 and np.allclose(k, k[::-1])


@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.0])
def test_gaussian_matches_scipy(sigma):
    nd = pytest.importorskip("scipy.ndimage")
    ref = nd.gaussian_filter(NOISY, sigma, truncate=3.0, mode="reflect")
    r = int(math.ceil(3 * sigma))
    ours = denoise_gaussian(NOISY, sigma=sigma)
    assert np.allclose(ours[r:-r, r:-r], ref[r:-r, r:-r], atol=1e-3)


@pytest.mark.parametrize("size", [3, 5, 7])
def test_median_matches_scipy(size):
    nd = pytest.importorskip("scipy.ndimage")
    ref = nd.median_filter(NOISY, size=size, mode="reflect")
    r = size // 2
    assert np.allclose(denoise_median(NOISY, size=size)[r:-r, r:-r], ref[r:-r, r:-r])


def test_median_removes_salt_and_pepper():
    img = np.full((40, 40), 100.0, dtype=np.float32)
    img[rng.rand(40, 40) < 0.05] = 255.0
    assert np.allclose(denoise_median(img, size=3), 100.0)


def test_bilateral_matches_bruteforce():
    img = (NOISY[:14, :14]).copy()
    s_s, s_r, radius = 1.5, 20.0, 5
    pad = np.pad(img, radius, mode="edge")
    ref = np.zeros_like(img)
    for y in range(14):
        for x in range(14):
            num = den = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy * dy + dx * dx > radius * radius:
                        continue
                    v = pad[radius + y + dy, radius + x + dx]
                    w = math.exp(-(dy * dy + dx * dx) / (2 * s_s ** 2)) * \
                        math.exp(-((v - img[y, x]) ** 2) / (2 * s_r ** 2))
                    num += w * v
                    den += w
            ref[y, x] = num / den
    out = denoise_bilateral(img, sigma_s=s_s, sigma_r=s_r, k=3.0, max_radius=radius)
    # radius = min(ceil(3*1.5)=5, 5) = 5
    assert np.allclose(out, ref, atol=1e-3)


def test_bilateral_keeps_step_edge():
    img = np.zeros((32, 32), dtype=np.float32)
    img[:, 16:] = 200.0
    out = denoise_bilateral(img, sigma_s=3.0, sigma_r=10.0)
    assert np.allclose(out, img, atol=1e-2)


def test_perona_malik_preserves_mean():
    for cond in ("rational", "exp"):
        out = denoise_perona_malik(NOISY, num_iter=30, lam=0.25, K=15.0, conductance=cond)
        assert abs(out.mean() - NOISY.mean()) < 1e-2


def test_perona_malik_stability_guard_and_edge_preservation():
    with pytest.raises(ValueError):
        denoise_perona_malik(NOISY, lam=0.3)
    step = np.zeros((32, 32), dtype=np.float32)
    step[:, 16:] = 200.0
    out = denoise_perona_malik(step, num_iter=50, lam=0.2, K=10.0)
    assert out[:, 15].max() < 3.0 and out[:, 16].min() > 197.0  # edge barely leaks (c is tiny at the step)
