import numpy as np
import pytest

pytest.importorskip("skimage")
from filters import METHODS
from filters.api import denoise
from filters.nlm import denoise_nlm

rng = np.random.RandomState(0)
FLAT = (120 + 25 * rng.randn(48, 48)).clip(0, 255).astype(np.float32)


def test_nlm_constant_shape_range():
    c = np.full((32, 32), 90.0, np.float32)
    assert np.allclose(denoise_nlm(c, h=20), 90.0, atol=0.5)
    out = denoise_nlm(FLAT, h=25)
    assert out.shape == FLAT.shape and out.min() >= 0 and out.max() <= 255


def test_nlm_reduces_noise():
    assert denoise_nlm(FLAT, h=30).std() < FLAT.std()


def test_registry_has_nlm():
    assert "nlm" in METHODS


def test_api_interface_01_and_no_inplace():
    x = (FLAT / 255.0).astype(np.float32)
    x0 = x.copy()
    y = denoise(x, {"method": "gaussian", "sigma": 1.0})
    assert y.shape == x.shape and y.dtype == np.float32 and 0 <= y.min() and y.max() <= 1
    assert np.array_equal(x, x0)
    y2 = denoise(x, {"method": "nlm", "h": 25.0})
    assert np.isfinite(y2).all()
