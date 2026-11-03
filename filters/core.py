"""Low-level building blocks for the classical filters.

Conventions
-----------
* Images are NumPy arrays, shape (H, W) or (H, W, C), intensities on the 0..255 scale.
* Internally everything is (H, W, C) float32 (see to_hwc / from_hwc).
* Filters are written with "vectorized shifts": for every kernel offset we add a shifted
  view of the padded image, so there are no per-pixel Python loops.
* OpenCV / scikit-image / SciPy are NOT used here; they appear only in tests/.
"""
from __future__ import annotations

import math

import numpy as np


def to_hwc(img):
    """Return (float32 array of shape (H, W, C), squeezed_flag)."""
    a = np.asarray(img)
    if a.ndim == 2:
        return a.astype(np.float32)[:, :, None], True
    if a.ndim == 3:
        return a.astype(np.float32), False
    raise ValueError(f"expected (H, W) or (H, W, C), got shape {a.shape}")


def from_hwc(a, squeezed):
    return a[:, :, 0] if squeezed else a


# --------------------------------------------------------------------------- correlation
def correlate2d(a, kernel, mode="reflect"):
    """2-D correlation of an (H, W, C) image with an odd-sized (kh, kw) kernel."""
    kernel = np.asarray(kernel, dtype=np.float32)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel sides must be odd")
    rh, rw = kh // 2, kw // 2
    H, W = a.shape[:2]
    p = np.pad(a, ((rh, rh), (rw, rw), (0, 0)), mode=mode)
    out = np.zeros(a.shape, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * p[i:i + H, j:j + W]
    return out


def convolve2d(a, kernel, mode="reflect"):
    """Convolution = correlation with the kernel flipped in both axes."""
    return correlate2d(a, np.asarray(kernel)[::-1, ::-1], mode)


def correlate1d(a, kernel, axis, mode="reflect"):
    """1-D correlation along axis 0 (rows) or 1 (columns) of an (H, W, C) image."""
    kernel = np.asarray(kernel, dtype=np.float32)
    if len(kernel) % 2 == 0:
        raise ValueError("kernel length must be odd")
    r = len(kernel) // 2
    n = a.shape[axis]
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode=mode)
    out = np.zeros(a.shape, dtype=np.float32)
    for i, k in enumerate(kernel):
        sl = [slice(None)] * 3
        sl[axis] = slice(i, i + n)
        out += k * p[tuple(sl)]
    return out


# --------------------------------------------------------------------------- Gaussian
def gaussian_kernel1d(sigma, truncate=3.0):
    """Normalized 1-D Gaussian, radius = ceil(truncate * sigma)."""
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    r = max(1, int(math.ceil(truncate * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def gaussian_filter(a, sigma, truncate=3.0, mode="reflect"):
    """Separable Gaussian filter on an (H, W, C) image."""
    if sigma <= 0:
        return a.copy()
    k = gaussian_kernel1d(sigma, truncate)
    return correlate1d(correlate1d(a, k, 0, mode), k, 1, mode)


# --------------------------------------------------------------------------- median
def median_filter(a, size, mode="reflect", max_block_elems=20_000_000):
    """size x size median filter (size odd). Processes row strips to cap memory."""
    if size % 2 == 0 or size < 1:
        raise ValueError("median window size must be a positive odd integer")
    r = size // 2
    H, W, C = a.shape
    p = np.pad(a, ((r, r), (r, r), (0, 0)), mode=mode)
    out = np.empty(a.shape, dtype=np.float32)
    rows = max(1, int(max_block_elems // (size * size * W * C)))
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        stack = np.empty((size * size, y1 - y0, W, C), dtype=np.float32)
        idx = 0
        for i in range(size):
            for j in range(size):
                stack[idx] = p[y0 + i:y1 + i, j:j + W]
                idx += 1
        out[y0:y1] = np.median(stack, axis=0)
    return out
