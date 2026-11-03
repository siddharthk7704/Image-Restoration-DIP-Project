"""Classical denoisers: Gaussian and bilateral (core); median and Perona-Malik are OPTIONAL extras of the study.

Public interface (this is what tune.py, run_all.py and B's code call):

    denoise_<method>(img, **params) -> float32 array, same shape as img, range 0..255

* img: (H, W) or (H, W, C), values on the 0..255 scale (uint8 or float).
* The output is NOT rounded or clipped to uint8 here (all four filters are convex
  combinations of input values, so they cannot leave the input range). Rounding to
  8-bit before computing metrics is a protocol decision made in the evaluation code.
* Multi-channel images: Gaussian and median act per channel; bilateral and
  Perona-Malik use a joint (vector) distance / gradient across channels, as in the
  original RGB scripts these were adapted from.
"""
from __future__ import annotations

import math

import numpy as np

from .core import from_hwc, gaussian_filter, median_filter, to_hwc


def denoise_gaussian(img, sigma=1.0, truncate=3.0):
    a, sq = to_hwc(img)
    return from_hwc(gaussian_filter(a, sigma, truncate), sq)


def denoise_median(img, size=3):
    a, sq = to_hwc(img)
    return from_hwc(median_filter(a, int(size)), sq)


def denoise_bilateral(img, sigma_s=2.0, sigma_r=25.0, k=3.0, max_radius=12):
    """Bilateral filter: circular spatial Gaussian x range Gaussian, normalized per pixel.

    sigma_s: spatial sigma (pixels). sigma_r: range sigma (intensity levels, 0..255 scale).
    Window radius = min(ceil(k * sigma_s), max_radius).
    Memory is O(H*W*C): we loop over window offsets and accumulate, instead of building
    an (H, W, N, C) neighbour tensor.
    """
    if sigma_s <= 0 or sigma_r <= 0:
        raise ValueError("sigma_s and sigma_r must be positive")
    a, sq = to_hwc(img)
    H, W, _ = a.shape
    radius = int(min(math.ceil(k * sigma_s), max_radius))
    radius = max(radius, 1)
    padded = np.pad(a, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    inv_2sr2 = np.float32(1.0 / (2.0 * sigma_r ** 2))

    num = np.zeros_like(a)
    den = np.zeros((H, W, 1), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d2 = dy * dy + dx * dx
            if d2 > radius * radius:
                continue
            w_s = np.float32(math.exp(-d2 / (2.0 * sigma_s ** 2)))
            nb = padded[radius + dy:radius + dy + H, radius + dx:radius + dx + W]
            diff = nb - a
            r2 = np.sum(diff * diff, axis=2, keepdims=True)
            w = w_s * np.exp(-r2 * inv_2sr2)
            num += w * nb
            den += w
    return from_hwc(num / den, sq)


def denoise_perona_malik(img, num_iter=20, lam=0.2, K=20.0, conductance="rational"):
    """Perona-Malik anisotropic diffusion, explicit 4-neighbour scheme.

    I <- I + lam * sum_{d in N,S,W,E} c(|grad_d|) * grad_d
    conductance: "rational" -> c = 1 / (1 + (g/K)^2)   ;   "exp" -> c = exp(-(g/K)^2)
    lam must be <= 0.25 for stability. Borders use edge padding (zero flux), so the
    mean intensity is preserved. For multi-channel input, |grad|^2 is summed over channels.
    """
    if not 0 < lam <= 0.25:
        raise ValueError("lam must satisfy 0 < lam <= 0.25 for the 4-neighbour explicit scheme")
    if K <= 0:
        raise ValueError("K must be positive")
    if conductance not in ("rational", "exp"):
        raise ValueError("conductance must be 'rational' or 'exp'")
    a, sq = to_hwc(img)
    I = a.astype(np.float64)
    K2 = float(K) ** 2
    for _ in range(int(num_iter)):
        p = np.pad(I, ((1, 1), (1, 1), (0, 0)), mode="edge")
        flux = np.zeros_like(I)
        for d in (p[:-2, 1:-1] - I, p[2:, 1:-1] - I, p[1:-1, :-2] - I, p[1:-1, 2:] - I):
            g2 = np.sum(d * d, axis=2, keepdims=True) / K2
            c = 1.0 / (1.0 + g2) if conductance == "rational" else np.exp(-g2)
            flux += c * d
        I = I + lam * flux
    return from_hwc(I.astype(np.float32), sq)
