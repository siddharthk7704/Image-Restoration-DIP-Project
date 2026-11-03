"""Small shared helpers (Person A). Scale convention: arrays are float in [0,1] unless named *_u8."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from filters.core import correlate2d, gaussian_filter

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def load_gray_u8(path):
    """Grayscale uint8 (H, W)."""
    return np.asarray(Image.open(path).convert("L"))


def save_gray01(path, img01):
    """Save a [0,1] float image as 8-bit PNG (rounded, clipped)."""
    Image.fromarray(np.rint(np.clip(img01, 0, 1) * 255).astype(np.uint8)).save(path)


def noisy_rule_np(clean_u8, sigma_255, rng):
    """THE noisy-image rule used everywhere (training and test):
    noisy = clip(round(clean + N(0, sigma^2)), 0, 255)  on the 0..255 scale. Returns uint8."""
    x = clean_u8.astype(np.float32)
    n = rng.randn(*x.shape).astype(np.float32) * np.float32(sigma_255)
    return np.clip(np.rint(x + n), 0, 255).astype(np.uint8)


def quantize01(x):
    """Round a [0,1] image to 8-bit levels (output rule: metrics are computed on 8-bit outputs)."""
    return np.rint(np.clip(x, 0, 1) * 255.0) / 255.0


def psnr01(ref, test, mask=None):
    """PSNR on [0,1] images. With a mask: PSNR over the masked pixels; NaN (not an exception, not a warning)
    if the mask selects no pixel."""
    d = (ref.astype(np.float64) - test.astype(np.float64)) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        mse = float(d[mask].mean())
    else:
        mse = float(d.mean())
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def masked_sq_error(ref, test, mask):
    """(sum of squared errors, number of pixels) inside a boolean mask. Both are 0 for an empty mask, so results
    from many images can be POOLED without producing NaN when a single image has no pixel of that region."""
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0
    d = (ref.astype(np.float64) - test.astype(np.float64)) ** 2
    return float(d[mask].sum()), n


def pooled_psnr(se_n_pairs):
    """PSNR from pooled (sum_sq_err, n_pixels) pairs; NaN only if the region is empty in ALL images."""
    se = sum(p[0] for p in se_n_pairs)
    n = sum(p[1] for p in se_n_pairs)
    if n == 0:
        return float("nan")
    mse = se / n
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def gradient_magnitude(img01, sigma=1.0):
    """Gaussian-derivative style gradient: Gaussian smoothing, then central differences."""
    g = gaussian_filter(img01[:, :, None].astype(np.float32), sigma)[:, :, 0]
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def local_std(img01, win=7):
    a = img01[:, :, None].astype(np.float32)
    k = np.ones((win, win), dtype=np.float32) / (win * win)
    m = correlate2d(a, k)[:, :, 0]
    m2 = correlate2d(a * a, k)[:, :, 0]
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def region_masks(clean_list, smooth_pct=40, edge_pct=90, win=7):
    """Smooth / edge masks for a list of clean [0,1] images, thresholds from pooled percentiles.

    smooth = gradient and local std both <= their smooth_pct percentile
    edge   = gradient >= its edge_pct percentile
    (Person B's final masks follow the same rule with thresholds fixed on the calibration images;
     this helper is for validation-time monitoring only.)
    """
    grads = [gradient_magnitude(c) for c in clean_list]
    stds = [local_std(c, win) for c in clean_list]
    gs = np.concatenate([g.ravel() for g in grads])
    ss = np.concatenate([s.ravel() for s in stds])
    g_lo, g_hi = np.percentile(gs, smooth_pct), np.percentile(gs, edge_pct)
    s_lo = np.percentile(ss, smooth_pct)
    return [((g <= g_lo) & (s <= s_lo), g >= g_hi) for g, s in zip(grads, stds)]
