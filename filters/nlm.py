"""Non-Local Means (library-based, LABELLED as such in the report).

Wraps scikit-image `denoise_nl_means`. Same conventions as the other filters in this package:
input (H, W) or (H, W, C) on the 0..255 scale, output float32 on the same scale.

Parameters (all on the 0..255 scale where a scale applies):
  patch_size      side of the comparison patch (odd)
  patch_distance  half-size of the search window (pixels)
  h               filtering strength: larger h = stronger smoothing (typically ~0.5 to 1.2 x noise sigma)
  sigma_255       noise std passed to skimage for bias correction; 0 = ignore (default)
  fast_mode       skimage's fast variant (recommended)
"""
from __future__ import annotations

import numpy as np


def denoise_nlm(img, patch_size=5, patch_distance=6, h=20.0, sigma_255=0.0, fast_mode=True):
    from skimage.restoration import denoise_nl_means  # library implementation (version: see requirements)

    a = np.asarray(img, dtype=np.float32) / 255.0
    channel_axis = -1 if a.ndim == 3 else None
    out = denoise_nl_means(a, patch_size=int(patch_size), patch_distance=int(patch_distance),
                           h=float(h) / 255.0, sigma=float(sigma_255) / 255.0, fast_mode=bool(fast_mode),
                           channel_axis=channel_axis)
    return np.clip(out * 255.0, 0, 255).astype(np.float32)
