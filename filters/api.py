"""The shared [0,1] interface from the project plan:

    denoise(image_float_0_1, config) -> image_float_0_1

config = {"method": "bilateral", **params}  (params as stored in params.yaml, 0..255-scale units).
The methods themselves work on the 0..255 scale (see filters/classical.py); this wrapper converts,
checks shape/finite values and clips explicitly. It never modifies its input.
"""
from __future__ import annotations

import numpy as np

from . import get_filter

# Units of every tuned parameter (the methods work on the 0..255 scale internally; this wrapper takes [0,1] images).
PARAMETER_UNITS = {
    "gaussian": {"sigma": "pixels (kernel std)", "truncate": "kernel radius in sigmas"},
    "bilateral": {"sigma_s": "pixels (spatial std)", "sigma_r": "intensity levels on the 0..255 scale (range std)",
                  "k": "window radius in sigma_s units", "max_radius": "pixels"},
    "nlm": {"patch_size": "pixels", "patch_distance": "pixels (search half-window)",
            "h": "intensity levels on the 0..255 scale (filtering strength)", "sigma_255": "0..255 scale (0 = unused)"},
    "median": {"size": "pixels (odd window)  [optional method]"},
    "perona_malik": {"K": "intensity levels 0..255", "num_iter": "iterations", "lam": "time step (<= 0.25)",
                     "conductance": "'rational' or 'exp'  [optional method]"},
}


def denoise(image_float_0_1, config):
    x = np.asarray(image_float_0_1, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("expected a 2-D grayscale image in [0,1]")
    cfg = dict(config)
    method = cfg.pop("method")
    y = get_filter(method)(x * 255.0, **cfg) / 255.0
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise RuntimeError(f"{method} produced an invalid output")
    return y
