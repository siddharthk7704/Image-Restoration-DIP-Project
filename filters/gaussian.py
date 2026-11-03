"""
Gaussian denoising baseline for the photograph-restoration project.

Input: grayscale image as a NumPy array, float in [0, 1] (or uint8 [0, 255]).
Output: float32 grayscale image in [0, 1].

Install:
    python -m pip install numpy scipy pillow

The parameter sigma_blur is the Gaussian kernel standard deviation in PIXELS.
It is NOT the AWGN noise sigma used to corrupt the image.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def denoise_gaussian(image: np.ndarray, sigma_blur: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to a grayscale image."""
    if sigma_blur < 0:
        raise ValueError("sigma_blur must be >= 0")

    x = np.asarray(image)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expected grayscale HxW image, got shape {x.shape}")

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Image contains NaN or Inf")
        if x.min() < -1e-6 or x.max() > 1.000001:
            raise ValueError("Float image must be normalized to [0, 1]")

    if sigma_blur == 0:
        return x.copy()

    # 'reflect' boundary handling; truncate=4 gives a finite kernel radius.
    y = gaussian_filter(x, sigma=sigma_blur, mode="reflect", truncate=4.0)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser(description="Gaussian-filter a grayscale image.")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--sigma-blur", type=float, default=1.0,
                        help="Gaussian kernel sigma in pixels (default: 1.0)")
    args = parser.parse_args()

    im = np.asarray(Image.open(args.input).convert("L"))
    result = denoise_gaussian(im, args.sigma_blur)
    Image.fromarray(np.round(result * 255).astype(np.uint8)).save(args.output)
    print(f"Saved {args.output}; sigma_blur={args.sigma_blur}")
