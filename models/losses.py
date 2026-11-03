"""Losses: pixel L1 and a Sobel gradient L1 (fixed convolution kernels, differentiable)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradLoss(nn.Module):
    """mean over pixels of |Sx(pred) - Sx(target)| + |Sy(pred) - Sy(target)|.

    Sobel kernels are divided by 8 so the gradient is on the same scale as the intensity
    difference between neighbouring pixels. Borders use replicate padding.
    """

    def __init__(self, channels=1):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        ky = kx.t().contiguous()
        w = torch.stack([kx, ky])[:, None]            # (2, 1, 3, 3)
        self.register_buffer("weight", w.repeat(channels, 1, 1, 1))  # (2C, 1, 3, 3)
        self.channels = channels

    def gradients(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, self.weight, groups=self.channels)  # (B, 2C, H, W)

    def forward(self, pred, target):
        diff = (self.gradients(pred) - self.gradients(target)).abs()
        b, _, h, w = diff.shape
        return diff.view(b, self.channels, 2, h, w).sum(dim=2).mean()
