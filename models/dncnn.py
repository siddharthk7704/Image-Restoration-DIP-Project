"""DnCNN-style residual denoiser (Zhang et al., TIP 2017; https://arxiv.org/abs/1608.03981).

Conv+ReLU, then (depth-2) x [Conv + BatchNorm + ReLU], then Conv. The network predicts the
NOISE residual and the restored image is  y - predicted_noise  (residual learning).
With 3x3 convolutions the receptive field is (2*depth+1) x (2*depth+1), e.g. 35x35 for depth 17.

This is a compact re-implementation under our protocol, not a claim of canonical reproduction:
record depth, width and parameter count in the report.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    pad_multiple = 1  # any input size works

    def __init__(self, in_ch=1, depth=17, width=64, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, 3, padding=1, bias=False))
            if bn:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(width, in_ch, 3, padding=1, bias=False))
        self.body = nn.Sequential(*layers)
        self.depth, self.width = depth, width
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def predict_noise(self, x):
        return self.body(x)

    def forward(self, x):
        return x - self.body(x)  # restored = noisy - predicted noise

    @property
    def description(self):
        return dict(arch="DnCNN-style residual", depth=self.depth, width=self.width, kernel=3, padding=1,
                    normalization="BatchNorm" if any(isinstance(m, nn.BatchNorm2d) for m in self.modules()) else "none",
                    target="noise residual (restored = noisy - predicted noise)", receptive_field=self.receptive_field)

    @property
    def receptive_field(self):
        return 2 * self.depth + 1


def build_dncnn(in_ch=1, depth=17, width=64, bn=True):
    return DnCNN(in_ch=in_ch, depth=depth, width=width, bn=bn)
