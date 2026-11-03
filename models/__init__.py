"""Architecture registry shared by both team members.

Person A: dncnn (this file registers it).
Person B: add `models/unet.py` that defines `build_unet(**kwargs)` returning an nn.Module with a
`pad_multiple` attribute (e.g. 8) and `forward(noisy) -> restored`; it is picked up automatically.
Both use the SAME train_dl.py, so the training protocol is identical by construction.
"""
from .dncnn import build_dncnn

ARCHS = {"dncnn": build_dncnn}

try:  # Person B's network, if present
    from .unet import build_unet  # type: ignore

    ARCHS["unet"] = build_unet
except ImportError:
    pass


def build_model(arch, **kwargs):
    if arch not in ARCHS:
        raise KeyError(f"unknown arch '{arch}'. Available: {sorted(ARCHS)}")
    return ARCHS[arch](**kwargs)
