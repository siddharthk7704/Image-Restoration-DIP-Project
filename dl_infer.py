#!/usr/bin/env python
"""Inference for the trained blind denoisers + the shared learned-model interface.

    from dl_infer import load_model, denoise_dl
    model = load_model("runs/dncnn_P_s0/best.pth")   # for batch use; denoise_dl(img, model) also works
    restored = denoise_dl(noisy_float_0_1, {"ckpt": "runs/dncnn_P_s0/best.pth"})   # blind: no sigma argument
    # (passing an already loaded model, as below, is also accepted)

CLI: denoise a folder and (optionally) report PSNR against the clean folder. Outputs are saved as
8-bit PNGs (rounded, clipped) in  <out-dir>/<method-name>/BSD68/sigma<S>/  to match the folder contract.

  python dl_infer.py --ckpt runs/dncnn_P_s0/best.pth --noisy-dir data/noisy/BSD68/sigma25 --sigma 25 \
      --method-name dncnn_P_s0 --out-dir outputs --clean-dir data/BSD68 --only-heldout results/tuning/tune_split.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import list_images, load_gray_u8, psnr01, quantize01, save_gray01
from models import build_model


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = build_model(ck["arch"], **ck["arch_cfg"])
    net.load_state_dict(ck["model"])
    net.eval().to(device)
    net.ckpt_info = {"path": str(ckpt_path), "step": ck.get("step"), "arch": ck["arch"], "cfg": ck["arch_cfg"]}
    return net


_CACHE = {}


def _resolve(model_config, device):
    """model_config: {"ckpt": path, "device": "cpu"|"cuda"}  (loaded once and cached)  OR an already loaded nn.Module."""
    if isinstance(model_config, torch.nn.Module):
        return model_config
    dev = device or model_config.get("device", "cpu")
    key = (str(model_config["ckpt"]), dev)
    if key not in _CACHE:
        _CACHE[key] = load_model(model_config["ckpt"], dev)
    return _CACHE[key]


@torch.no_grad()
def denoise_dl(image_float_0_1, model_config, device=None):
    """Shared contract for learned models (project plan):  denoise_dl(image_float_0_1, model_config) -> image_float_0_1

    Input and output: float32 (H, W) in [0,1], explicit clipping, input never modified. The model is BLIND: there is
    no sigma argument, so the wrapper cannot use the noise level."""
    model = _resolve(model_config, device)
    x = np.asarray(image_float_0_1, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("expected a 2-D grayscale image in [0,1]")
    device = device or next(model.parameters()).device
    t = torch.from_numpy(x)[None, None].to(device)
    h, w = t.shape[-2:]
    m = getattr(model, "pad_multiple", 1)
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    y = model(t)[..., :h, :w][0, 0].float().cpu().numpy()
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(y).all():
        raise RuntimeError("non-finite output")
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--noisy-dir", required=True)
    ap.add_argument("--sigma", type=int, required=True, help="only used to name the output folder")
    ap.add_argument("--method-name", required=True); ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--dataset", default="BSD68"); ap.add_argument("--clean-dir", default="")
    ap.add_argument("--only-heldout", default="", help="tune_split.json: process only its heldout (test) images")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model = load_model(a.ckpt, a.device)
    files = list_images(a.noisy_dir)
    if a.only_heldout:
        keep = set(json.loads(Path(a.only_heldout).read_text())["heldout_images"])
        files = [p for p in files if p.name in keep]
    out = Path(a.out_dir) / a.method_name / a.dataset / f"sigma{a.sigma}"
    out.mkdir(parents=True, exist_ok=True)
    clean = {p.name: p for p in list_images(a.clean_dir)} if a.clean_dir else {}
    ps = []
    for p in files:
        y = denoise_dl(load_gray_u8(p).astype(np.float32) / 255., model)
        save_gray01(out / p.name, y)
        if p.name in clean:
            ps.append(psnr01(load_gray_u8(clean[p.name]).astype(np.float32) / 255., quantize01(y)))
    print(f"{len(files)} images -> {out}" + (f" | mean PSNR {np.mean(ps):.3f} dB" if ps else ""))


if __name__ == "__main__":
    main()
