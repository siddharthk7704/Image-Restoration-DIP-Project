#!/usr/bin/env python
"""Runtime benchmark for ALL methods under one set of rules (Person A).

Rules (state them in the report):
  * every method is timed on the SAME CPU; learned models are additionally timed on GPU if available
  * warm-up calls are discarded; each image is timed `--repeats` times; we report the median
  * timing covers the denoise call only: no file reading, no model loading, no metric computation
  * GPU timings synchronize before and after each call
  * hardware / software versions are written into every row

  python runtime_bench.py --noisy-dir data/noisy/BSD68/sigma25 --sigma 25 --params params.yaml \
      --methods gaussian bilateral nlm perona_malik median \
      --dl dncnn_P=runs/dncnn_P_s0/best.pth dncnn_G=runs/dncnn_G0.5_s0/best.pth --out results/runtime.csv
"""
from __future__ import annotations

import argparse, csv, os, platform, time
from pathlib import Path

import numpy as np
import torch
import yaml

from common import list_images, load_gray_u8
from dl_infer import denoise_dl, load_model
from filters import METHODS


def hardware():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return dict(cpu=platform.processor() or platform.machine(), cpu_count=os.cpu_count(),
                torch_threads=torch.get_num_threads(), gpu=gpu, torch=torch.__version__,
                numpy=np.__version__, python=platform.python_version(), os=platform.system())


def time_calls(fn, imgs, warmup, repeats, cuda):
    for im in imgs[:max(1, warmup)]:
        fn(im)
    ts = []
    for im in imgs:
        for _ in range(repeats):
            if cuda: torch.cuda.synchronize()
            t0 = time.perf_counter(); fn(im)
            if cuda: torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return np.array(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noisy-dir", required=True); ap.add_argument("--sigma", type=int, required=True)
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--methods", nargs="*", default=[]); ap.add_argument("--dl", nargs="*", default=[],
                    help="name=checkpoint pairs")
    ap.add_argument("--n-images", type=int, default=10); ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--out", default="results/runtime.csv")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)

    files = list_images(a.noisy_dir)[:a.n_images]
    imgs01 = [load_gray_u8(p).astype(np.float32) / 255. for p in files]
    imgs255 = [x * 255. for x in imgs01]
    mpix = float(np.mean([x.size for x in imgs01])) / 1e6
    params = yaml.safe_load(open(a.params)) if Path(a.params).exists() else {}
    hw = hardware()
    rows = []

    def add(name, device, ts):
        rows.append(dict(method=name, device=device, sigma_255=a.sigma, n_images=len(imgs01), repeats=a.repeats,
                         megapixels=f"{mpix:.4f}", median_s=f"{np.median(ts):.5f}", p25_s=f"{np.percentile(ts, 25):.5f}",
                         p75_s=f"{np.percentile(ts, 75):.5f}", median_s_per_mpix=f"{np.median(ts) / mpix:.4f}",
                         includes_model_load="no", **hw))
        print(f"{name:14s} {device:4s} median {np.median(ts) * 1000:9.1f} ms/image")

    for m in a.methods:
        p = params.get(m, {}).get(f"sigma{a.sigma}")
        if p is None:
            print(f"skip {m}: no params for sigma{a.sigma} in {a.params}"); continue
        fn = METHODS[m]
        add(m, "cpu", time_calls(lambda im, fn=fn, p=p: fn(im, **p), imgs255, a.warmup, a.repeats, False))
    for spec in a.dl:
        name, ckpt = spec.split("=", 1)
        model = load_model(ckpt, "cpu")
        with torch.no_grad():
            add(name, "cpu", time_calls(lambda im: denoise_dl(im, model, "cpu"), imgs01, a.warmup, a.repeats, False))
            if torch.cuda.is_available():
                model = load_model(ckpt, "cuda")
                add(name, "gpu", time_calls(lambda im: denoise_dl(im, model, "cuda"), imgs01, a.warmup, a.repeats, True))
    if rows:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("wrote", out)


if __name__ == "__main__":
    main()
