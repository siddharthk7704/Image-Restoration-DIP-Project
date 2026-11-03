#!/usr/bin/env python
"""Noise pipeline + manifest (Person A).

  gen    deterministic noisy images for every clean image and sigma, plus a manifest CSV
  check  inspect an EXISTING noisy folder (e.g. the BSD68_noisy sets you already have)

Noisy-image rule (identical to training):  noisy = clip(round(clean + N(0, sigma^2)), 0, 255)  [0..255 scale]
Seed per image and sigma = crc32("<image_id>|<sigma>") + --seed, so every run reproduces the same file.
The manifest records the seed and the PRE-clipping noise mean/std and the fraction of clipped pixels.

  python make_noisy.py gen   --clean-dir data/BSD68 --out-dir data/noisy/BSD68 --sigmas 15 25 50
  python make_noisy.py check --clean-dir data/BSD68 --noisy-dir data/BSD68_noisy --sigma 25

If you regenerate noisy sets that you already tuned on, rerun tune.py (the realizations changed).
"""
from __future__ import annotations

import argparse, csv, json, math, zlib
from pathlib import Path

import numpy as np
from PIL import Image

from common import list_images, load_gray_u8, noisy_rule_np, psnr01


def seed_for(base, image_id, sigma):
    return (base + zlib.crc32(f"{image_id}|{sigma}".encode())) % (2 ** 31 - 1)


def gen(a):
    clean = list_images(a.clean_dir)
    if not clean:
        raise SystemExit(f"no images in {a.clean_dir}")
    calib = set()
    if a.split_file and Path(a.split_file).exists():
        calib = set(json.loads(Path(a.split_file).read_text()).get("tune_images", []))
    rows = []
    for s in a.sigmas:
        out = Path(a.out_dir) / f"sigma{s}"
        out.mkdir(parents=True, exist_ok=True)
        for p in clean:
            c = load_gray_u8(p)
            seed = seed_for(a.seed, p.name, s)
            rng = np.random.RandomState(seed)
            noise = rng.randn(*c.shape).astype(np.float32) * np.float32(s)   # pre-clipping noise
            raw = c.astype(np.float32) + noise
            noisy = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
            Image.fromarray(noisy).save(out / p.name)
            rows.append(dict(image_id=p.name, split="calibration" if p.name in calib else "test",
                             preprocessing="grayscale uint8; noisy=clip(round(clean+N(0,sigma^2)),0,255)",
                             sigma_255=s, seed=seed, noise_mean_preclip=f"{noise.mean():.4f}",
                             noise_std_preclip=f"{noise.std():.4f}",
                             clipped_fraction=f"{float(((raw < 0) | (raw > 255)).mean()):.5f}",
                             noisy_psnr=f"{psnr01(c / 255., noisy / 255.):.3f}",
                             noisy_path=str(out / p.name)))
    man = Path(a.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for s in a.sigmas:
        r = [x for x in rows if x["sigma_255"] == s]
        print(f"sigma {s}: {len(r)} images | pre-clip noise std {np.mean([float(x['noise_std_preclip']) for x in r]):.2f} "
              f"(target {s}) | mean noisy PSNR {np.mean([float(x['noisy_psnr']) for x in r]):.2f} dB "
              f"(theory without clipping {20 * math.log10(255 / s):.2f})")
    print(f"manifest: {man}")


def check(a):
    clean = {p.stem: p for p in list_images(a.clean_dir)}
    noisy = {p.stem: p for p in list_images(a.noisy_dir)}
    stems = sorted(set(clean) & set(noisy))
    if not stems:
        raise SystemExit("no matching names between the two folders")
    means, stds, sat, ps = [], [], [], []
    for st in stems:
        c, n = load_gray_u8(clean[st]).astype(np.float32), load_gray_u8(noisy[st]).astype(np.float32)
        if c.shape != n.shape:
            raise SystemExit(f"shape mismatch for {st}")
        d = n - c
        means.append(d.mean()); stds.append(d.std()); sat.append(float(((n == 0) | (n == 255)).mean()))
        ps.append(psnr01(c / 255., n / 255.))
    print(f"{len(stems)} pairs | (noisy - clean): mean {np.mean(means):+.3f}, std {np.mean(stds):.2f} "
          f"| saturated pixels {100 * np.mean(sat):.2f}% | noisy PSNR {np.mean(ps):.2f} dB")
    if a.sigma:
        print(f"theory without clipping: std {a.sigma}, PSNR {20 * math.log10(255 / a.sigma):.2f} dB. "
              f"After clipping the std is a bit smaller and PSNR a bit higher (more so at large sigma).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--clean-dir", required=True); g.add_argument("--out-dir", required=True)
    g.add_argument("--sigmas", type=int, nargs="+", default=[15, 25, 50])
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--manifest", default="data/manifest.csv")
    g.add_argument("--split-file", default="results/tuning/tune_split.json")
    c = sub.add_parser("check")
    c.add_argument("--clean-dir", required=True); c.add_argument("--noisy-dir", required=True)
    c.add_argument("--sigma", type=int, default=0)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else check(a)


if __name__ == "__main__":
    main()
