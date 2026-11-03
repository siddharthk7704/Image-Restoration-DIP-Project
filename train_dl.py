#!/usr/bin/env python
"""Train a blind residual denoiser (DnCNN by default) on BSD400 under the shared protocol.

SAME script for Person A (dncnn) and Person B (unet): protocol identical by construction.

Protocol (see the project plan, "Frozen protocol"):
  * BSD400 grayscale; fixed split (380 train / 20 validation) saved to --split-file and SHARED by both people
  * 64x64 patches, flips + 90-degree rotations, online AWGN, sigma ~ U[15, 50] per patch (blind model)
  * noisy rule: clip(round(clean + noise)) to 8-bit levels, identical to the saved test images
  * loss  L = L1(pred, clean) + lam * SobelL1(pred, clean)     lam = 0 -> variant P, lam > 0 -> variant G
  * checkpoint choice: best validation PSNR averaged over sigma 15, 25, 50 (validation images only)
  * P and G with the same --seed see identical init, data order and noise draws (only the loss differs)

Validation monitoring (history.csv)
  psnr15/25/50, psnr_mean : mean over the validation images of the per-image PSNR (images are the unit)
  edge_psnr_mean, smooth_psnr_mean : region PSNR POOLED over all validation pixels of that region (sum of squared
      errors / number of pixels). Pooling cannot produce NaN just because one image has no edge (or no smooth)
      pixels, which is what made the earlier per-image version return NaN. A region that is empty in EVERY image
      is logged as nan and ignored in the averages, with a warning.

Typical commands
  python train_dl.py --time-test                          # size the run first
  python train_dl.py --overfit-check                      # pipeline check, then discard
  python train_dl.py --lam 0   --seed 0                   # P, seed 0
  python train_dl.py --lam 0.5 --seed 0                   # G, seed 0
  python train_dl.py --lam 0.5 --seed 0 --resume runs/dncnn_G0.5_s0/last.pth
Kaggle: add --amp, point --train-dir to the BSD400 folder, --out to /kaggle/working/runs.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common import (list_images, load_gray_u8, masked_sq_error, noisy_rule_np, pooled_psnr, psnr01,
                    quantize01, region_masks)
from filters import denoise_gaussian
from models import build_model
from models.losses import SobelGradLoss

MAX_BAD_STEPS = 20  # consecutive non-finite training steps tolerated before aborting


# --------------------------------------------------------------------------- data
def make_or_load_split(files, n_val, seed, split_file):
    names = [p.name for p in files]
    split_file = Path(split_file)
    if split_file.exists():
        s = json.loads(split_file.read_text())
        if sorted(s["train"] + s["val"]) == sorted(names):
            return s["train"], s["val"]
        raise SystemExit(f"{split_file} does not match the images in --train-dir; delete it or fix the folder.")
    rng = np.random.RandomState(seed)
    val_idx = set(rng.choice(len(names), n_val, replace=False).tolist())
    s = {"seed": seed, "val": [n for i, n in enumerate(names) if i in val_idx],
         "train": [n for i, n in enumerate(names) if i not in val_idx]}
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(s, indent=2))
    return s["train"], s["val"]


class PatchSampler:
    """Random patches (flips + rot90) from images held in RAM as uint8. The RNG is local => reproducible."""

    def __init__(self, images, patch, seed):
        self.images, self.s = images, patch
        self.rng = np.random.RandomState(seed)

    def batch(self, n):
        out = np.empty((n, 1, self.s, self.s), np.uint8)
        for i in range(n):
            a = self.images[self.rng.randint(len(self.images))]
            h, w = a.shape
            y, x = self.rng.randint(0, h - self.s + 1), self.rng.randint(0, w - self.s + 1)
            p = a[y:y + self.s, x:x + self.s]
            if self.rng.rand() < .5:
                p = p[:, ::-1]
            if self.rng.rand() < .5:
                p = p[::-1]
            out[i, 0] = np.rot90(p, self.rng.randint(4))
        return torch.from_numpy(out)


def add_noise(clean_u8, sigma_min, sigma_max, gen):
    """clean uint8 (N,1,s,s) on device -> (noisy, clean) in [0,1]; per-patch sigma; noisy rule applied."""
    x = clean_u8.float()
    n = x.shape[0]
    sig = sigma_min + (sigma_max - sigma_min) * torch.rand(n, 1, 1, 1, device=x.device, generator=gen)
    noisy = (x + torch.randn(x.shape, device=x.device, generator=gen) * sig).clamp_(0, 255).round_()
    return noisy / 255., x / 255.


# --------------------------------------------------------------------------- validation
class Validator:
    SIGMAS = (15, 25, 50)

    def __init__(self, val_images, seed=0):
        self.clean01 = [v.astype(np.float32) / 255. for v in val_images]
        self.masks = region_masks(self.clean01)  # monitoring only (Person B's masks are the final ones)
        n_edge = [int(e.sum()) for _, e in self.masks]
        n_smooth = [int(s.sum()) for s, _ in self.masks]
        self.empty_edge = sum(1 for n in n_edge if n == 0)
        self.empty_smooth = sum(1 for n in n_smooth if n == 0)
        print(f"validation masks: {sum(n_edge)} edge pixels, {sum(n_smooth)} smooth pixels in total; "
              f"{self.empty_edge} image(s) without edge pixels, {self.empty_smooth} without smooth pixels "
              f"(handled by pooling)")
        self.noisy01 = {}
        for s in self.SIGMAS:
            self.noisy01[s] = []
            for i, v in enumerate(val_images):
                rng = np.random.RandomState(10_000 + i * 101 + s)  # fixed: identical for every run
                self.noisy01[s].append(noisy_rule_np(v, s, rng).astype(np.float32) / 255.)

    def baselines(self, params_path):
        """Reference scores on the validation images: noisy input, and the calibrated Gaussian if params.yaml exists.
        Acceptance test: the trained network must beat both."""
        gp = None
        if Path(params_path).exists():
            gp = (yaml.safe_load(open(params_path)) or {}).get("gaussian", {})
        out = {}
        for s in self.SIGMAS:
            noisy = float(np.mean([psnr01(c, n) for c, n in zip(self.clean01, self.noisy01[s])]))
            gauss = None
            if gp and f"sigma{s}" in gp:
                g = [psnr01(c, quantize01(denoise_gaussian(n * 255.0, **gp[f"sigma{s}"]) / 255.0))
                     for c, n in zip(self.clean01, self.noisy01[s])]
                gauss = float(np.mean(g))
            out[s] = dict(noisy=noisy, gaussian=gauss)
        return out

    @torch.no_grad()
    def run(self, net, dev, amp):
        """Returns {sigma: (mean per-image PSNR, pooled edge PSNR, pooled smooth PSNR)}; region values may be nan
        only if that region is empty in every validation image."""
        net.eval()
        res = {}
        m = getattr(net, "pad_multiple", 1)
        for s in self.SIGMAS:
            per_image, edge_parts, smooth_parts = [], [], []
            for c, n, (sm, ed) in zip(self.clean01, self.noisy01[s], self.masks):
                x = torch.from_numpy(n)[None, None].to(dev)
                h, w = x.shape[-2:]
                ph, pw = (-h) % m, (-w) % m
                if ph or pw:
                    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    y = net(x)[..., :h, :w]
                y = quantize01(y[0, 0].float().cpu().numpy())
                per_image.append(psnr01(c, y))
                edge_parts.append(masked_sq_error(c, y, ed))
                smooth_parts.append(masked_sq_error(c, y, sm))
            res[s] = (float(np.mean(per_image)), pooled_psnr(edge_parts), pooled_psnr(smooth_parts))
        net.train()
        return res


def mean_over_sigmas(res, idx):
    """Mean over sigma of entry `idx` of the results, ignoring nan; nan only if every sigma is nan."""
    vals = [res[s][idx] for s in Validator.SIGMAS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(vals)) if not all(np.isnan(vals)) else float("nan")


# --------------------------------------------------------------------------- misc
def cosine_lr(step, total, lr, lr_min):
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * min(step, total) / total))


def run_name(a):
    v = "P" if a.lam == 0 else f"G{a.lam:g}"
    return a.name or f"{a.arch}_{v}_s{a.seed}"


def fmt(x, nd=4):
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", default="data/BSD400")
    ap.add_argument("--split-file", default="data/bsd400_split.json")
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default="")
    ap.add_argument("--arch", default="dncnn")
    ap.add_argument("--depth", type=int, default=17)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--sigma-min", type=float, default=15.)
    ap.add_argument("--sigma-max", type=float, default=50.)
    ap.add_argument("--lam", type=float, default=0.0, help="weight of the Sobel gradient loss (0 = variant P)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--params", default="params.yaml", help="classical settings (for the Gaussian reference score)")
    ap.add_argument("--time-test", action="store_true")
    ap.add_argument("--overfit-check", action="store_true")
    a = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = a.amp and dev.type == "cuda"
    torch.backends.cudnn.benchmark = True

    files = list_images(a.train_dir)
    if len(files) < 50:
        raise SystemExit(f"only {len(files)} images in {a.train_dir}; point --train-dir to BSD400")
    train_names, val_names = make_or_load_split(files, a.n_val, 0, a.split_file)  # split seed fixed, shared
    byname = {p.name: p for p in files}
    train_imgs = [load_gray_u8(byname[n]) for n in train_names]
    val_imgs = [load_gray_u8(byname[n]) for n in val_names]
    train_imgs = [t for t in train_imgs if min(t.shape) >= a.patch]

    arch_cfg = dict(in_ch=1, depth=a.depth, width=a.width) if a.arch == "dncnn" else dict(in_ch=1)
    torch.manual_seed(a.seed)                                   # init depends only on the seed
    net = build_model(a.arch, **arch_cfg).to(dev)
    nparams = sum(p.numel() for p in net.parameters())
    name = run_name(a)
    run_dir = Path(a.out) / name
    print(f"{name}: {a.arch} {nparams / 1e6:.3f}M params | train {len(train_imgs)} / val {len(val_imgs)} images | "
          f"device {dev.type} amp={amp} | lam={a.lam}")

    gen = torch.Generator(device=dev)
    gen.manual_seed(1000 + a.seed)                              # noise draws depend only on the seed
    sampler = PatchSampler(train_imgs, a.patch, a.seed)
    grad_loss = SobelGradLoss(1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(.9, .999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def step_fn(clean_u8, lr):
        """One optimizer step. Returns (l1_pix, l1_grad); (nan, nan) if the step was skipped as non-finite."""
        for g in opt.param_groups:
            g["lr"] = lr
        noisy, clean = add_noise(clean_u8.to(dev), a.sigma_min, a.sigma_max, gen)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            pred = net(noisy)
        pred = pred.float()
        l_pix = F.l1_loss(pred, clean)
        l_grad = grad_loss(pred, clean) if a.lam > 0 else torch.zeros((), device=dev)
        loss = l_pix + a.lam * l_grad
        if not torch.isfinite(loss):                            # skip instead of corrupting the weights
            opt.zero_grad(set_to_none=True)
            return float("nan"), float("nan")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if amp or torch.isfinite(gn):                           # with AMP the scaler skips inf grads itself
            scaler.step(opt)
        scaler.update()
        return float(l_pix.detach()), float(l_grad.detach())

    if a.time_test:
        for _ in range(3):
            step_fn(sampler.batch(a.batch), a.lr)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            step_fn(sampler.batch(a.batch), a.lr)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 20
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
        print(f"{dt:.3f} s/step, peak GPU {peak:.2f} GB -> {a.steps} steps ~ {dt * a.steps / 3600:.2f} h "
              f"(one run). 4 runs + pilots ~ {dt * a.steps * 6 / 3600:.1f} h")
        return

    if a.overfit_check:  # pipeline check: memorise 8 fixed patches, then discard
        fixed = sampler.batch(8)
        first = last = None
        for _ in range(200):
            gen.manual_seed(7)  # same noise each step -> pure memorisation
            lp, _ = step_fn(fixed, 1e-3)
            first = lp if first is None else first
            last = lp
        print(f"overfit check: L1 {first:.4f} -> {last:.4f} ({'OK' if last < 0.5 * first else 'NOT DECREASING: debug'})")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        **vars(a), "run_name": name, "params": nparams, "arch_cfg": arch_cfg,
        "model_description": getattr(net, "description", {}),
        "hardware": dict(device=dev.type, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "none",
                         torch=torch.__version__, python=platform.python_version(), os=platform.system())},
        indent=2, default=str))
    validator = Validator(val_imgs)
    base = validator.baselines(a.params)
    (run_dir / "baselines.json").write_text(json.dumps(base, indent=2))
    print("validation reference (the network must beat these):")
    for s, v in base.items():
        print(f"  sigma {s}: noisy {v['noisy']:.2f} dB" +
              (f" | calibrated Gaussian {v['gaussian']:.2f} dB" if v["gaussian"] else ""))

    hist = run_dir / "history.csv"
    step, best, elapsed_prev = 0, -1.0, 0.0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best, elapsed_prev = ck["step"], ck["best"], ck.get("elapsed", 0.0)
        sampler.rng.set_state(ck["np_rng"])
        gen.set_state(ck["gen_state"].cpu())
        print(f"resumed from step {step}")
    if not hist.exists():
        hist.write_text("step,l1_pix,l1_grad,lr,psnr15,psnr25,psnr50,psnr_mean,edge_psnr_mean,smooth_psnr_mean,sec_per_step\n")

    t_start = time.time()

    def save(path, model_only=False):
        d = dict(model=net.state_dict(), arch=a.arch, arch_cfg=arch_cfg, step=step, best=best, args=vars(a),
                 elapsed=elapsed_prev + (time.time() - t_start))
        if not model_only:
            d.update(optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                     np_rng=sampler.rng.get_state(), gen_state=gen.get_state())
        torch.save(d, path)

    net.train()
    acc_p = acc_g = 0.0
    n_acc = bad_total = bad_consec = 0
    t0 = time.time()
    warned_region = False
    while step < a.steps:
        lr = cosine_lr(step, a.steps, a.lr, a.min_lr)
        lp, lg = step_fn(sampler.batch(a.batch), lr)
        step += 1
        if math.isnan(lp):
            bad_total += 1
            bad_consec += 1
            if bad_consec >= MAX_BAD_STEPS:
                raise SystemExit(f"{MAX_BAD_STEPS} consecutive non-finite steps at step {step}: lower --lr")
        else:
            bad_consec = 0
            acc_p += lp
            acc_g += lg
            n_acc += 1
        if step % 100 == 0 and n_acc:
            print(f"step {step}/{a.steps} L1 {acc_p / n_acc:.4f} grad {acc_g / n_acc:.4f} lr {lr:.1e}"
                  + (f" | skipped non-finite steps so far: {bad_total}" if bad_total else ""))
        if step % a.val_every == 0 or step == a.steps:
            r = validator.run(net, dev, amp)
            mean = mean_over_sigmas(r, 0)
            edge, smo = mean_over_sigmas(r, 1), mean_over_sigmas(r, 2)
            if (math.isnan(edge) or math.isnan(smo)) and not warned_region:
                print("WARNING: a validation region (edge or smooth) is empty in every image; its PSNR is logged as nan "
                      "and is not used for checkpoint selection.")
                warned_region = True
            sps = (time.time() - t0) / max(n_acc, 1)
            print(f"== step {step}: val PSNR {r[15][0]:.2f}/{r[25][0]:.2f}/{r[50][0]:.2f} mean {mean:.3f} | "
                  f"edge {fmt(edge, 2)} smooth {fmt(smo, 2)} | {sps:.3f} s/step")
            with hist.open("a") as f:
                f.write(f"{step},{fmt(acc_p / max(n_acc, 1), 5)},{fmt(acc_g / max(n_acc, 1), 5)},{lr:.3e},"
                        f"{r[15][0]:.4f},{r[25][0]:.4f},{r[50][0]:.4f},{mean:.4f},{fmt(edge)},{fmt(smo)},{sps:.4f}\n")
            if not math.isnan(mean) and mean > best:             # selection uses the global validation PSNR only
                best = mean
                save(run_dir / "best.pth", model_only=True)
                print("   new best (validation only)")
            acc_p = acc_g = 0.0
            n_acc = 0
            t0 = time.time()
        if step % a.save_every == 0 or step == a.steps:
            save(run_dir / "last.pth")
    total = elapsed_prev + (time.time() - t_start)
    (run_dir / "run_summary.json").write_text(json.dumps(dict(
        run=name, steps=step, best_val_mean_psnr=best, training_time_s=round(total, 1), parameters=nparams,
        skipped_nonfinite_steps=bad_total, seed=a.seed, lam=a.lam,
        hardware=json.loads((run_dir / "config.json").read_text())["hardware"]), indent=2))
    print(f"done. best validation mean PSNR {best:.3f} | training time {total / 60:.1f} min")


if __name__ == "__main__":
    main()
