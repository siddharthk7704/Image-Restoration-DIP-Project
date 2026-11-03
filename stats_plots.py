#!/usr/bin/env python
"""Statistics and plots from the per-image evaluation CSV (Person A owns plots 2-4, statistics, parameter curves).

Input  : results/eval.csv  with columns (Person B's evaluator):
         image_id, split, seed, sigma_255, method, parameter_config, model_id_optional, loss_variant,
         train_seed, psnr, ssim, smooth_psnr, edge_psnr, gradient_error, ...   (extra/missing metric columns are fine)
Rules  : images are the experimental unit. DL models trained with several seeds are first averaged per image
         over seeds; the seed-to-seed spread is reported separately. Bootstrap CIs resample IMAGES.
Output : results/stats/summary.csv, paired.csv, seed_spread.csv, plot2_psnr_vs_sigma.png, plot3_ssim_vs_sigma.png,
         plot4_quality_vs_runtime.png, param_response_*.png

  python stats_plots.py --eval-csv results/eval.csv --runtime-csv results/runtime.csv --tuning-dir results/tuning
Label = method, or method-loss_variant for trained models (e.g. dncnn-P, dncnn-G).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["psnr", "ssim", "smooth_psnr", "edge_psnr", "gradient_error", "gradient_error_sobel", "edge_f1_optional"]


def load_eval(path, split):
    df = pd.read_csv(path)
    for c in ("loss_variant", "train_seed"):
        if c not in df: df[c] = np.nan
    if split and "split" in df:
        df = df[df["split"] == split]
    lv = df["loss_variant"].fillna("").astype(str)
    df["label"] = np.where(lv != "", df["method"].astype(str) + "-" + lv, df["method"].astype(str))
    dup = df.duplicated(["label", "sigma_255", "image_id", "train_seed"], keep=False)
    if dup.any():
        print(f"WARNING: {int(dup.sum())} duplicate rows (same label/sigma/image/seed). Check the evaluator.")
    return df


def per_image(df, metric):
    """One value per label x sigma x image (DL seeds averaged)."""
    return df.groupby(["label", "sigma_255", "image_id"], as_index=False)[metric].mean()


def boot_mean_ci(x, rng, B=2000):
    x = np.asarray(x, float)
    idx = rng.randint(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(df, metrics, rng):
    rows = []
    for m in metrics:
        pi = per_image(df, m)
        for (lab, sig), g in pi.groupby(["label", "sigma_255"]):
            v = g[m].dropna().values
            if len(v) == 0: continue
            lo, hi = boot_mean_ci(v, rng)
            rows.append(dict(metric=m, label=lab, sigma_255=sig, n_images=len(v), mean=v.mean(), median=np.median(v),
                             std=v.std(ddof=1) if len(v) > 1 else np.nan, ci95_lo=lo, ci95_hi=hi))
    return pd.DataFrame(rows)


def seed_spread(df, metrics):
    d = df.dropna(subset=["train_seed"])
    rows = []
    for m in metrics:
        if m not in d: continue
        g = d.groupby(["label", "sigma_255", "train_seed"])[m].mean().reset_index()
        for (lab, sig), h in g.groupby(["label", "sigma_255"]):
            if len(h) >= 2:
                rows.append(dict(metric=m, label=lab, sigma_255=sig, n_seeds=len(h), seed_means=list(np.round(h[m], 4)),
                                 seed_std=h[m].std(ddof=1), seed_range=h[m].max() - h[m].min()))
    return pd.DataFrame(rows)


def paired(df, a, b, metric, sigma, rng):
    pi = per_image(df, metric)
    pa = pi[(pi.label == a) & (pi.sigma_255 == sigma)].set_index("image_id")[metric]
    pb = pi[(pi.label == b) & (pi.sigma_255 == sigma)].set_index("image_id")[metric]
    ids = pa.index.intersection(pb.index)
    if len(ids) == 0: return None
    d = (pa[ids] - pb[ids]).values
    lo, hi = boot_mean_ci(d, rng)
    return dict(metric=metric, sigma_255=sigma, a=a, b=b, n_images=len(ids), mean_diff_a_minus_b=d.mean(),
                ci95_lo=lo, ci95_hi=hi, frac_images_a_higher=float((d > 0).mean()),
                ci_excludes_zero=bool(lo > 0 or hi < 0))


def auto_pairs(labels):
    out = []
    for lab in labels:
        if lab.endswith("-G"):
            base = lab[:-2]
            if base + "-P" in labels: out.append((lab, base + "-P"))
    return out


def line_plot(sm, metric, ylabel, path, title):
    s = sm[sm.metric == metric]
    if s.empty: return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sig = sorted(s.sigma_255.unique())
    labels = sorted(s.label.unique())
    for i, lab in enumerate(labels):
        g = s[s.label == lab].sort_values("sigma_255")
        dl = "-" in lab
        ax.errorbar(g.sigma_255, g["mean"], yerr=[g["mean"] - g.ci95_lo, g.ci95_hi - g["mean"]], capsize=3,
                    marker="s" if dl else "o", ls="-" if dl else "--", label=lab, lw=1.5)
    ax.set_xticks(sig); ax.set_xlabel("noise sigma (0-255 scale)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def quality_runtime(sm, rt, sigma, device, metric, path):
    s = sm[(sm.metric == metric) & (sm.sigma_255 == sigma)].set_index("label")
    r = rt[(rt.device == device) & (rt.sigma_255 == sigma)].set_index("method")
    common = [l for l in s.index if l in r.index]
    missing = [l for l in s.index if l not in r.index]
    if missing: print(f"runtime missing for: {missing} (names in runtime.csv must equal the plot labels)")
    if not common: return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for l in common:
        ax.scatter(float(r.loc[l, "median_s"]), s.loc[l, "mean"], s=45)
        ax.annotate(l, (float(r.loc[l, "median_s"]), s.loc[l, "mean"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel(f"median seconds per image ({device.upper()}; hardware in runtime.csv)")
    ax.set_ylabel(f"mean {metric.upper()} (sigma {sigma})"); ax.grid(alpha=.3, which="both")
    ax.set_title("Quality versus runtime"); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def param_response(tuning_dir, out):
    for f in sorted(Path(tuning_dir).glob("*_sigma*.csv")):
        df = pd.read_csv(f)
        if "score" not in df: continue
        skip = {"trial", "score", "psnr_min", "psnr_std", "sec_per_image", "error"}
        cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
        if not cols: continue
        fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.2), squeeze=False)
        best = df.loc[df.score.idxmax()]
        for ax, c in zip(axes[0], cols):
            ax.scatter(df[c], df.score, s=10, alpha=.6)
            ax.scatter([best[c]], [best.score], c="r", s=40, zorder=3, label="best")
            if df[c].min() > 0 and df[c].max() / df[c].min() > 20: ax.set_xscale("log")
            ax.set_xlabel(c); ax.set_ylabel("PSNR (calibration images)"); ax.grid(alpha=.3)
        fig.suptitle(f.stem, fontsize=9); fig.tight_layout(); fig.savefig(Path(out) / f"param_response_{f.stem}.png", dpi=180)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-csv", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--runtime-csv", default=""); ap.add_argument("--tuning-dir", default="")
    ap.add_argument("--out", default="results/stats"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-focus", type=int, default=25); ap.add_argument("--runtime-device", default="cpu")
    ap.add_argument("--pairs", nargs="*", default=None, help="a:b pairs, e.g. dncnn-G:dncnn-P (default: all X-G vs X-P)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(a.seed)

    df = load_eval(a.eval_csv, a.split)
    metrics = [m for m in METRICS if m in df.columns and df[m].notna().any()]
    sm = summary(df, metrics, rng); sm.to_csv(out / "summary.csv", index=False)
    sp = seed_spread(df, metrics); sp.to_csv(out / "seed_spread.csv", index=False)

    labels = sorted(df.label.unique())
    pairs = [tuple(p.split(":")) for p in a.pairs] if a.pairs else auto_pairs(labels)
    rows = []
    for x, y in pairs:
        for m in metrics:
            for s in sorted(df.sigma_255.unique()):
                r = paired(df, x, y, m, s, rng)
                if r: rows.append(r)
    pd.DataFrame(rows).to_csv(out / "paired.csv", index=False)

    line_plot(sm, "psnr", "mean PSNR (dB), 95% bootstrap CI over images", out / "plot2_psnr_vs_sigma.png", "PSNR versus noise level")
    line_plot(sm, "ssim", "mean SSIM, 95% bootstrap CI over images", out / "plot3_ssim_vs_sigma.png", "SSIM versus noise level")
    if a.runtime_csv and Path(a.runtime_csv).exists():
        quality_runtime(sm, pd.read_csv(a.runtime_csv), a.sigma_focus, a.runtime_device, "psnr", out / "plot4_quality_vs_runtime.png")
    if a.tuning_dir and Path(a.tuning_dir).exists():
        param_response(a.tuning_dir, out)
    print(f"wrote results to {out}")
    if rows:
        d = pd.DataFrame(rows)
        print(d[d.metric == "psnr"][["a", "b", "sigma_255", "mean_diff_a_minus_b", "ci95_lo", "ci95_hi", "ci_excludes_zero"]].to_string(index=False))


if __name__ == "__main__":
    main()
