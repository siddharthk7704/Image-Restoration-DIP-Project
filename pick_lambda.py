#!/usr/bin/env python
"""Apply the declared lambda-selection rule to the pilot runs (validation data only).

Pilot runs: same seed, same (short) schedule, only the loss differs. Name them pilot_P, pilot_G0.1, pilot_G0.5, ...
    python train_dl.py --lam 0   --steps 3000 --name pilot_P
    python train_dl.py --lam 0.1 --steps 3000 --name pilot_G0.1   (and 0.5, 1.0)

Rule (write it in PROTOCOL.md BEFORE running; validation images only, never test data):
  1. keep the lambdas whose best validation mean PSNR is within --tol dB of the pilot_P run;
  2. among those, lambda* = the one with the highest validation edge-region PSNR (ties: the smaller lambda);
  3. if none passes step 1, take the smallest lambda and say so in the report.
Both quantities come from the validation images; the test set is never used.
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def best_row(run):
    h = pd.read_csv(Path(run) / "history.csv")
    return h.loc[h.psnr_mean.idxmax()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs"); ap.add_argument("--tol", type=float, default=0.15)
    a = ap.parse_args()
    base = Path(a.runs) / "pilot_P"
    if not base.exists():
        raise SystemExit(f"{base} not found: run the pilot_P run first")
    b = best_row(base)
    print(f"pilot_P: val PSNR {b.psnr_mean:.3f} dB, edge {b.edge_psnr_mean:.3f}, smooth {b.smooth_psnr_mean:.3f}")
    cands, dirs = [], {}
    for d in sorted(Path(a.runs).glob("pilot_G*")):
        lam = float(re.sub(r"^pilot_G", "", d.name))
        dirs[lam] = d
        r = best_row(d)
        ok = b.psnr_mean - r.psnr_mean <= a.tol
        cands.append((lam, ok))
        print(f"{d.name:12s}: val PSNR {r.psnr_mean:.3f} ({r.psnr_mean - b.psnr_mean:+.3f})  "
              f"edge {r.edge_psnr_mean:.3f} ({r.edge_psnr_mean - b.edge_psnr_mean:+.3f})  "
              f"smooth {r.smooth_psnr_mean:.3f} ({r.smooth_psnr_mean - b.smooth_psnr_mean:+.3f})  "
              f"{'within tolerance' if ok else 'too costly'}")
    if not cands:
        raise SystemExit("no pilot_G* runs found")
    ok = [l for l, k in cands if k]
    if ok:
        edge = {l: best_row(dirs[l]).edge_psnr_mean for l in ok}
        valid = {l: e for l, e in edge.items() if pd.notna(e)}
        if not valid:
            raise SystemExit("edge-region PSNR is nan for every candidate (empty validation edge region?): "
                             "check the validation masks printed at the start of training")
        if len(valid) < len(edge):
            print("note: lambdas with nan edge PSNR are ignored in step 2")
        top = max(valid.values())
        chosen = min(l for l in valid if abs(valid[l] - top) < 1e-9)
    else:
        chosen = min(l for l, _ in cands)
    print(f"\nlambda* = {chosen:g}" + ("" if ok else "  (no lambda within tolerance: smallest taken; say so in the report)"))


if __name__ == "__main__":
    main()
