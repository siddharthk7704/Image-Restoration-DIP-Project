#!/usr/bin/env python
"""tune.py: automatic per-sigma parameter calibration for the classical denoisers (and any other search).

Follows the project plan ("Calibration only"):
  * methods are tuned PER SIGMA on the frozen calibration images only (default 8 crops of BSD68);
  * the same trial budget, the same objective (mean 8-bit PSNR) and the same rule for every method;
  * the final test images are never touched: results/tuning/tune_split.json lists calibration ("tune_images")
    and held-out test ("heldout_images") images, frozen on first use and reused for every sigma and method;
  * search ranges, rule and chosen settings are saved (CSV of every trial + best_*.json + spaces_used_*.json).

Data layout (relative to the project root; override with flags)
    data/BSD68/*.png                       clean images
    data/noisy/BSD68/sigma15|25|50/*.png   noisy images (make_noisy.py gen, or your existing noisy folders)

Outputs
    results/tuning/<method>_sigma<S>.csv           every trial (parameters, mean PSNR, runtime)
    results/tuning/best_<method>_sigma<S>.json     best trial + provenance + the search space and rule used
    results/tuning/tune_split.json                 calibration / held-out image names
    params.yaml                                    {method: {sigma<S>: {param: value, ...}}}  (0..255-scale units)

Typical use
    python tune.py --time-test                         # 1) time one trial per method, project total
    python tune.py                                     # 2) all core methods, sigma 15 25 50 (resumable)
    python tune.py --methods nlm --sigmas 25 --budget 150
    python tune.py --noisy-dir C:/path/BSD68_noisy --sigma 25     # one existing noisy folder, old style

Methods: core = gaussian, bilateral, tv (Person B registers it in filters/), nlm.  median and perona_malik are
OPTIONAL extras (not tuned unless named in --methods).  Search ranges can be overridden per sigma in config.yaml:

    search_spaces:                  # all sigmas
      bilateral: {sigma_r: {type: loguniform, low: 5, high: 80}, ...}
    search_spaces_by_sigma:         # only sigma 50 (wider ranges are needed at strong noise)
      50:
        bilateral: {sigma_s: {...}, sigma_r: {type: loguniform, low: 20, high: 300}, k: 3.0, max_radius: 12}

Reuse for other searches (e.g. B's Canny thresholds)
    from tune import Space, search
    space = Space({"low": {"type": "uniform", "low": 0.02, "high": 0.2},
                   "high": {"type": "uniform", "low": 0.1, "high": 0.5}})
    trials = space.sample(budget=40, seed=0)
    rows = search(my_objective, trials, "results/tuning/canny.csv", n_jobs=4)   # my_objective(params) -> {"score": F1, ...}

Notes
  * If you change a search range, delete that method's CSV (the script refuses to resume from a log made with
    different ranges). A larger --budget alone just extends the log.
  * Windows uses "spawn" multiprocessing, which is why everything runs under main().
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from filters import METHODS, get_filter

CLEAN_DIR = Path("data/BSD68")
NOISY_ROOT = Path("data/noisy/BSD68")
CORE_ORDER = ["gaussian", "bilateral", "tv", "nlm"]          # tv appears once Person B registers it in filters/
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SELECTION_RULE = "maximize mean PSNR over the calibration images, outputs rounded and clipped to 8-bit"

# --------------------------------------------------------------------------- search spaces
# Types: loguniform / uniform (continuous), int (inclusive), choice (list), fixed. All units are the 0..255 scale
# unless noted. Defaults are the ones already used for sigma 15 and 25; sigma 50 needs the wider overrides in config.yaml.
DEFAULT_SPACES = {
    "gaussian": {
        "sigma": {"type": "loguniform", "low": 0.3, "high": 3.0},
        "truncate": {"type": "fixed", "value": 3.0},
    },
    "bilateral": {
        "sigma_s": {"type": "loguniform", "low": 0.8, "high": 4.0},
        "sigma_r": {"type": "loguniform", "low": 5.0, "high": 80.0},
        "k": {"type": "fixed", "value": 3.0},
        "max_radius": {"type": "fixed", "value": 12},
    },
    "tv": {   # Person B's wrapper must accept these names; weight follows scikit-image denoise_tv_chambolle ([0,1] image scale)
        "weight": {"type": "loguniform", "low": 0.01, "high": 1.0},
        "eps": {"type": "fixed", "value": 2e-4},
        "max_num_iter": {"type": "fixed", "value": 200},
    },
    "nlm": {
        "patch_size": {"type": "choice", "values": [3, 5, 7]},
        "patch_distance": {"type": "choice", "values": [4, 6, 9]},
        "h": {"type": "loguniform", "low": 5.0, "high": 120.0},
        "sigma_255": {"type": "fixed", "value": 0.0},
        "fast_mode": {"type": "fixed", "value": True},
    },
    # optional extras
    "median": {"size": {"type": "choice", "values": [3, 5, 7, 9, 11]}},
    "perona_malik": {
        "K": {"type": "loguniform", "low": 2.0, "high": 60.0},
        "num_iter": {"type": "int", "low": 5, "high": 100},
        "lam": {"type": "uniform", "low": 0.05, "high": 0.25},
        "conductance": {"type": "choice", "values": ["rational", "exp"]},
    },
}

# Trial 0 of every search: a reasonable starting point (also a sanity baseline).
DEFAULT_POINTS = {
    "gaussian": {"sigma": 1.0, "truncate": 3.0},
    "bilateral": {"sigma_s": 2.0, "sigma_r": 25.0, "k": 3.0, "max_radius": 12},
    "tv": {"weight": 0.1, "eps": 2e-4, "max_num_iter": 200},
    "nlm": {"patch_size": 5, "patch_distance": 6, "h": 25.0, "sigma_255": 0.0, "fast_mode": True},
    "median": {"size": 3},
    "perona_malik": {"K": 20.0, "num_iter": 20, "lam": 0.2, "conductance": "rational"},
}

METRICS = ("score", "psnr_min", "psnr_std", "sec_per_image")


# =========================================================================== search space
class Space:
    """Parameter space with deterministic, prefix-stable sampling."""

    def __init__(self, spec):
        self.spec = {}
        for name, s in spec.items():
            self.spec[name] = s if isinstance(s, dict) else {"type": "fixed", "value": s}
        for name, s in self.spec.items():
            if s["type"] not in ("fixed", "uniform", "loguniform", "int", "choice"):
                raise ValueError(f"{name}: unknown type '{s['type']}'")

    @property
    def names(self):
        return list(self.spec)

    @property
    def tuned(self):
        return [n for n, s in self.spec.items() if s["type"] != "fixed"]

    def _is_discrete(self):
        return all(s["type"] in ("fixed", "int", "choice") for s in self.spec.values())

    def _values(self, s):
        if s["type"] == "fixed":
            return [s["value"]]
        if s["type"] == "int":
            return list(range(int(s["low"]), int(s["high"]) + 1))
        return list(s["values"])

    def _draw(self, rng, s):
        t = s["type"]
        if t == "fixed":
            return s["value"]
        if t == "uniform":
            return float(rng.uniform(s["low"], s["high"]))
        if t == "loguniform":
            return float(np.exp(rng.uniform(np.log(s["low"]), np.log(s["high"]))))
        if t == "int":
            return int(rng.randint(int(s["low"]), int(s["high"]) + 1))
        return s["values"][rng.randint(len(s["values"]))]

    def sample(self, budget, seed=0, default=None):
        """Up to `budget` parameter dicts; trial 0 is `default` if given. A discrete space with no more than `budget`
        points is returned whole. Sampling is sequential, so a larger budget extends a smaller one (resumable)."""
        trials = []
        if default is not None:
            trials.append({n: default.get(n, self.spec[n].get("value")) for n in self.names})
        key = lambda d: tuple(d[n] for n in self.names)  # noqa: E731
        seen = {key(trials[0])} if trials else set()

        if self._is_discrete():
            grid = [dict(zip(self.names, combo))
                    for combo in itertools.product(*[self._values(self.spec[n]) for n in self.names])]
            if len(grid) <= budget:
                for g in grid:
                    if key(g) not in seen:
                        trials.append(g)
                        seen.add(key(g))
                return trials

        rng = np.random.RandomState(seed)
        attempts = 0
        while len(trials) < budget and attempts < budget * 50:
            attempts += 1
            cand = {n: self._draw(rng, self.spec[n]) for n in self.names}
            if self._is_discrete() and key(cand) in seen:
                continue
            seen.add(key(cand))
            trials.append(cand)
        return trials

    def at_bounds(self, params, margin=0.03):
        """Tuned parameters whose best value sits at/near the edge of the range. For `choice` lists only the UPPER
        edge counts (small sizes such as a 3x3 window are natural lower limits, not a search-range problem)."""
        hits = []
        for n in self.tuned:
            s, v = self.spec[n], params[n]
            if s["type"] in ("uniform", "loguniform"):
                lo, hi = s["low"], s["high"]
                if s["type"] == "loguniform":
                    lo, hi, v = math.log(lo), math.log(hi), math.log(v)
                t = (v - lo) / (hi - lo)
                if t < margin or t > 1 - margin:
                    hits.append(n)
            elif s["type"] == "int":
                if v in (s["low"], s["high"]):
                    hits.append(n)
            elif s["type"] == "choice":
                vals = s["values"]
                if len(vals) >= 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals) \
                        and v == max(vals):
                    hits.append(n)
        return hits


# =========================================================================== generic search
def _read_log(path, names):
    """{trial_id: {metric: float, '_params': {...}}} from an existing CSV (for resuming)."""
    done = {}
    if not Path(path).exists():
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                tid = int(row["trial"])
            except (KeyError, ValueError):
                continue
            if row.get("error"):
                continue  # retry failed trials
            done[tid] = {m: float(row[m]) for m in METRICS if row.get(m) not in (None, "")}
            done[tid]["_params"] = {n: row.get(n) for n in names}
    return done


def _same(logged, value):
    try:
        return math.isclose(float(logged), float(value), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(logged) == str(value)


def search(objective, trials, log_path, *, n_jobs=1, initializer=None, initargs=(),
           metric_names=METRICS, verbose=True):
    """Evaluate `objective(params) -> dict(score=..., ...)` on every trial, in parallel.

    Every finished trial is appended to `log_path` immediately (safe to Ctrl-C and rerun). `objective` must be
    picklable (module-level function or functools.partial of one). Returns [(trial_id, params, metrics), ...].
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(trials[0].keys())
    done = _read_log(log_path, names)
    for tid, d in done.items():  # refuse to resume from a log made with a different space/seed
        if tid < len(trials) and not all(_same(d["_params"][n], trials[tid][n]) for n in names):
            sys.exit(f"\n{log_path} was produced with a different search space, seed or parameter set "
                     f"(trial {tid} does not match).\nMove or delete that CSV and rerun; a larger --budget alone is fine.")
    todo = [(i, p) for i, p in enumerate(trials) if i not in done]

    fields = ["trial"] + names + list(metric_names) + ["error"]
    new_file = not log_path.exists()
    best = max((m["score"] for m in done.values()), default=-math.inf)
    if verbose:
        print(f"  {len(done)} trials already logged, {len(todo)} to run ({n_jobs} worker{'s' if n_jobs != 1 else ''})")

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()

        def record(tid, params, result, error=""):
            nonlocal best
            row = {"trial": tid, **params, "error": error}
            if result:
                row.update({m: result.get(m, "") for m in metric_names})
                done[tid] = {m: float(result[m]) for m in metric_names if m in result}
                done[tid]["_params"] = {n: params[n] for n in names}
                best = max(best, result["score"])
            writer.writerow(row)
            f.flush()
            if verbose:
                s = f"{result['score']:.3f}" if result else "FAILED"
                print(f"  trial {tid:3d}  score {s:>7}  best {best:.3f}   "
                      + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()))

        if n_jobs <= 1 or len(todo) <= 1:
            if initializer:
                initializer(*initargs)
            for tid, params in todo:
                try:
                    record(tid, params, objective(params))
                except Exception as e:  # keep going; log the failure
                    record(tid, params, None, error=repr(e))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs, initializer=initializer, initargs=initargs) as ex:
                futs = {ex.submit(objective, p): (tid, p) for tid, p in todo}
                for fut in as_completed(futs):
                    tid, params = futs[fut]
                    try:
                        record(tid, params, fut.result())
                    except Exception as e:
                        record(tid, params, None, error=repr(e))

    return [(i, trials[i], done[i]) for i in sorted(done) if i < len(trials)]


# =========================================================================== data + metric
def psnr(ref, test, peak=255.0):
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10.0 * math.log10(peak ** 2 / mse)


def list_pairs(clean_dir, noisy_dir, noisy_suffix=""):
    """Pair clean and noisy files by stem (noisy stem = clean stem + noisy_suffix)."""
    for d in (clean_dir, noisy_dir):
        if not Path(d).is_dir():
            sys.exit(f"folder not found: {d}  (use --clean-dir / --noisy-root / --noisy-dir)")
    clean = sorted(p for p in Path(clean_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    noisy = {p.stem: p for p in Path(noisy_dir).iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs, missing = [], []
    for c in clean:
        n = noisy.get(c.stem + noisy_suffix)
        (pairs if n is not None else missing).append((c, n) if n is not None else c.name)
    if missing:
        print(f"WARNING: {len(missing)} clean images have no noisy match (first few: {missing[:3]}). "
              f"Use --noisy-suffix if names differ.")
    if not pairs:
        sys.exit("No clean/noisy pairs found. Check the folders and --noisy-suffix.")
    return pairs


def choose_split(pairs, n_tune, seed, split_file):
    """Freeze the calibration images once (tune_split.json) and reuse them for every method and sigma."""
    names = [c.name for c, _ in pairs]
    split_file = Path(split_file)
    if split_file.exists():
        saved = json.loads(split_file.read_text())
        tune_names = [n for n in saved["tune_images"] if n in names]
        if len(tune_names) == len(saved["tune_images"]):
            return tune_names
        print("WARNING: tune_split.json does not match the current image list; recreating it.")
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(names), size=min(n_tune, len(names)), replace=False))
    tune_names = [names[i] for i in idx]
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps({
        "seed": seed,
        "tune_images": tune_names,
        "heldout_images": [n for n in names if n not in tune_names],
        "note": "tune_images = calibration images (tune and set mask thresholds here only); "
                "report final results on heldout_images only.",
    }, indent=2))
    return tune_names


def load_image(path, mode, crop):
    a = np.array(Image.open(path).convert(mode), dtype=np.float32)
    if crop:
        H, W = a.shape[:2]
        ch, cw = min(crop, H), min(crop, W)
        y0, x0 = (H - ch) // 2, (W - cw) // 2
        a = a[y0:y0 + ch, x0:x0 + cw]
    return a


_DATA = None  # per-process cache, filled by _init_worker


def _init_worker(spec):
    global _DATA
    clean, noisy = [], []
    for c, n in spec["pairs"]:
        a, b = load_image(c, spec["mode"], spec["crop"]), load_image(n, spec["mode"], spec["crop"])
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {c}: clean {a.shape} vs noisy {b.shape}")
        clean.append(a)
        noisy.append(b)
    _DATA = (clean, noisy)


def denoise_objective(method, quantize, params):
    """One trial: mean PSNR of `method(**params)` over the calibration images."""
    clean, noisy = _DATA
    fn = get_filter(method)
    scores, t0 = [], time.perf_counter()
    for c, n in zip(clean, noisy):
        out = fn(n, **params)
        if quantize:
            out = np.clip(np.rint(out), 0, 255)
        scores.append(psnr(c, out))
    return {"score": float(np.mean(scores)), "psnr_min": float(np.min(scores)), "psnr_std": float(np.std(scores)),
            "sec_per_image": (time.perf_counter() - t0) / len(clean)}


# =========================================================================== outputs
def _native(x):
    return x.item() if isinstance(x, np.generic) else x


def update_params_file(path, method, sigma, params):
    path = Path(path)
    data = {}
    try:
        import yaml
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
        data.setdefault(method, {})[f"sigma{sigma}"] = {k: _native(v) for k, v in params.items()}
        path.write_text(yaml.safe_dump(data, sort_keys=True))
    except ImportError:
        path = path.with_suffix(".json")
        if path.exists():
            data = json.loads(path.read_text())
        data.setdefault(method, {})[f"sigma{sigma}"] = {k: _native(v) for k, v in params.items()}
        path.write_text(json.dumps(data, indent=2))
        print(f"PyYAML not installed (pip install pyyaml); wrote {path} instead.")
    return path


def _lib_versions():
    out = {"numpy": np.__version__, "python": sys.version.split()[0]}
    for mod in ("skimage", "scipy", "cv2"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def load_spaces(config_path, sigma):
    """DEFAULT_SPACES, then config.yaml `search_spaces`, then `search_spaces_by_sigma[sigma]`."""
    spaces = deepcopy(DEFAULT_SPACES)
    if config_path and Path(config_path).exists():
        try:
            import yaml
        except ImportError:
            print("PyYAML not installed; ignoring config.yaml.")
            return spaces
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
        for m, s in (cfg.get("search_spaces") or {}).items():
            spaces[m] = s
        by_sigma = cfg.get("search_spaces_by_sigma") or {}
        for key in (sigma, str(sigma)):
            for m, s in (by_sigma.get(key) or {}).items():
                spaces[m] = s
    return spaces


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    ap.add_argument("--noisy-root", type=Path, default=NOISY_ROOT, help="folder containing sigma<S>/ subfolders")
    ap.add_argument("--sigmas", type=int, nargs="+", default=[15, 25, 50])
    ap.add_argument("--noisy-dir", type=Path, default=None, help="one existing noisy folder (use with --sigma)")
    ap.add_argument("--sigma", type=int, default=25, help="its noise level (label only)")
    ap.add_argument("--noisy-suffix", default="", help="noisy file stem = clean stem + suffix")
    ap.add_argument("--methods", nargs="+", default=None, choices=list(METHODS),
                    help="default: the core methods that are registered (gaussian bilateral tv nlm); "
                         "median and perona_malik are optional extras")
    ap.add_argument("--budget", type=int, default=150, help="trials per method (the same for all methods and sigmas)")
    ap.add_argument("--n-tune", type=int, default=8, help="number of calibration images")
    ap.add_argument("--crop", type=int, default=256, help="center-crop calibration images (0 = full size)")
    ap.add_argument("--rgb", action="store_true", help="keep RGB (default: grayscale)")
    ap.add_argument("--no-quantize", action="store_true", help="skip rounding/clipping to 8-bit before PSNR (must match PROTOCOL.md)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/tuning"))
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--config", type=Path, default=Path("config.yaml"), help="optional search-space overrides")
    ap.add_argument("--time-test", action="store_true", help="time one trial per method, print projected total, then exit")
    ap.add_argument("--list-methods", action="store_true")
    args = ap.parse_args()

    registered = list(METHODS)
    if args.list_methods:
        print("registered:", registered, "| core order:", CORE_ORDER)
        return
    methods = args.methods or [m for m in CORE_ORDER if m in METHODS]
    if args.methods is None and "tv" not in METHODS:
        print("note: 'tv' is not registered in filters/ yet (Person B adds it); tuning the other core methods.")
    unknown = [m for m in methods if m not in DEFAULT_SPACES]
    if unknown:
        sys.exit(f"no search space defined for {unknown}; add one to DEFAULT_SPACES or config.yaml")

    mode, quantize = ("RGB" if args.rgb else "L"), not args.no_quantize
    sigma_dirs = {args.sigma: args.noisy_dir} if args.noisy_dir else {s: args.noisy_root / f"sigma{s}" for s in args.sigmas}

    summary = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for sigma, noisy_dir in sigma_dirs.items():
        pairs = list_pairs(args.clean_dir, noisy_dir, args.noisy_suffix)
        tune_names = choose_split(pairs, args.n_tune, args.seed, args.out / "tune_split.json")
        tune_pairs = [(str(c), str(n)) for c, n in pairs if c.name in tune_names]
        spec = {"pairs": tune_pairs, "mode": mode, "crop": args.crop}
        print(f"\n########## sigma {sigma}: {len(pairs)} pairs, calibrating on {len(tune_pairs)} "
              f"({'RGB' if args.rgb else 'grayscale'}, crop {args.crop or 'off'}); {len(pairs) - len(tune_pairs)} held out for testing")

        _init_worker(spec)
        clean, noisy = _DATA
        noisy_psnr = float(np.mean([psnr(c, n) for c, n in zip(clean, noisy)]))
        print(f"Noisy-input PSNR on the calibration images: {noisy_psnr:.2f} dB (theory without clipping "
              f"{20.0 * math.log10(255.0 / sigma):.2f} dB; clipping at 0/255 raises it a little, more at high sigma)")
        spaces = load_spaces(args.config, sigma)

        if args.time_test:
            for m in methods:
                t0 = time.perf_counter()
                r = denoise_objective(m, quantize, DEFAULT_POINTS[m])
                dt = time.perf_counter() - t0
                n_trials = len(Space(spaces[m]).sample(args.budget, args.seed, DEFAULT_POINTS[m]))
                print(f"{m:13s} default trial {dt:6.1f} s ({r['sec_per_image']:.2f} s/image), PSNR {r['score']:.2f} dB "
                      f"-> ~{dt * n_trials / args.jobs / 60:.1f} min for {n_trials} trials on {args.jobs} workers "
                      f"(costly settings may take longer)")
            return

        for m in methods:
            space = Space(spaces[m])
            trials = space.sample(args.budget, args.seed, DEFAULT_POINTS[m])
            print(f"\n=== {m} (sigma {sigma}): {len(trials)} trials, tuning {space.tuned} ===")
            log = args.out / f"{m}_sigma{sigma}.csv"
            results = search(partial(denoise_objective, m, quantize), trials, log, n_jobs=args.jobs,
                             initializer=_init_worker, initargs=(spec,))
            if not results:
                print(f"  no successful trials for {m}; see {log}")
                continue
            tid, best, met = max(results, key=lambda r: r[2]["score"])
            flagged = space.at_bounds(best)
            print(f"  BEST trial {tid}: PSNR {met['score']:.3f} dB (+{met['score'] - noisy_psnr:.2f} over noisy)  {best}")
            if flagged:
                print(f"  WARNING: best value of {flagged} is at/near the search bound. Widen that range in config.yaml "
                      f"(and delete {log.name}) or accept it deliberately.")
            update_params_file(args.params, m, sigma, {n: best[n] for n in space.names})
            (args.out / f"best_{m}_sigma{sigma}.json").write_text(json.dumps({
                "method": m, "sigma": sigma, "params": {k: _native(v) for k, v in best.items()},
                "tune_psnr": met["score"], "noisy_psnr": noisy_psnr, "n_trials": len(results), "budget": args.budget,
                "selection_rule": SELECTION_RULE, "search_space": space.spec, "default_trial": DEFAULT_POINTS[m],
                "at_bound": flagged, "calibration_images": tune_names, "crop": args.crop, "mode": mode,
                "quantized_to_8bit": quantize, "seed": args.seed, "git_commit": git_commit(),
                "versions": _lib_versions(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2, default=str))
            summary[(m, sigma)] = (met["score"], flagged)

    print("\nSummary: best calibration-set PSNR per method and sigma")
    for (m, s), (v, fl) in summary.items():
        print(f"  {m:13s} sigma {s:3d}  {v:6.2f} dB" + (f"   [at bound: {', '.join(fl)}]" if fl else ""))
    print(f"\nBest parameters written to {args.params}; search spaces and rule saved in {args.out}/best_*.json")


if __name__ == "__main__":
    main()
