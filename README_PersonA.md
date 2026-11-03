# Person A: files and order of work

## Layout (run everything from the project root)
```
common.py            shared helpers: noisy rule, PSNR, region masks (validation monitoring)
make_noisy.py        noise pipeline + manifest (gen) / inspect existing noisy sets (check)
tune.py              automatic classical tuning (Gaussian, median, bilateral, Perona-Malik, NLM)   [existing, NLM added]
filters/             core.py, classical.py, nlm.py, api.py   ([0,1] interface: filters.api.denoise; units in PARAMETER_UNITS)
                     (median and perona_malik stay in the package as OPTIONAL extras; not tuned by default)
models/              dncnn.py, losses.py (Sobel gradient loss), __init__.py (architecture registry)
train_dl.py          training for DnCNN (and Person B's U-Net, SAME script)
pick_lambda.py       applies the declared lambda rule to the pilot runs (validation PSNR + validation edge-region PSNR)
dl_infer.py          denoise_dl(image_0_1, model_config) + folder inference  (shared learned-model contract; blind: no sigma)
runtime_bench.py     one timing protocol for all methods
stats_plots.py       bootstrap CIs, paired differences, seed spread, plots 2-4, parameter-response curves
run_experiments.sh   the whole DL experiment matrix in order
tests/               python -m pytest -q
notes/               paper_adaptation_card.md, theory_note_A.md  (templates: fill after reading)
```
Install: `pip install -r requirements.txt`. Test: `python -m pytest -q`.

## Order of work
**Days 1-2.** Fill `PROTOCOL.md` from the project plan's frozen-protocol table. Put BSD400 in `data/BSD400/`, BSD68 in `data/BSD68/`.
`python make_noisy.py check --clean-dir data/BSD68 --noisy-dir <your existing noisy folder> --sigma 25` tells you whether your
existing noisy sets are usable. Using them is fine (the tuning you did stays valid). To regenerate with seeds and a manifest instead:
`python make_noisy.py gen --clean-dir data/BSD68 --out-dir data/noisy/BSD68 --sigmas 15 25 50`, then rerun `tune.py` (realizations change).

**Days 3-5.** Gaussian: already done. NLM: `python tune.py --methods nlm --sigma 25 --noisy-dir <sigma25 folder> --budget 150` (and 15, 50).
`python train_dl.py --time-test --amp` on Kaggle to size depth, width and steps; `--overfit-check` must report OK. Then start the pilots.

**Days 6-7.** `run_experiments.sh` section 1 (pilots) then `python pick_lambda.py`; freeze lambda* in PROTOCOL.md before starting the final runs.
Freeze classical settings (`params.yaml`) and `git commit`.
The rule (written in PROTOCOL.md before the pilots): keep lambdas whose validation PSNR is within 0.15 dB of P; among them take the highest validation edge-region PSNR.

**Days 8-10.** Final runs (2 seeds each of P and G; the run folder is named `dncnn_G<lambda as %g>_s<seed>`, e.g. `dncnn_G1_s0` for lambda 1.0), then inference into `outputs/<run>/BSD68/sigma<S>/` for B's evaluator.
`python runtime_bench.py --noisy-dir data/noisy/BSD68/sigma25 --sigma 25 --methods gaussian bilateral nlm --dl dncnn_P=runs/dncnn_P_s0/best.pth ...`
(runtime method names must equal the plot labels, e.g. `dncnn-P`).

**Days 11-12.** `python stats_plots.py --eval-csv results/eval.csv --runtime-csv results/runtime.csv --tuning-dir results/tuning`

## Kaggle notes
Upload this folder as a Kaggle dataset (code) and BSD400 as another dataset. In the notebook:
```
!cp -r /kaggle/input/<code-dataset>/* /kaggle/working/ && cd /kaggle/working
!python train_dl.py --train-dir /kaggle/input/<bsd400>/BSD400 --out /kaggle/working/runs --amp --lam 0 --seed 0
```
Sessions end: `last.pth` is saved every 500 steps; continue with `--resume runs/<name>/last.pth`.
Download `runs/*/best.pth` and `history.csv` before the session closes. Training must NOT use BSD68 (the script only reads `--train-dir`).

## Matching rules the code enforces
P and G with the same `--seed` have identical initialization, data order and noise draws; only the loss differs. The split of
BSD400 is written once to `data/bsd400_split.json`: give the same file to Person B so both networks use the same 380/20 split.
Checkpoints are chosen by validation PSNR only. Noisy inputs follow the 8-bit clip-and-round rule in training and in the test files.

## Hand-off to Person B
`filters/` + `params.yaml` (frozen), `dl_infer.py` (`load_model`, `denoise_dl`), `outputs/<run>/BSD68/sigma<S>/*.png`, `data/bsd400_split.json`,
`runs/*/config.json` and `run_summary.json` (hyper-parameters, model description, parameter count, training time, hardware), `models/__init__.py` (B adds `models/unet.py`, same `train_dl.py`).

## What each training run records (guide section 6)
`config.json`: depth, width, kernel, padding, normalization, optimizer settings, batch size, steps, seed, parameter count, hardware.
`history.csv`: loss curves and validation PSNR (per sigma, edge-region, smooth-region). `baselines.json`: validation noisy-input and
calibrated-Gaussian scores the network has to beat. `run_summary.json`: best validation score and total training time.
