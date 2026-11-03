# BSD68 Denoising Experiment Protocol

**Status:** Protocol draft based on the scripts and project details currently available.  
**Purpose:** Compare baseline DnCNN (P), edge-regularized DnCNN (G), and tuned classical denoisers under matched evaluation conditions.

> **Reproducibility note:** The evaluator that generates `results/eval.csv` and final frozen configuration files were not supplied. This document records the supported methodology and flags details that still require confirmation.

## 1. Research questions

1. How does edge-regularized training (G) compare with the baseline pixelwise objective (P)?
2. How do learned models compare with tuned classical denoisers across noise levels?
3. What are the quality–runtime trade-offs?
4. How do results vary across smooth and edge-rich regions and across training seeds?

The pilot selects the G loss coefficient; it is not confirmatory evidence.

## 2. Data and splits

### Training

- Dataset: BSD400, grayscale.
- Fixed split: 380 training images / 20 validation images.
- Split seed: `0`.
- Manifest: `data/bsd400_split.json`.

Retain the fixed manifest with experiment outputs.

### Evaluation

- Benchmark: BSD68.
- Classical calibration: fixed subset of 8 images by default, center-cropped to 256×256.
- Calibration and held-out membership: `results/tuning/tune_split.json`.
- Classical parameter tuning uses calibration images only.
- Final evaluation should use held-out images and common image IDs across methods wherever possible.
- Inference supports `--only-heldout` with the tuning split.

## 3. Noise model and representation

For clean image `x` on the 8-bit scale:

\[
y=\operatorname{round}(\operatorname{clip}(x+n,0,255)),\quad n\sim\mathcal{N}(0,\sigma^2).
\]

Normalize noisy images to `[0,1]` for model input. Report sigma on the 0–255 scale.

- Training AWGN sigma: independently sampled per patch, uniformly from `[15,50]`.
- Training noise is added, clipped to `[0,255]`, rounded to integer levels, then normalized.
- Validation sigmas: `15, 25, 50`.
- Validation noise RNG: `RandomState(10000 + i*101 + sigma)`, where `i` is the implementation's image index.
- Reuse identical noisy realizations across methods for paired comparisons where the evaluator allows it.

## 4. Learned models

### Architecture

Both variants use the same grayscale DnCNN-style model:

- 17 convolutional layers; width 64; 3×3 kernels.
- Batch normalization and ReLU in intermediate blocks.
- Kaiming initialization.
- Predicted noise residual `fθ(y)`; restoration is `ŷ = y - fθ(y)`.
- Nominal receptive field: 35×35 pixels.
- Blind inference: sigma is not supplied to the model.

### Sampling and augmentation

- Patch size: 64×64; batch size: 64.
- Flips and 90° rotations.
- Dedicated patch sampler `np.random.RandomState(seed)`.
- Torch generator seed: `1000 + seed`; `torch.manual_seed(seed)` before model construction.

### Objectives

**P — pixelwise baseline**

\[
\mathcal L_P=\|\hat{x}-x\|_1.
\]

**G — edge-regularized objective**

\[
\mathcal L_G=\|\hat{x}-x\|_1+\lambda\mathcal L_{\mathrm{Sobel}}.
\]

The Sobel term is mean absolute discrepancy in x/y Sobel responses.

- Sobel kernels divided by 8.
- Replicate border padding.
- Loss averages absolute differences across Sobel directions.

### Optimization

- Adam, learning rate `1e-3`, cosine schedule to `1e-6`.
- Betas `(0.9, 0.999)`, weight decay `0`, epsilon `1e-8`.
- Gradient norm clipping: `1.0`.
- Pilot: 3,000 steps. Final runs: 20,000 steps.
- Mixed precision enabled by final run script.
- Validate every 1,000 steps; checkpoint every 500 steps.

### Checkpoint selection

At validation, compute full-image/regular PSNR at sigma 15, 25, and 50. Select the checkpoint maximizing the arithmetic mean of mean per-image PSNR across the three levels. Ties retain the earlier checkpoint. Quantize predictions to 8-bit levels for scoring.

The best checkpoint is model-only. The last checkpoint saves optimizer/scaler and sampler/generator state, but exact replay is not certified: global RNG state and deterministic-kernel settings have not been verified.

## 5. Pilot and lambda selection

Pilot results at step 3,000:

| Configuration | Regular PSNR (dB) | Edge PSNR (dB) | Smooth PSNR (dB) |
|---|---:|---:|---:|
| P, λ=0 | 27.2944 | 23.1178 | 32.7385 |
| G, λ=0.1 | 27.2485 | 23.0268 | 32.7329 |
| G, λ=0.5 | 27.1704 | 23.0904 | 31.7234 |
| G, λ=1.0 | 26.7209 | 23.0704 | 28.7150 |

Selection rule:

1. Eligibility threshold = pilot P regular PSNR − 0.15 dB.
2. Retain G candidates within the threshold.
3. Maximize edge-region PSNR among eligible candidates.
4. Tie-break with the smaller lambda.
5. If none qualify, select the smallest tested lambda.

Threshold: 27.1444 dB. G λ=0.1 and λ=0.5 qualify; λ=0.5 is selected by the stated rule.

**Caveat:** The edge-PSNR difference between λ=0.5 and λ=0.1 is 0.0636 dB, based on one pilot seed. This does not establish robust superiority.

## 6. Final learned runs

| Variant | λ | Seeds |
|---|---:|---|
| P | 0 | 0, 1 |
| G | 0.5 | 0, 1 |

Run names follow `dncnn_P_s0`, `dncnn_G0.5_s0`, etc. Matched seeds are intended to pair initialization, patch sampling, and noise RNG streams; bitwise-identical inputs or deterministic execution are not guaranteed.

Retain logs, configurations, checkpoint identifiers, and split manifest.

## 7. Classical methods and tuning

Core methods in the tuning script:

- Gaussian filtering
- Bilateral filtering
- Total variation (TV), when registered/available
- Non-local means (NLM)

Optional methods: median filtering and Perona–Malik diffusion. Record the actual registered method set; do not silently count unavailable methods as evaluated.

Tuning protocol:

- Tune independently for each sigma.
- Calibration subset only (8 images by default), center crop 256×256.
- Default budget: 150 trials per method.
- Objective: maximize mean calibration PSNR.
- Clip/round outputs to uint8 levels before scoring.
- Record parameter values, ranges, trial budget, objective, and split provenance.
- Check and report parameters near search boundaries.

No held-out image may be used to select parameters.

## 8. Inference and output

- Learned models are blind; do not pass sigma to the network.
- Output is clipped to `[0,1]` and saved as rounded 8-bit grayscale PNG.
- Use `--only-heldout results/tuning/tune_split.json` for held-out-only inference where applicable.
- Preserve image IDs and method/config metadata for paired analysis.

## 9. Metrics and region definitions

Available common utilities include:

- PSNR on normalized images; exact-zero MSE is reported as 100 dB.
- `quantize01` rounds to 8-bit levels.
- Pooled regional PSNR from squared-error sums and pixel counts.
- Gradient magnitude: Gaussian filter with sigma 1, then numerical gradients and Euclidean magnitude.
- Local standard deviation, 7×7 by default.

The available region helper defines:

- Smooth: gradient magnitude **and** local standard deviation are both at or below their 40th percentiles.
- Edge: gradient magnitude is at or above its 90th percentile.

This helper is documented for validation monitoring. The intended final protocol calibrates region thresholds on calibration images and freezes them for evaluation. **The final mask construction/freezing code was not supplied.**

The final evaluator was not supplied either. Exact final PSNR/SSIM and auxiliary metric implementations, regional masks, aggregation, and generation of `results/eval.csv` cannot be certified. Do not infer them from the plotting script.

## 10. Runtime benchmark

- All methods on the same CPU; learned methods on CPU and GPU when available.
- First 10 images by default; 2 warmups; 3 timed repeats per image.
- Time denoising only: exclude I/O, model loading, and metrics.
- Synchronize GPU around timing.
- Report median, 25th/75th percentiles, and seconds per megapixel.
- Record hardware and software versions.

Report CPU and GPU results separately. Do not present unlike hardware results as matched comparisons.

## 11. Statistical analysis

`stats_plots.py` treats images as experimental units.

- Average learned seed rows per image before image-level summaries.
- Report seed spread separately.
- Bootstrap 95% confidence intervals using 2,000 image resamples.
- Paired differences use overlapping image IDs and bootstrap resampling.
- Stratify summaries by noise level.
- Expected outputs: `summary.csv`, `paired.csv`, `seed_spread.csv`, and plots for PSNR vs sigma, SSIM vs sigma, quality vs runtime, and parameter response.

The script expects an evaluation CSV with image ID, split, seed, sigma, method, configuration, and metric columns. Confirm the actual schema and evaluator provenance before analysis.

## 12. Reproducibility checklist

- [ ] Code revision or archived scripts
- [ ] Dataset and split manifests
- [ ] Full config and command line
- [ ] Training seed, selected checkpoint and step
- [ ] Training/validation logs
- [ ] Noise generation settings and seeds
- [ ] Tuning split, search space, budget, and selected parameters
- [ ] Evaluation script version and metric definitions
- [ ] Per-image results and IDs
- [ ] Runtime hardware/software details
- [ ] Skipped/failed methods and reasons

Report results by noise level and method, image-level confidence intervals and paired differences, learned-seed spread, runtime with hardware context, and pilot-based lambda selection as exploratory.

## 13. Known limitations / pending verification

1. **Evaluator absent:** exact PSNR/SSIM and auxiliary metric definitions, regional masks, and result aggregation are not fully verified.
2. **Final configs absent:** preserve actual run configs and compare them against this protocol.
3. **Reproducibility:** paired seeds are intended, but deterministic execution and exact RNG replay have not been demonstrated.
4. **Pilot selection:** λ=0.5 was selected with one pilot seed and a small edge-PSNR difference.
5. **Method availability:** TV and optional classical methods depend on registration/environment.
6. **Benchmark scope:** runtime is interpretable only with hardware/software and timing details.

## Appendix: source artifacts

This protocol was assembled from the project notes and these scripts:

- `common.py`
- `train_dl.py`
- `run_experiments.sh`
- `dncnn.py`
- `models/losses.py`
- `dl_infer.py`
- `pick_lambda.py`
- `runtime_bench.py`
- `tune.py`
- `stats_plots.py`

This is the documented intended protocol; it does not certify that every experiment has completed or that all implementation details have been independently reproduced.
