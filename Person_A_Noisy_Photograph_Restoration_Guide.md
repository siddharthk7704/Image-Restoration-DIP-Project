# Person A Guide
## Paper Adaptation, Noise Pipeline, Gaussian and NLM, DnCNN, Runtime and Statistics

### Mission
Build reproducible input generation and implement or integrate Gaussian, Non-Local Means (NLM) and DnCNN for the study:

**An Experimental Study of Classical and Deep Learning-Based Image Denoising: Noise Robustness, Detail Preservation, and Computational Efficiency**

You own:
- Primary-paper reading note and adaptation card.
- Shared preprocessing, noise generation, seed and manifest, I/O contract tests.
- Gaussian and NLM implementations.
- **DnCNN: implement and train on Kaggle GPU**, pixel loss (P) and pixel + Sobel loss (G), 2 seeds each.
- **Runtime benchmarking** for all methods.
- **Statistics** (paired differences, bootstrap confidence intervals, seed spread, P-versus-G tests) and **plots 2 to 4** plus parameter-response curves.

Person B owns bilateral, TV, the U-Net, the evaluator core, region masks, gradient error, the equal-smoothing analysis and the failure gallery. Both of you must understand every method.

## 1. Research question and scope
Compare classical and learned denoisers under controlled AWGN: how do noise suppression, edge/texture retention and runtime vary with noise severity and image content?

Noise sigma = 15, 25, 50 on the 0 to 255 scale. Grayscale. Classical core: Gaussian, bilateral, TV, NLM. Modern core: your DnCNN and Person B's U-Net. **Optional only after the core is complete:** Perona-Malik, median and impulse noise, Restormer reference, Canny agreement, Fourier analysis, lambda sweep, sigma-specific DL models, Wiener or BM3D (not both).

## Frozen protocol (identical in all three documents)

| Item | Decision |
| --- | --- |
| Test data | BSD68, grayscale. Image IDs frozen in the manifest: **8 calibration images** (classical tuning and mask thresholds) and **60 test images**. Test images are never used for tuning, checkpoint choice or threshold choice. |
| DL training data | **BSD400** (400 images). Split **before** patch extraction: 380 train, **20 validation** (checkpoint choice and the lambda pilot). BSD68 is never used for training or model selection. |
| Preprocessing | Grayscale, float32 in [0,1]. Interface scale is [0,1] everywhere; classical parameters that live on the 0 to 255 scale (bilateral range sigma) are converted inside the method wrapper and stored in documented units. |
| Test noise | AWGN with sigma = 15, 25, 50 on the 0 to 255 scale (std = sigma/255 on [0,1]). One deterministic noisy realization per image and sigma, reused by every method. |
| Training noise | Online AWGN, sigma sampled uniformly from **[15, 50]** per patch. The DL models are **blind** (they are not told sigma). Test sigma 15 and 50 sit at the ends of this range; report that as a limitation. |
| Noisy-image rule | `noisy = clip(round(255*clean + N(0, sigma^2)), 0, 255) / 255`, applied **identically in training and test**, so the networks see the same kind of input as the saved test files. Check the pre-clipping mean and std. |
| Output rule | Every method outputs float in [0,1] with explicit clipping. For metrics all outputs are rounded to 8-bit first. |
| DL recipe | 64 x 64 patches, flips and 90-degree rotations, Adam, L1 pixel loss, compact networks. Same patches or steps, validation images, checkpoint rule and noise sampler for both networks. Parameter counts reported. **2 training seeds** per configuration. |
| Core ablation | **P** = pixel loss only. **G** = pixel loss + lambda x Sobel gradient loss (L1 on Sobel x and y responses, kernels divided by 8). Lambda is chosen by a short pilot on the validation images only (grid 0.1, 0.5, 1.0) and then fixed. Each person runs P and G on their own network, 2 seeds each. |
| Fairness rule | Classical methods are tuned **per sigma** on the calibration images (they are told sigma); the DL models are blind. State this next to every comparison. A sigma-specific DL model is an optional extension. |
| Metrics | PSNR (fixed range, quantized output), SSIM (Gaussian window 11, sigma 1.5, data range 1), runtime, region metrics, gradient error, optional Canny agreement. |
| Region masks | Built from the **clean** image with a **Gaussian-derivative** gradient (sigma 1 px) and local standard deviation (7 x 7). Defaults, fixed on the calibration images by percentile: smooth = gradient and local std both at or below their 40th percentile; edge = gradient at or above the 90th percentile; optional texture = local std at or above the 60th percentile and not edge. Report pixel counts and test sensitivity by moving the percentiles by 10 points. |
| Gradient error | Headline: Gaussian-derivative gradient (sigma 1 px), mean absolute difference of gradient magnitude, globally and inside edge regions. A Sobel-based version may be reported only as a clearly labelled "same operator as the G loss" number, because G is trained on Sobel gradients and would be favoured by construction. |
| Statistics | Images are the experimental unit. Paired differences per image, bootstrap 95% confidence intervals over images, seed spread for DL models. |

## 2. Paper adaptation
Gingichashvili and Lischinski, *Evaluation and Comparison of Edge-Preserving Filters* (2020), https://arxiv.org/abs/2012.13778. Read the paper itself and write a one-page note:
1. Question and task
2. Methods and evaluation descriptors
3. Parameter fairness and the idea of comparing at an equivalent degree of smoothing
4. Findings and limitations (it concerns edge-preserving smoothing operators in computational photography, not denoising as such)
5. Ideas we borrow
6. How our study differs

Our study is an adaptation and extension, not an exact reproduction.

## 3. Shared input and noise pipeline
Contracts:
```python
denoise(image_float_0_1, config) -> image_float_0_1          # classical
denoise_dl(image_float_0_1, model_config) -> image_float_0_1  # learned, blind: must not use sigma
```
- Input: 2-D grayscale float32 in [0,1]. Output: same shape, finite values, explicit clipping. Never modify clean or noisy inputs in place.
- Noise: AWGN with std `sigma_255/255`, deterministic seed per image and sigma, identical noisy realization for all methods, then the noisy-image rule from the frozen protocol (round and clip to 8-bit). Verify mean and std before clipping and repeatability.
- Manifest with image ID, split, seed, sigma, preprocessing.
- The same generator, with the same rule, is used online in training (sigma uniform in [15, 50]).
- Existing `filters/` and `tune.py` operate on the 0 to 255 scale: wrap them so the public interface is [0,1] and document the unit of every parameter.

## 4. Gaussian baseline
Concept: spatial weighted averaging with a Gaussian kernel; suppresses fluctuations but may blur edges and fine detail. Reuse the existing from-scratch separable implementation. Document kernel sigma and radius, border mode, parameter grid. Test constant-image preservation, shape, finite values, range, and agreement with a reference implementation (testing only). Tune per sigma on the 8 calibration images.

## 5. Non-Local Means
Concept: estimate each pixel by a weighted average of pixels whose surrounding patches look similar. A library implementation is acceptable if labelled; for example scikit-image `denoise_nl_means(image, patch_size, patch_distance, h, fast_mode, sigma)`. Document patch size, search window, filtering strength h, the noise-sigma argument, border handling, library version. Use a modest declared grid; tune per sigma on calibration images with a fixed selection rule (extend `tune.py` with an NLM search space). Watch for rare structures, heavy noise, patch artifacts and runtime.

## 6. DnCNN (your DL model)
Train a compact DnCNN-style residual denoiser: predict the noise residual r(y) and reconstruct y minus r(y). Reference: Zhang et al., *Beyond a Gaussian Denoiser* (2017), https://arxiv.org/abs/1608.03981. Starting point: Conv+ReLU, then (D-2) blocks of Conv+BatchNorm+ReLU, then Conv, 3 x 3 kernels, so the receptive field is (2D+1) x (2D+1). Size D and C (the original uses 17 and 64) by a timing test on Kaggle; label a compact version accurately instead of claiming canonical reproduction.

### Shared training rules (identical for Person B's U-Net)
- Train on **BSD400**: 380 train and 20 validation images, split **before** patch extraction; never train on or select with BSD68.
- 64 x 64 patches, flips and 90-degree rotations, online AWGN with sigma uniform in [15, 50] and the noisy-image rule, L1 pixel loss, Adam, same steps or patches, same validation images, same checkpoint rule (best validation PSNR averaged over sigma 15, 25, 50).
- Record depth and width, kernels, padding, normalization, optimizer, learning rate, batch size, steps, seed, parameter count, training time, hardware.
- Save config, curves and checkpoints regularly; Kaggle sessions are interrupted, so make training resumable.

### Core ablation: P versus G
- **P:** L1 pixel loss. **G:** L1 pixel loss + lambda x L1 Sobel gradient loss (Sobel kernels divided by 8, replicate padding, x and y responses summed).
- Choose lambda by short pilot runs (about 10 to 20 percent of the budget) with the grid 0.1, 0.5, 1.0, using **validation PSNR and validation edge-region PSNR only**; write the rule down before looking at test data; then fix lambda.
- Train P and G with identical initialization seeds, data order, noise draws and steps; **2 seeds each** (if compute is short, a single G seed is the minimum).
- Unit tests for the loss: zero for identical images; Sobel of a linear ramp gives the expected constant; gradients reach the network.

### Acceptance tests
1. Overfit a tiny batch as a pipeline check, then discard that checkpoint.
2. Loss falls; validation is plausible; output beats noisy input and the Gaussian baseline on validation images.
3. Verify residual sign, shape, finite values, range at all three sigmas.
4. No test-set tuning of any kind.

## 7. Runtime benchmarking (all methods)
Write one harness used for every method. Run **all methods on the same CPU**, including DnCNN and the U-Net, and additionally time the DL models on GPU. State hardware, software versions, warm-up runs, number of repeats and the robust summary (median); state whether model loading and data transfer are included. Report quality versus runtime with the conditions in the caption.

## 8. Statistics and plots (you own these)
- Images are the experimental unit: paired differences per image, mean, median, spread.
- Bootstrap 95% confidence intervals over images; describe the resampling unit. Label results preliminary if the subset is small.
- DL models: report the mean and the seed-to-seed spread; for P versus G compare per-image differences averaged over seeds and say how large the difference is relative to seed spread.
- Plots: **2** PSNR versus sigma, **3** SSIM versus sigma, **4** quality versus runtime, plus **parameter-response curves** for Gaussian and NLM (and for bilateral and TV from B's results). Regional and gradient plots come from B's evaluator.
- Generate figures and tables by script from the CSV; no numbers typed by hand.

## 9. Calibration and test discipline
Freeze the split before tuning; tune Gaussian and NLM only on the 8 calibration images, per sigma; save ranges, rule and final settings; fixed settings on test; no per-image tuning; DnCNN inference settings are never altered based on test outcomes.

## 10. Theory notes (half a page, for the report and viva)
Variance of a linear filter: for a flat region and normalized coefficients h_i, noise variance becomes sigma^2 times the sum of h_i^2, which is below sigma^2, while the same averaging blurs edges; NLM weights from patch distance; residual learning (why predicting noise is easier than predicting the image); receptive field 2D+1; what a gradient (Sobel) loss rewards and what it may cost.

## 11. Handoffs
| When | You give B | B gives you |
| --- | --- | --- |
| Day 2 | Manifest, splits, noise generator, frozen protocol | Agreement on masks and metric conventions |
| Day 5 | Gaussian and NLM wrappers, noisy test sets | PSNR and SSIM functions, bilateral and TV wrappers |
| Day 7 | Frozen calibration settings (Gaussian, NLM), DnCNN P checkpoint | Frozen settings (bilateral, TV), evaluator CSV format |
| Days 8 to 10 | DnCNN P and G outputs and checkpoints, runtime numbers | Full evaluation CSV, masks, regional metrics |
| Days 11 to 12 | Statistics, plots 2 to 4, parameter curves | Equal-smoothing curves, failure gallery, plots 1, 5, 7 |

## 12. Schedule
- Days 1 to 2: paper note, splits, noise pipeline; agree on protocol, DL budget, lambda grid; verify Kaggle GPU.
- Days 3 to 5: Gaussian, NLM, DnCNN code and first P run; tests.
- Days 6 to 7: integrate B's methods; calibrate Gaussian and NLM; lambda pilot; start G and second seeds.
- Days 8 to 10: assist full runs, runtime benchmarking, fix failures.
- Days 11 to 12: statistics, plots, anomalies, representative cases.
- Days 13 to 14: co-write findings, README, slides.

## 13. Deliverables checklist
- [ ] Adaptation card, manifest, noise generator and tests
- [ ] Gaussian and NLM wrappers, settings, grids
- [ ] DnCNN code, configs, curves, checkpoints, P and G with 2 seeds
- [ ] Loss tests and lambda pilot record
- [ ] Runtime harness and results
- [ ] Statistics code, plots 2 to 4, parameter-response curves
- [ ] Theory note

## 14. Viva preparation
Be ready to explain: why averaging reduces noise and blurs detail; why patch similarity exploits redundancy; how DnCNN learns a noise residual and what blind training means; what the Sobel loss adds and why gradient error is measured with a different operator; why paired inputs and a held-out test set matter; how training data, hardware and the known-versus-unknown sigma affect fairness; what AWGN and grayscale leave out about real photographs.

CSV fields:
`image_id, split, seed, sigma_255, method, parameter_config, model_id_optional, loss_variant, train_seed, psnr, ssim, smooth_psnr, edge_psnr, gradient_error, gradient_error_sobel, edge_f1_optional, runtime_seconds, hardware`
