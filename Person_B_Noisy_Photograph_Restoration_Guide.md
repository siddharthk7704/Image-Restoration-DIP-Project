# Person B Guide
## Bilateral and TV, Residual U-Net, Evaluator, Regional Analysis and Equal-Smoothing Comparison

### Mission
Implement bilateral and total-variation denoising, train a compact residual U-Net, and own the common evaluator and the structure-preservation analysis for:

**An Experimental Study of Classical and Deep Learning-Based Image Denoising: Noise Robustness, Detail Preservation, and Computational Efficiency**

You own:
- Bilateral and TV methods.
- **Compact residual U-Net: implement and train on Kaggle GPU**, pixel loss (P) and pixel + Sobel loss (G), 2 seeds each.
- Evaluator core: PSNR, SSIM, CSV, integrity checks.
- Region masks, regional metrics, gradient error.
- **Equal-smoothing analysis.**
- Failure gallery and plots 1, 5, 7.

Person A owns the noise pipeline, Gaussian, NLM, DnCNN, runtime benchmarking, statistics, plots 2 to 4 and parameter-response curves. Both of you must understand every method.

## 1. Research question
How do classical and modern denoisers trade off noise suppression, edge/texture retention and runtime under controlled AWGN, and how does the trade-off depend on noise level and image content?

Methods: Gaussian, bilateral, TV, NLM, A's DnCNN, your residual U-Net. Noise sigma = 15, 25, 50 on the 0 to 255 scale; grayscale images normalized to [0,1]. AWGN is not a full camera-noise model. **Optional only after the core is complete:** Perona-Malik, median and impulse noise, Restormer reference, Canny agreement, Fourier analysis, lambda sweep, sigma-specific DL models, Wiener or BM3D (not both).

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

## 2. Bilateral filter
Concept: local weighted averaging where weights depend on spatial proximity and intensity similarity. Reuse the existing from-scratch implementation behind the [0,1] interface (its range sigma is on the 0 to 255 scale internally; document the unit). Document spatial scale, range scale, neighbourhood support, border handling, implementation and version. Tune per sigma on the 8 calibration images only, using `tune.py` with a small declared grid. Expected failure cases: weak-contrast boundaries, noisy similarity weights, texture flattening, runtime.

## 3. Total variation (TV) denoising
Concept: minimize a balance between fidelity to the noisy image and a regularizer that favours sparse gradients. State the exact objective and TV convention (isotropic or anisotropic), solver, regularization parameter, stopping criterion and iteration limit. A library implementation is acceptable if labelled; for example scikit-image `denoise_tv_chambolle(image, weight, eps, max_num_iter)`; note that other solvers (for example `denoise_tv_bregman`) use a different weight convention, so never mix them. Discuss: stronger regularization removes noise but flattens texture; staircasing; library parameter conventions. Tune per sigma on calibration data only; log convergence warnings.

## 4. Your DL model: compact residual U-Net
Encoder-decoder with skip connections and a one-channel noise-residual head; restore by subtracting predicted noise. Fix and document depth (for example 3 levels), channel widths, normalization and parameter count. Keep the parameter count in the same range as A's DnCNN and report both. The original U-Net paper (https://arxiv.org/abs/1505.04597) is for biomedical segmentation; this is an architectural adaptation for denoising, not a claim that the paper evaluated this task. Because depth, skip connections and normalization all differ from DnCNN, do not attribute differences to architecture type alone.

Training protocol, core ablation (P versus G with Sobel loss, lambda pilot on validation only, 2 seeds each), acceptance tests and unit tests for the loss are **identical to Person A's DnCNN** (see the frozen protocol above and Person A's guide, section 6). Additional checks for the U-Net: input height and width must be handled consistently (pad to a multiple of the downsampling factor and crop back, and record this).

## 5. Evaluator core
- **PSNR:** fixed peak on [0,1] after 8-bit rounding of outputs. **SSIM:** Gaussian window 11, sigma 1.5, data range 1; document the implementation (own, validated against a reference, or library).
- Compare every method on the same clean image and the same noisy realization.
- Output one CSV row per image, sigma, method, loss variant and train seed (fields at the end).
- Integrity checks: same shape and scale, finite outputs, no in-place changes, correct references, missing or duplicate rows, failures and warnings logged.
- Sanity: noisy-input PSNR near 24.6, 20.2, 14.2 dB at sigma 15, 25, 50.

## 6. Region masks, regional metrics, gradient error
Use the mask rule in the frozen protocol (Gaussian-derivative gradient on the clean image, local standard deviation, percentile thresholds fixed on the calibration images). Report pixel counts per region; test sensitivity by moving the percentiles by 10 points; tiny regions give unstable scores. Do not conflate textured regions with boundaries.

Per region report PSNR or MAE. **Gradient error:** Gaussian-derivative gradient magnitude difference between restored and clean images, globally and in edge regions. A Sobel-based version is reported separately and labelled "same operator as the G loss", because the G variant is trained on Sobel gradients. Treat gradient error as a structural proxy, not a perceptual metric.

Optional: Canny precision/recall/F1 against edges from the clean image (fixed and tuned thresholds, small tolerance, strict one-to-one matching), called detector agreement. Optional: frequency analysis, only after the core is complete.

## 7. Equal-smoothing comparison (you own this)
Following the idea in the paper we adapt: compare operators at an equivalent degree of smoothing.
1. Declare in advance a sweep grid for the main smoothing parameter of each classical method (Gaussian sigma, bilateral spatial or range sigma, TV weight, NLM h), others at calibrated values.
2. For each setting and sigma compute, averaged over test images, **smooth-region PSNR** (noise removed) and **edge-region PSNR and gradient error** (structure kept).
3. Plot edge-region quality against smooth-region PSNR: one curve per classical method, one point for each trained network (P and G separately).
4. Read each method's edge-region metric at matched smooth-region PSNR (for example the Gaussian's calibrated level) and report who is above whom, with image-level uncertainty from A's bootstrap code.
This is descriptive analysis on the test set with a pre-declared grid; it does not tune anything.

## 8. Failure gallery and plots 1, 5, 7
Select examples by defined content criteria or systematic ranking (largest per-image gap between methods, lowest edge-region score), not only visual wins. Cover thin lines, foliage or hair, weak edges, repeated texture and smooth gradients. Plots: **1** clean, noisy and all-method grid; **5** region-wise results; **7** failure gallery.

## 9. Compare classical and modern methods carefully
Learned networks are trained models; classical methods use explicit priors. Two networks cannot represent all modern denoisers. State the fair-comparison rule next to every table (classical tuned per sigma and told sigma; DL blind, trained on sigma 15 to 50, with 15 and 50 at the ends of the range). If the CNN runs on GPU and classical methods on CPU, separate quality from timing conclusions.

## 10. Interpretation questions
- Which method is best per noise level, and do overall and regional rankings differ?
- At equal smoothing, which classical method keeps the most structure, and where do the networks sit?
- Does G help edge regions, and what does it cost in smooth regions? Is the effect larger than seed spread, for both networks?
- Which failures appear and which algorithmic assumptions explain them?
- Are conclusions stable across test images? What can grayscale AWGN reveal and not reveal about real photographs?
Expected results are hypotheses, not required outcomes.

## 11. Theory notes (half a page)
Bilateral weights (spatial times range Gaussian, normalized per pixel); the TV objective (data term plus lambda times total variation), what isotropic versus anisotropic means, why staircasing appears; why skip connections help preserve detail in an encoder-decoder; why measuring Sobel error on a Sobel-trained model is circular.

## 12. Handoffs
| When | You give A | A gives you |
| --- | --- | --- |
| Day 2 | Agreement on masks and metric conventions | Manifest, splits, noise generator, frozen protocol |
| Day 5 | PSNR and SSIM functions, bilateral and TV wrappers | Gaussian and NLM wrappers, noisy test sets |
| Day 7 | Frozen settings (bilateral, TV), evaluator CSV format, U-Net P checkpoint | Frozen settings (Gaussian, NLM), DnCNN P checkpoint |
| Days 8 to 10 | Full evaluation CSV, masks, regional metrics, U-Net P and G outputs | DnCNN P and G outputs, runtime numbers |
| Days 11 to 12 | Equal-smoothing curves, failure gallery, plots 1, 5, 7 | Statistics, plots 2 to 4, parameter-response curves |

## 13. Schedule
- Days 1 to 2: agree masks, metrics, splits, protocol, DL budget, lambda grid.
- Days 3 to 5: bilateral, TV, evaluator prototype, U-Net code and first P run.
- Days 6 to 7: integration, calibration and frozen settings; lambda pilot; start G and second seeds.
- Days 8 to 10: full evaluation and validation.
- Days 11 to 12: region analysis, equal-smoothing curves, failure gallery, P versus G.
- Days 13 to 14: conclusions, limitations, slides, rehearsal.
Optional third week: one extension only, after the core is complete.

## 14. Deliverables checklist
- [ ] Bilateral and TV wrappers, grids, frozen settings
- [ ] U-Net code, configs, curves, checkpoints, P and G with 2 seeds; loss tests; lambda pilot record
- [ ] Evaluator core, CSV, integrity checks
- [ ] Masks, thresholds, pixel counts, sensitivity analysis; regional and gradient-error results
- [ ] Equal-smoothing curves and table
- [ ] Failure gallery, plots 1, 5, 7
- [ ] Theory note

## 15. Presentation contribution
Explain bilateral weights and TV regularization; lead the audience through metric limitations, regional findings, the equal-smoothing curves and failure examples. Help answer why a learned model might outperform or underperform the classical methods, what the Sobel loss did and did not change, and what the experiment cannot establish.

CSV fields:
`image_id, split, seed, sigma_255, method, parameter_config, model_id_optional, loss_variant, train_seed, psnr, ssim, smooth_psnr, edge_psnr, gradient_error, gradient_error_sobel, edge_f1_optional, runtime_seconds, hardware`
