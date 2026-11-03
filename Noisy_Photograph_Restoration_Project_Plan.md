# Project Plan
## An Experimental Study of Classical and Deep Learning-Based Image Denoising: Noise Robustness, Detail Preservation, and Computational Efficiency

### 1. Project framing
**Working title:** *An Experimental Study of Classical and Deep Learning-Based Image Denoising: Noise Robustness, Detail Preservation, and Computational Efficiency: A Quality, Detail-Preservation, and Runtime Study*

**Research question:** How do representative classical and deep learning-based denoising methods trade off noise suppression, edge/texture preservation, and computational cost under controlled noise, and how do these trade-offs vary with noise severity and image content?

**Core additions in this revision (the core idea is unchanged):**
- an **equal-smoothing comparison** (methods compared at an equivalent degree of smoothing, not only at their PSNR-optimal settings);
- a **core ablation**: each trained network with pixel loss only (P) versus pixel loss plus a Sobel gradient loss (G), with 2 seeds;
- everything else that was added in discussion (Perona-Malik, median/impulse noise, Restormer, Canny, Fourier) is **optional**.

### 2. Scope and method selection
**Classical methods (core):**
1. Gaussian smoothing, the simple baseline.
2. Bilateral filtering, edge-aware local smoothing.
3. Total-variation (TV) denoising, regularization-based.
4. Non-Local Means (NLM), non-local patch similarity.

**Modern methods (core):**
5. **Person A: DnCNN**, trained on Kaggle GPU.
6. **Person B: compact residual U-Net denoiser**, trained on Kaggle GPU.

Train compact versions from scratch. If schedule pressure rises, shrink architecture or training budget transparently and protect the overall study. Because the two networks differ in many ways at once (depth, skip connections, normalization), do not attribute differences to "architecture type" alone; report parameter counts and say so.

**Optional only after the core is complete (pick at most one or two):**
- Perona-Malik diffusion (in the course, already implemented) as a seventh method.
- Median filter and an impulse-noise test.
- Restormer as a modern reference (inference with official weights, or the from-scratch training script).
- Canny edge-detector agreement, frequency-band analysis.
- Lambda sweep for the G loss; sigma-specific DL models.
- Wiener or BM3D (not both).

### 3. Research claims and fairness
The study compares methods under a shared controlled protocol. It is not a claim that the chosen classical methods represent all traditional denoising or that one CNN represents all modern methods.

For both trained models, record: exact architecture and variant, source paper, checkpoint provenance, training noise range, input normalization, whether it is noise-level-specific or blind, framework and version, hardware, precision, runtime convention, and any mismatch between training assumptions and the test protocol.

**Fair-comparison statement (put it next to every table):** classical methods are tuned per sigma and know sigma; the DL models are blind and trained on sigma in [15, 50].

### Frozen protocol (identical in all three documents)

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

### 4. Noise model and dataset
- Controlled additive white Gaussian noise (AWGN), sigma = 15, 25, 50 on the 0 to 255 scale.
- Grayscale photographs throughout, to keep comparisons interpretable.
- Test set BSD68 (8 calibration + 60 test images, frozen before tuning). DL training on BSD400 (380 train + 20 validation), split before patch extraction. Describe any deviation as a subset study.
- If a paper checkpoint expects another scale or noise setup, document the conversion and its consequences.

AWGN is a controlled proxy, not a complete model of real camera noise (which may be signal-dependent, processed, compressed, or spatially varying).

### 5. Core hypotheses (test, do not assume)
- Gaussian filtering may suppress noise while blurring fine structures.
- Bilateral filtering may retain strong edges better but can struggle with weak boundaries and texture.
- TV may retain prominent contours but produce staircasing or flatten texture.
- NLM may benefit from repeated patches but cost more runtime and struggle with rare patterns or severe noise.
- DnCNN and the U-Net may offer a strong quality/runtime trade-off when training assumptions match the test noise, but can show domain mismatch (especially at the ends of the training range) or over-smoothing/artifacts.
- The Sobel gradient loss (G) may improve edge-region and gradient metrics, may cost PSNR or amplify noise in smooth regions, or may make no measurable difference.
Results may contradict any of these expectations.

### 6. Experimental protocol
1. **Freeze dataset and splits.** Save IDs, source, preprocessing, attribution (manifest).
2. **Preprocess consistently.** Grayscale float32 [0,1]; originals untouched.
3. **Create paired noise.** Deterministic seed per image and sigma; one noisy realization reused by every method; apply the noisy-image rule from the frozen protocol; verify pre-clipping mean and std.
4. **Calibration only.** Tune classical parameters per sigma on the 8 calibration images; freeze settings before testing. Search ranges, rule and final settings are saved.
5. **Train the DL models** under the shared recipe: P and G, 2 seeds each, per network. Choose checkpoints and lambda on the 20 validation images only.
6. **Inference.** Follow each model's input/output conventions; explicit clipping; record padding or cropping.
7. **Evaluate paired outputs** on the 60 test images at all three sigmas.
8. **Validate:** finite values, dimensions, reference integrity, noise pairing, no missing or duplicate rows, logs.
9. **Analyze:** image-level scores, paired differences, spread, equal-smoothing curves, P versus G, runtimes, failure cases.

### 7. Metrics
**Required:** PSNR (dB), SSIM, runtime per image (hardware, framework, warm-up, inclusion rules).

**Region-aware:** smooth-region and edge-region PSNR or MAE with the mask rule in the frozen protocol; gradient error (Gaussian-derivative headline).

**Optional:** Canny precision/recall/F1 against edges detected on the clean image (call it detector agreement, not semantic ground truth; fixed and tuned thresholds); frequency-band analysis (high frequencies contain both real detail and noise; Fourier magnitude discards phase); ZNCC only if defined consistently.

### 8. Core analyses beyond the main table

**8.1 Equal-smoothing comparison.** The paper we adapt makes the point that operators should be compared at an equivalent degree of smoothing, not only each at its own best setting. Procedure:
1. For each classical method sweep its main smoothing parameter (Gaussian sigma, bilateral spatial or range sigma, TV weight, NLM h) over a grid declared in advance, other parameters at their calibrated values.
2. For each setting compute **smooth-region PSNR** (how much noise is removed) and **edge-region PSNR and gradient error** (how much structure is kept), averaged over test images, per sigma.
3. Plot edge-region quality against smooth-region PSNR: one curve per classical method, one point per trained network.
4. Read off the edge-region metric of each method at matched smooth-region PSNR (for example at the Gaussian's calibrated level). A method is "better at equal smoothing" if its curve lies above.

**8.2 Core ablation: P versus G.** Each person trains their network with pixel loss only (P) and with pixel plus Sobel gradient loss (G), 2 seeds each, identical in every other respect (initialization seeds, data order, noise draws, steps). Compare per image: PSNR, SSIM, edge-region PSNR, gradient error. Report seed spread so a P-versus-G difference can be judged against seed noise. This is a cheap ablation: the same code with one extra loss term.

### 9. Runtime and comparison fairness
- Report CPU/GPU and software versions; warm up learned-model inference; time repeated runs and report a robust summary; state whether model loading and data transfer are included.
- Run **all** methods on the same CPU, including the DL models, and additionally report DL on GPU. Do not compare one method's GPU time against another's CPU time as if hardware were equivalent.
- Do not tune on test images or hand-pick best output examples.

### 10. Analysis questions
The primary contribution is a controlled, reproducible investigation, not a leaderboard of six algorithms.
- Which method has the best PSNR/SSIM at each noise level, and do rankings change with noise severity?
- Do overall and regional rankings differ? Which methods preserve edge and texture regions best?
- At equal smoothing, which classical method keeps the most structure, and where do the networks sit relative to those curves?
- Does the Sobel loss help edge regions, and what does it cost elsewhere? Is the effect larger than seed-to-seed variation, and does it hold for both networks?
- What is the quality/runtime trade-off under the stated hardware?
- Do the networks behave differently at sigma 15 and 50 (ends of the training range) than at 25?
- Where do classical and learned priors fail? Are results stable across images or dominated by a few?
- What can controlled grayscale AWGN reveal about real photographs, and what can it not?

### 11. Schedule (2 to 3 weeks)
**Days 1 to 2: freeze scope.** Splits, manifest, frozen protocol (this document), metrics; read the primary paper; verify Kaggle GPU access; build noise generation and I/O; agree the DL budget and lambda grid.

**Days 3 to 5: implement.** A: Gaussian, NLM, noise pipeline, DnCNN code and a first P run. B: bilateral, TV, evaluator core, U-Net code and a first P run. Both: sanity tests.

**Days 6 to 7: integration and calibration.** Smoke tests on a few images; calibrate classical methods per sigma on the 8 calibration images and freeze; lambda pilots on validation; G runs and second seeds start.

**Days 8 to 10: main runs.** Evaluate every method on the 60 test images at all sigmas; save outputs, per-image CSV, configs, logs; check missing runs.

**Days 11 to 12: analysis.** Paired comparisons and uncertainty, region analysis, equal-smoothing curves, P versus G, runtime, parameter-response curves, failure gallery.

**Days 13 to 14: presentation.** Select defensible findings; slides, README, reproduction steps; rehearse a 10-minute talk.

**Optional third week:** one focused extension from the optional list, after the core is complete.

### 12. Division of work (redistributed; each person keeps their methods)

| Item | Owner | Days (rough) |
| --- | --- | --- |
| Paper adaptation note | A | 0.5 |
| Noise pipeline, manifest, seeds, I/O tests | A | 1.5 |
| Gaussian (reuse existing code) and NLM | A | 1.5 |
| DnCNN code and runs (P and G, 2 seeds each) | A | 4.0 |
| Runtime benchmarking harness (all methods) | A | 1.0 |
| Statistics: paired differences, bootstrap CIs, seed spread, P-versus-G tests | A | 1.5 |
| Plots 2 to 4 (PSNR and SSIM versus sigma, quality versus runtime) and parameter-response curves | A | 1.5 |
| Theory notes (linear-filter variance, NLM weights, residual learning, receptive field) | A | 0.5 |
| **Total A** | | **12.0** |
| Bilateral (reuse existing code) and TV | B | 1.5 |
| U-Net code and runs (P and G, 2 seeds each) | B | 4.0 |
| Evaluator core: PSNR, SSIM, CSV, integrity checks | B | 1.5 |
| Region masks, regional metrics, gradient error | B | 2.0 |
| Equal-smoothing analysis | B | 1.5 |
| Failure gallery and plots 1, 5, 7 | B | 1.0 |
| Theory notes (bilateral weights, TV objective and solver, skip connections, Sobel loss) | B | 0.5 |
| **Total B** | | **12.0** |

Both: understand every method, freeze the protocol together, cross-check outputs, co-write conclusions and slides. Existing `filters/` and `tune.py` (Gaussian, bilateral, median, Perona-Malik) can be reused for tuning; extend the search spaces for NLM and TV.

### 13. Shared interface and records
Classical methods:
```python
denoise(image_float_0_1, config) -> image_float_0_1
```
Learned models (blind: the wrapper must not use sigma):
```python
denoise_dl(image_float_0_1, model_config) -> image_float_0_1
```
- Shape preserved, finite values, explicit clipping, no in-place changes; document normalization, padding, runtime.
- Suggested CSV: `image_id, split, seed, sigma_255, method, parameter_config, model_id_optional, loss_variant, train_seed, psnr, ssim, smooth_psnr, edge_psnr, gradient_error, gradient_error_sobel, edge_f1_optional, runtime_seconds, hardware`
- Also save checkpoint identifier or hash, source paper or model card, software versions, hardware, manifest, run logs.

### 14. Sanity checks, reproducibility and cut order
**Sanity yardsticks**
- Noisy-input PSNR should be near the theoretical 24.6, 20.2 and 14.2 dB at sigma 15, 25, 50 (slightly higher because of clipping).
- Existing tuning on 8 calibration crops gave roughly 29.3, 26.3 and 22.6 dB for the best classical method at sigma 15, 25, 50. A working DL model should clearly beat classical results on validation images; if not, suspect a bug (scale, residual sign, normalization mode, too few steps).
- For context only: published DnCNN results on BSD68 are about 31.6, 29.2 and 26.2 dB (Restormer paper, Table 4). A compact, briefly trained network will probably be lower; do not tune toward these numbers on test data.

**Checklist**
- [ ] Test set and splits frozen before tuning; BSD68 never used for training or model selection.
- [ ] Same noisy realization across methods; seeds and preprocessing recorded; noisy-image rule identical in training and test.
- [ ] Classical search ranges and chosen settings saved.
- [ ] Both networks: architecture, parameters, steps, seeds, checkpoint rule, training curves documented.
- [ ] Lambda chosen on validation only; P and G matched in everything else.
- [ ] Runtime conditions stated for every implementation.
- [ ] Correct references, metric range and image scale verified.
- [ ] Failed runs and exclusions logged; conclusions based on per-image outcomes.

**If we slip, cut in this order:** optional items; the second seed of G; Canny; the regional texture class; NLM parameter grid size. **Never cut:** the six-method comparison, region-aware metrics, paired statistics, P versus G for at least one network.

### 15. Paper adaptation
Primary inspiration: Sarah Gingichashvili and Dani Lischinski, *Evaluation and Comparison of Edge-Preserving Filters* (2020), https://arxiv.org/abs/2012.13778. The paper introduces a systematic methodology for evaluating and comparing edge-preserving smoothing operators and a common baseline for equivalent parameter mappings. It is about smoothing operators in computational photography, not denoising as such, so say exactly what transfers.

**Adaptation card**
- Borrow: systematic comparison, comparison at an equivalent degree of smoothing, structural and edge-aware evaluation.
- Adapt: controlled noisy-photograph restoration, quality metrics, regional analysis and runtime.
- Extend: trained DnCNN and compact residual U-Net denoisers; the P-versus-G ablation.
- Change: selected methods, noise protocol, dataset and objectives.
- Do not claim exact reproduction or direct comparability to the paper's results.

### 16. Course alignment
| Course expectation | Where it is covered |
| --- | --- |
| Read papers | Edge-preserving filter evaluation paper; DnCNN; U-Net (architectural adaptation, not an evaluation of denoising in the original paper); NLM, TV, bilateral sources |
| Understand the mathematics | Theory notes by each person: linear-filter variance, bilateral weights, TV objective and solver, NLM weights, residual learning, receptive field, Sobel loss |
| Implement, ideally from scratch | Gaussian and bilateral from scratch (existing code); DnCNN and U-Net written by us in PyTorch; NLM and TV may use a library, labelled as such |
| Datasets | BSD400 (training), BSD68 (test), masks and manifest as curated artifacts |
| Empirical analysis and ablations | Main comparison, equal-smoothing curves, parameter-response curves, P-versus-G ablation with seeds |
| Summarize and explain | 10-minute talk, README, report |

### 17. Ten-minute presentation
1. Application, motivation, question (1 min)
2. Dataset, noise, fair protocol (1.5 min)
3. Classical vs learned method intuition, including the Sobel loss idea (2 min)
4. Main quality, regional and runtime results, equal-smoothing curves (2.5 min)
5. P-versus-G ablation, failure cases and interpretation (2 min)
6. Limitations and conclusion (1 min)

**Deliverables (Moodle):** adaptation card; dataset manifest; code with one-command reproduction; model provenance; configs; per-image CSV; plots; failure gallery; README; report and slides.
