# Theory note (Person A, about half a page, for the report and viva)

1. **Linear filters reduce noise variance.** For a flat region y = x + n and normalized coefficients h_i:
   Var(n_hat) = sigma^2 * sum(h_i^2) < sigma^2 for any averaging filter. The same averaging mixes pixels across
   edges, so larger support means less noise but more blur. (Connect to your tuned Gaussian sigma.)
2. **Non-local means.** Weight of pixel q for pixel p = exp(-d(patch_p, patch_q) / h^2); what h controls; why repeated
   structures help and rare structures hurt.
3. **Residual learning (DnCNN).** The network predicts the noise n from y and outputs y - n_hat. Why this is easier than
   predicting x; what BatchNorm does here; receptive field = 2*depth + 1 for 3x3 convolutions.
4. **Gradient (Sobel) loss.** L = L1(pred, clean) + lambda * L1(Sobel(pred), Sobel(clean)). What it rewards (edge position
   and strength) and what it may cost (noise has large gradients, so it can raise residual noise in smooth regions).
   Sobel is a fixed convolution, so the loss is differentiable and needs no extra parameters.
5. **Why the headline gradient-error metric uses a different operator** (Gaussian derivative): measuring Sobel error on a
   Sobel-trained model would favour it by construction.
