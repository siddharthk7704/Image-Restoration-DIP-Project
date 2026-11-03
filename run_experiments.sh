#!/usr/bin/env bash
# Person A: the full DL experiment matrix, in order. Edit DEPTH/WIDTH/STEPS after the Day-3 timing test.
# Run on Kaggle (GPU) or any machine with a GPU. Each run is resumable: add  --resume runs/<name>/last.pth
set -e
TRAIN=data/BSD400; DEPTH=17; WIDTH=64; STEPS=20000; PILOT_STEPS=3000
COMMON="--train-dir $TRAIN --depth $DEPTH --width $WIDTH --amp"

# 0) sizing and pipeline check (do these first, then set STEPS)
python train_dl.py $COMMON --time-test --steps $STEPS
python train_dl.py $COMMON --overfit-check

# 1) lambda pilots: same seed, same short schedule, only the loss differs
python train_dl.py $COMMON --lam 0   --seed 0 --steps $PILOT_STEPS --name pilot_P
for L in 0.1 0.5 1.0; do
  python train_dl.py $COMMON --lam $L --seed 0 --steps $PILOT_STEPS --name pilot_G$L
done
python pick_lambda.py --runs runs            # prints lambda* (validation PSNR and edge-region PSNR only); put it below
LAMBDA=0.5                                   # <-- the lambda* printed above, written as printed (e.g. 0.5, or 1 for 1.0)

# 2) final runs: P and G, 2 seeds each
for S in 0 1; do
  python train_dl.py $COMMON --lam 0        --seed $S --steps $STEPS
  python train_dl.py $COMMON --lam $LAMBDA  --seed $S --steps $STEPS
done

# 3) inference on the held-out test images, all sigmas, all runs
for RUN in dncnn_P_s0 dncnn_P_s1 dncnn_G${LAMBDA}_s0 dncnn_G${LAMBDA}_s1; do
  for SIG in 15 25 50; do
    python dl_infer.py --ckpt runs/$RUN/best.pth --noisy-dir data/noisy/BSD68/sigma$SIG --sigma $SIG \
      --method-name $RUN --out-dir outputs --clean-dir data/BSD68 --only-heldout results/tuning/tune_split.json
  done
done
