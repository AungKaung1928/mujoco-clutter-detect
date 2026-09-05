#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout.
#
# Tiered on purpose. The first two tiers need no dataset and no trained weights,
# so a reader can check the metric and the target encoding -- the two places a
# silent bug would invalidate every number in the README -- in under a minute.
# The later tiers need data/ (2.9 GB, gitignored) and runs/*.pt (gitignored), and
# say how to regenerate them rather than failing.
set -u
cd "$(dirname "$0")"
source ~/personal/ml/env.sh

hr() { printf '\n=== %s ===\n' "$1"; }
have_data() { [ -f "data/hard/val_images.npy" ]; }
have_ckpt() { [ -f "runs/det_hard_none.pt" ]; }

hr "1/5  AP metric -- 15 hand-computed cases"
# Every expected value derived on paper. A test that records what the code
# printed last time proves only that the code is deterministic.
python test_ap.py || exit 1

hr "2/5  target encoding -- encode/decode are inverses"
# Run before any training: this bug class does not crash, it trains to low loss
# and puts the boxes in the wrong place.
python test_detector.py || exit 1

if ! have_data; then
  hr "3-5/5  skipped -- no dataset"
  cat <<'MSG'
data/ is gitignored (2.9 GB). To regenerate it (~8 min, single-threaded render,
CPU-light -- rendering is llvmpipe and uses one core):

    python gen_dataset.py --regime hard --n 200 --smoke    # check it works first
    python gen_dataset.py --regime hard --n 12000
    python gen_dataset.py --regime easy --n 12000
MSG
  exit 0
fi

hr "3/5  metric end-to-end -- known inputs, known outputs"
python sanity_ap.py || exit 1

hr "4/5  classical baseline on val"
python baseline_cv.py --regime hard --methods bgsub+ws || exit 1

if have_ckpt; then
  hr "5/5  detector on val + qualitative figure"
  python view_cnn.py --ckpt runs/det_hard_none.pt --regime hard --out out/cnn_detections.png || exit 1
  echo "wrote out/cnn_detections.png"
else
  hr "5/5  skipped -- no checkpoint"
  echo "runs/*.pt is gitignored. Retrain with:"
  echo "    nice -n 10 python train_det.py --regime hard --epochs 25 \\"
  echo "        --save runs/det_hard_none.json --ckpt runs/det_hard_none.pt"
  echo "~35 min on 8 threads. See README for the thermal note."
fi
