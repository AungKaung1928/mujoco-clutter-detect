"""Does a sim regime reach the fine detail of the real photos? Check before training on it.

gap_real_hope.py measured the real HOPE crops at median Laplacian variance 1461.5 and
the `hard` sim set at 83.8; blurring the photos down to the sim level did not help, so
the open question is whether training on more detail does. A regime can only test that
if its images carry the detail. This renders a few images per regime (one core, seconds)
and prints the same statistics, plus `hard` under train_det's photometric augmentation
(which adds pixel noise).

Renders go to data/<regime>/detailprobe_* (ignored), never over a training split.

    python render_detail.py --out runs/render_detail.json
"""
import argparse
import json

import numpy as np

import gap_real_hope as G
import gen_dataset as GD
from train_det import photometric

N = 48


def main(out):
    res = {"n": N, "real": json.load(open("runs/real_hope_gap.json"))["real"], "sets": {}}
    hard = np.asarray(np.load("data/hard/val_images.npy", mmap_mode="r")[:N])
    res["sets"]["hard"] = G.stats(hard)
    rng = np.random.default_rng(0)
    res["sets"]["hard_aug_photo"] = G.stats(np.stack([np.rint(photometric(x, rng) * 255).astype(np.uint8) for x in hard]))
    for regime in ("tex", "texd"):
        GD.generate(regime, N, 192, 0, "detailprobe", "tilt")
        res["sets"][regime] = G.stats(np.load(f"data/{regime}/detailprobe_images.npy"))
    for k, v in res["sets"].items():
        print(f"{k:15s} laplacian {v['laplacian_var_median']:7.1f}  saturation {v['saturation_mean']:5.1f}")
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="")
    main(a.parse_args().out)
