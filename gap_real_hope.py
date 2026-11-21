"""Where the sim-to-real gap sits on HOPE, and what test-time fixes buy.

Step 7 ruled out scale, the per-image cap and the class label. This script
measures the remaining image-level differences between the sim validation
images and the 50 real HOPE crops (per-channel mean/std, saturation, Laplacian
variance as a sharpness/texture proxy), then re-scores both detectors after
test-time corrections that move the real photos toward the sim statistics:

  colour  per-image, per-channel mean/std matched to the model's own sim set
  blur    Gaussian blur, sigma picked from a fixed grid so the median Laplacian
          variance comes closest to the sim set's (image statistics only)
  both    colour after blur

Every correction is calibrated on image statistics, never on labels or mAP, so
the rows are not tuned on the test photos.

    python gap_real_hope.py --root /path/to/hope/val --out runs/real_hope_gap.json
"""
import argparse
import json

import cv2
import numpy as np
import torch

import ap as AP
import common as C
import detector as D
import eval_real_hope as E
from train_det import predict

SIGMAS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
N_SIM = 500                                   # sim val images used for the statistics


def lap_var(imgs):
    return np.array([cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for x in imgs])


def stats(imgs):
    f = np.asarray(imgs).reshape(-1, 3).astype(np.float64)
    sat = [cv2.cvtColor(x, cv2.COLOR_RGB2HSV)[..., 1].mean() for x in imgs]
    return {"mean_rgb": [round(float(v), 1) for v in f.mean(0)],
            "std_rgb": [round(float(v), 1) for v in f.std(0)],
            "saturation_mean": round(float(np.mean(sat)), 1),
            "laplacian_var_median": round(float(np.median(lap_var(imgs))), 1)}


def match_colour(imgs, mean, std):
    """Shift and scale each image's channels to the target mean/std (RGB, 0..255)."""
    x = imgs.astype(np.float64)
    mu = x.mean(axis=(1, 2), keepdims=True)
    sd = x.std(axis=(1, 2), keepdims=True) + 1e-6
    y = (x - mu) / sd * np.asarray(std) + np.asarray(mean)
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def blur(imgs, sigma):
    return np.stack([cv2.GaussianBlur(x, (0, 0), sigma) for x in imgs])


def pick_sigma(imgs, target_lap):
    """Grid sigma whose blurred median Laplacian variance is closest to target (log scale)."""
    err = [abs(np.log(np.median(lap_var(blur(imgs, s)))) - np.log(target_lap)) for s in SIGMAS]
    return SIGMAS[int(np.argmin(err))]


def score(model, imgs, gts, ign, k=10):
    dets = predict(model, imgs, E.IMG_SIZE, k=k)
    r = AP.evaluate(dets, gts, gt_ignore=ign)
    one = lambda x: np.c_[x[:, :1], np.zeros(len(x)), x[:, 2:]]
    ra = AP.evaluate(one(dets), one(gts), n_classes=1, gt_ignore=ign)
    return {"mAP": round(float(r["mAP"]), 4), "AP50": round(float(r["AP50"]), 4),
            "agnostic_mAP": round(float(ra["mAP"]), 4),
            "recall50_box": round(float(r["recall"][0, C.CLS_ID["box"]]), 4),
            "recall50_cylinder": round(float(r["recall"][0, C.CLS_ID["cylinder"]]), 4)}


def main(root, out):
    imgs, gts, ign, _ = E.load(root)
    res = {"real": stats(imgs), "n_sim": N_SIM, "models": {}}
    for regime in ("easy", "hard"):
        sim = np.asarray(np.load(f"data/{regime}/val_images.npy", mmap_mode="r")[:N_SIM])
        s = stats(sim)
        sigma = pick_sigma(imgs, s["laplacian_var_median"])
        ck = torch.load(f"runs/det_{regime}_none.pt", map_location="cpu", weights_only=False)
        model = D.build(ck)
        model.load_state_dict(ck["model"])
        variants = {"raw": imgs, "colour": match_colour(imgs, s["mean_rgb"], s["std_rgb"]),
                    "blur": blur(imgs, sigma)}
        variants["both"] = match_colour(variants["blur"], s["mean_rgb"], s["std_rgb"])
        row = {"sim": s, "blur_sigma": sigma,
               "blurred_laplacian_var_median": round(float(np.median(lap_var(variants["blur"]))), 1),
               "rows": {}}
        for name, x in variants.items():
            row["rows"][name] = score(model, x, gts, ign)
            print(regime, name, row["rows"][name])
        res["models"][regime] = row
    print(json.dumps({"real": res["real"], **{k: v["sim"] for k, v in res["models"].items()}}))
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--root", required=True, help="unzipped hope/val")
    a.add_argument("--out", default="")
    args = a.parse_args()
    torch.set_num_threads(1)
    main(args.root, args.out)
