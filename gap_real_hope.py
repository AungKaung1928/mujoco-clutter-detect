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


IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)


def noise_sigma(im):
    """Immerkaer 1996 (CVIU 64(2), 'Fast Noise Variance Estimation'): white-noise sigma in
    grey levels. Texture leaks into it, so on real photos it is an upper bound."""
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = g.shape
    r = cv2.filter2D(g, -1, IMMERKAER)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).sum() / (6 * (w - 2) * (h - 2)))


def stats(imgs):
    f = np.asarray(imgs).reshape(-1, 3).astype(np.float64)
    sat = [cv2.cvtColor(x, cv2.COLOR_RGB2HSV)[..., 1].mean() for x in imgs]
    return {"mean_rgb": [round(float(v), 1) for v in f.mean(0)],
            "std_rgb": [round(float(v), 1) for v in f.std(0)],
            "saturation_mean": round(float(np.mean(sat)), 1),
            "laplacian_var_median": round(float(np.median(lap_var(imgs))), 1),
            "noise_sigma_median": round(float(np.median([noise_sigma(x) for x in imgs])), 2)}


def obj_bg(imgs, boxes):
    """Median Laplacian variance inside the object boxes and outside them, per image.
    boxes: rows (image, class, x0, y0, x1, y1, ...); ignored objects count as objects."""
    obj, bg = [], []
    for i, im in enumerate(imgs):
        lap = cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
        m = np.zeros(lap.shape, bool)
        for x0, y0, x1, y1 in boxes[boxes[:, 0] == i][:, 2:6].astype(int):
            m[max(y0, 0):y1, max(x0, 0):x1] = True
        if m.any():
            obj.append(lap[m].var())
        if (~m).any():
            bg.append(lap[~m].var())
    return {"laplacian_var_objects_median": round(float(np.median(obj)), 1),
            "laplacian_var_background_median": round(float(np.median(bg)), 1)}


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


def best_iou(dets, gts):
    """Per detection: highest IoU with any labelled or ignored object in its image."""
    out = np.zeros(len(dets))
    for j, r in enumerate(dets):
        g = gts[gts[:, 0] == r[0], 2:6]
        if len(g):
            w = np.clip(np.minimum(r[4], g[:, 2]) - np.maximum(r[2], g[:, 0]), 0, None)
            h = np.clip(np.minimum(r[5], g[:, 3]) - np.maximum(r[3], g[:, 1]), 0, None)
            inter = w * h
            out[j] = (inter / ((r[4] - r[2]) * (r[5] - r[3]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter)).max()
    return out


def confidence(model, sim, imgs, gts, meta, thr=0.3, n_boot=500, seed=0):
    """Is the model silent on the photos or confidently wrong? Score and size of its
    detections, and how many above thr sit on an object (IoU >= 0.5) against the same
    boxes on a photo of another scene (i -> i + n/2), with a scene-level bootstrap."""
    res = {"thr": thr}
    for name, x in (("sim", sim), ("real", imgs)):
        d = predict(model, x, E.IMG_SIZE, k=10)
        top = [d[d[:, 0] == i, 6].max() if (d[:, 0] == i).any() else 0.0 for i in range(len(x))]
        hi = d[d[:, 6] > thr]
        res[name] = {"top_score_median": round(float(np.median(top)), 3),
                     "dets_above_thr_per_image": round(len(hi) / len(x), 2),
                     "det_side_px_median": round(float(np.median(np.sqrt((hi[:, 4] - hi[:, 2]) * (hi[:, 5] - hi[:, 3])))), 1)}
    moved = hi.copy()
    moved[:, 0] = (moved[:, 0] + len(imgs) // 2) % len(imgs)
    hit, chance = best_iou(hi, gts) >= 0.5, best_iou(moved, gts) >= 0.5
    scene = np.array([meta[int(i)]["scene"] for i in hi[:, 0]])
    names = sorted(set(scene))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        m = np.concatenate([np.flatnonzero(scene == names[k]) for k in rng.integers(0, len(names), len(names))])
        diffs.append(hit[m].mean() - chance[m].mean())
    res.update(hit50=round(float(hit.mean()), 3), chance_hit50=round(float(chance.mean()), 3),
               ci95_hit_minus_chance_by_scene=[round(float(v), 3) for v in np.quantile(diffs, [0.025, 0.975])])
    return res


def main(root, out):
    imgs, gts, ign, meta = E.load(root)
    res = {"real": stats(imgs), "real_split": obj_bg(imgs, gts), "n_sim": N_SIM, "models": {}}
    for regime in ("easy", "hard"):
        sim = np.asarray(np.load(f"data/{regime}/val_images.npy", mmap_mode="r")[:N_SIM])
        s = stats(sim)
        sim_boxes = np.load(f"data/{regime}/val_boxes.npy")
        sigma = pick_sigma(imgs, s["laplacian_var_median"])
        ck = torch.load(f"runs/det_{regime}_none.pt", map_location="cpu", weights_only=False)
        model = D.build(ck)
        model.load_state_dict(ck["model"])
        variants = {"raw": imgs, "colour": match_colour(imgs, s["mean_rgb"], s["std_rgb"]),
                    "blur": blur(imgs, sigma)}
        variants["both"] = match_colour(variants["blur"], s["mean_rgb"], s["std_rgb"])
        row = {"sim": s, "sim_split": obj_bg(sim, sim_boxes[sim_boxes[:, 0] < N_SIM]), "blur_sigma": sigma,
               "blurred_laplacian_var_median": round(float(np.median(lap_var(variants["blur"]))), 1),
               "rows": {}}
        for name, x in variants.items():
            row["rows"][name] = score(model, x, gts, ign)
            print(regime, name, row["rows"][name])
        if regime == "hard":
            row["confidence"] = confidence(model, sim, imgs, gts, meta)
            print(row["confidence"])
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
