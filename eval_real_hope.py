"""Sim-trained detector on real photos: HOPE-Image validation set (BOP format).

HOPE (Tyree et al., IROS 2022, arXiv 2203.05701) has 50 labelled real RGB images
of 28 toy groceries in 10 scenes, RealSense D415, 1920x1080, CC-BY-SA-4.0, from
huggingface.co/datasets/bop-benchmark/hope (hope_val_realsense.zip + hope_base.zip).
The images are not in this repo; pass the unzipped `hope/val` folder.

Classes. The detector knows box / cylinder / sphere. Canned goods are cylinders,
cartons-on-the-shelf boxes are boxes; that split is by product type from the
dataset's object names (hope/dataset_info.md), an assumption, not a label.
Bottles, the milk and juice cartons and the yogurt cup fit neither and are
IGNORED: a detection on them is neither a hit nor a miss (COCO-style ignore,
added once per class so a wrong class label is not punished either).

Geometry. Same rule as prep_real.py: crop the centred square the sim camera
would see (45 deg), resize to 192 px. At fy = 1387 px that square is 1150 px,
taller than the 1080 px frame, so the crop is the full 1080 px height (42.5 deg)
and objects come out 6 % larger than at 45 deg. The camera height and tilt are
not the sim's; nothing corrects for that.

    python eval_real_hope.py --root /path/to/hope/val --out runs/real_hope.json
"""
import argparse
import glob
import json
import math
import os

import cv2
import numpy as np
import torch

import ap as AP
import common as C
import detector as D
from train_det import predict

SIM_FOV_DEG = 45.0
IMG_SIZE = 192
BOX_IDS = {3, 5, 6, 8, 9, 12, 22, 23, 25}          # Butter ... Spaghetti: cardboard boxes
CAN_IDS = {1, 4, 7, 10, 15, 19, 20, 21, 26, 27}     # AlphabetSoup ... Tuna: cans
MIN_VISIB = 0.1                                     # less visible than this: ignored


def crop_window(w, h, fy, fov_deg=SIM_FOV_DEG):
    """Centred square that sees fov_deg vertically, clipped to the frame."""
    side = min(round(2 * fy * math.tan(math.radians(fov_deg / 2))), w, h)
    return (w - side) // 2, (h - side) // 2, side


def to_crop(bbox_xywh, x0, y0, side, size=IMG_SIZE):
    """BOP [x, y, w, h] in the full frame -> [x0, y0, x1, y1] in the resized crop, or None."""
    x, y, bw, bh = bbox_xywh
    a = np.clip([x - x0, y - y0, x + bw - x0, y + bh - y0], 0, side)
    if a[2] - a[0] < 2 or a[3] - a[1] < 2:
        return None
    return a * size / side


def load(root):
    imgs, gts, ign, meta = [], [], [], []
    for scene in sorted(glob.glob(os.path.join(root, "*"))):
        gt = json.load(open(os.path.join(scene, "scene_gt.json")))
        info = json.load(open(os.path.join(scene, "scene_gt_info.json")))
        cam = json.load(open(os.path.join(scene, "scene_camera.json")))
        for k in sorted(gt, key=int):
            rgb = cv2.cvtColor(cv2.imread(os.path.join(scene, "rgb", f"{int(k):06d}.png")), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            x0, y0, side = crop_window(w, h, cam[k]["cam_K"][4])
            i = len(imgs)
            imgs.append(cv2.resize(rgb[y0:y0 + side, x0:x0 + side], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
            meta.append({"scene": os.path.basename(scene), "im": int(k), "crop_px": side})
            for o, inf in zip(gt[k], info[k]):
                b = to_crop(inf["bbox_visib"], x0, y0, side)
                if b is None:
                    continue
                cls = C.CLS_ID["box"] if o["obj_id"] in BOX_IDS else C.CLS_ID["cylinder"] if o["obj_id"] in CAN_IDS else None
                if cls is None or inf["visib_fract"] < MIN_VISIB:
                    for c in (C.CLS_ID["box"], C.CLS_ID["cylinder"]):
                        gts.append([i, c, *b])
                        ign.append(True)
                else:
                    gts.append([i, cls, *b])
                    ign.append(False)
    return np.stack(imgs), np.array(gts, np.float64).reshape(-1, 6), np.array(ign, bool), meta


def scene_bootstrap(dets, moved, gts, ign, meta, n_boot=500, seed=0):
    """95 % interval of class-agnostic mAP minus its chance control, resampling whole scenes
    (photos of one scene share objects and layout, so they are not independent)."""
    one = lambda x: np.c_[x[:, :1], np.zeros(len(x)), x[:, 2:]]
    scenes = sorted({m["scene"] for m in meta})
    by_scene = {sc: [i for i, m in enumerate(meta) if m["scene"] == sc] for sc in scenes}

    def take(x, pick):          # rows of the picked photos, renumbered 0..len(pick)-1
        return np.concatenate([np.c_[np.full((x[:, 0] == i).sum(), j), x[x[:, 0] == i][:, 1:]]
                               for j, i in enumerate(pick)])

    rng = np.random.default_rng(seed)
    diff = []
    for _ in range(n_boot):
        pick = [i for sc in rng.choice(scenes, len(scenes)) for i in by_scene[sc]]
        g = one(take(gts, pick))
        ig = np.concatenate([ign[gts[:, 0] == i] for i in pick])
        diff.append(AP.evaluate(one(take(dets, pick)), g, n_classes=1, gt_ignore=ig)["mAP"]
                    - AP.evaluate(one(take(moved, pick)), g, n_classes=1, gt_ignore=ig)["mAP"])
    return [round(float(x), 4) for x in np.percentile(diff, [2.5, 97.5])]


def main(root, ckpts, out, k=10, boot=500):
    imgs, gts, ign, meta = load(root)
    n_pos = {c: int(((gts[:, 1] == C.CLS_ID[c]) & ~ign).sum()) for c in ("box", "cylinder")}
    sizes = np.sqrt((gts[~ign, 4] - gts[~ign, 2]) * (gts[~ign, 5] - gts[~ign, 3]))
    res = {"images": len(imgs), "objects": n_pos, "ignored_entries": int(ign.sum()),
           "object_px_median": round(float(np.median(sizes)), 1), "crop_px": sorted({m["crop_px"] for m in meta}),
           "models": {}}
    per_img = np.bincount(gts[~ign, 0].astype(int), minlength=len(imgs))
    half = len(imgs) // 2
    res["chance_pairs_same_scene"] = sum(meta[i]["scene"] == meta[(i + half) % len(imgs)]["scene"]
                                         for i in range(len(imgs)))
    res.update(k=k, images_over_k=int((per_img > k).sum()),
               max_recall_at_k=round(float(np.minimum(per_img, k).sum() / per_img.sum()), 3))
    print(f"{len(imgs)} images, objects {n_pos}, ignored entries {int(ign.sum())}, "
          f"median object {res['object_px_median']} px at {IMG_SIZE} px")
    for path in ckpts:
        if path.endswith(".onnx"):          # fp32 or INT8 graph, 1 thread, same decode as torch
            import export_onnx as E
            dets = E.onnx_predict(E.sess(path, 1), imgs, IMG_SIZE, k=k)
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = D.build(ck)
            model.load_state_dict(ck["model"])
            dets = predict(model, imgs, IMG_SIZE, k=k)
        r = AP.evaluate(dets, gts, gt_ignore=ign)
        one = lambda x: np.c_[x[:, :1], np.zeros(len(x)), x[:, 2:]]   # every class -> 0
        ra = AP.evaluate(one(dets), one(gts), n_classes=1, gt_ignore=ign)
        # Chance control: the same boxes scored against a photo of another scene (i -> i + n/2).
        moved = dets.copy()
        moved[:, 0] = (moved[:, 0] + len(imgs) // 2) % len(imgs)
        rc = AP.evaluate(one(moved), one(gts), n_classes=1, gt_ignore=ign)
        row = {"mAP": round(float(r["mAP"]), 4), "AP50": round(float(r["AP50"]), 4),
               "AP_box": round(float(r["ap_per_class"][C.CLS_ID["box"]]), 4),
               "AP_cylinder": round(float(r["ap_per_class"][C.CLS_ID["cylinder"]]), 4),
               "recall50_box": round(float(r["recall"][0, C.CLS_ID["box"]]), 4),
               "recall50_cylinder": round(float(r["recall"][0, C.CLS_ID["cylinder"]]), 4),
               "agnostic_mAP": round(float(ra["mAP"]), 4), "agnostic_AP50": round(float(ra["AP50"]), 4),
               "agnostic_recall50": round(float(ra["recall"][0, 0]), 4),
               "chance_agnostic_recall50": round(float(rc["recall"][0, 0]), 4),
               "chance_agnostic_mAP": round(float(rc["mAP"]), 4),
               "dets_above_0.3_by_class": {c: int(((dets[:, 6] > 0.3) & (dets[:, 1] == i)).sum())
                                           for c, i in C.CLS_ID.items()},
               "ci95_agnostic_mAP_minus_chance_by_scene":
                   scene_bootstrap(dets, moved, gts, ign, meta, boot) if boot else None}
        res["models"][os.path.basename(path)] = row
        print(f"{os.path.basename(path):24s} " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--root", required=True, help="unzipped hope/val")
    a.add_argument("--ckpt", nargs="+", help=".pt checkpoints or .onnx graphs", default=["runs/det_easy_none.pt", "runs/det_hard_none.pt"])
    a.add_argument("--out", default="")
    a.add_argument("--boot", type=int, default=500, help="scene bootstrap draws (0 = skip; slow at --k 100)")
    a.add_argument("--k", type=int, default=10, help="detections kept per image (10 = the sim setting)")
    args = a.parse_args()
    torch.set_num_threads(1)
    main(args.root, args.ckpt, args.out, args.k, args.boot)
