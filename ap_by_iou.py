"""AP at each IoU threshold, sim val vs real photos, for the scratch detectors.

One table behind the sim-to-real chapter of the study app: the same two
checkpoints scored on `hard/val` (renders) and on the 50 HOPE photos
(eval_real_hope.py), threshold by threshold, mean over classes with positives.

    python ap_by_iou.py --hope /path/to/hope/val --out runs/ap_by_iou.json
"""
import argparse
import json
import os

import numpy as np
import torch

import ap as AP
import common as C
import detector as D
import eval_real_hope as E
from train_det import predict

ROOT = os.path.dirname(os.path.abspath(__file__))


def curve(r):
    return [round(float(np.nanmean(row)), 4) for row in r["ap"]]


def main(hope, ckpts, out):
    va_i, va_b, meta = C.load_split(ROOT, "hard", "val")
    sim_gts = va_b[:, :6].astype(np.float64)
    real_i, real_gts, real_ign, _ = E.load(hope)
    res = {"iou_thrs": [round(float(t), 2) for t in AP.IOU_THRS], "sim_split": "hard/val",
           "sim_images": len(va_i), "real_images": len(real_i), "models": {}}
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = D.build(ck)
        model.load_state_dict(ck["model"])
        sim = AP.evaluate(predict(model, va_i, meta["img_size"]), sim_gts)
        real = AP.evaluate(predict(model, real_i, E.IMG_SIZE), real_gts, gt_ignore=real_ign)
        name = os.path.basename(path).removesuffix(".pt")
        res["models"][name] = {"sim_mAP": round(float(sim["mAP"]), 4), "real_mAP": round(float(real["mAP"]), 4),
                               "sim": curve(sim), "real": curve(real)}
        print(f"{name:16s} sim mAP {sim['mAP']:.4f}  real mAP {real['mAP']:.4f}")
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--hope", required=True, help="unzipped hope/val")
    a.add_argument("--ckpt", nargs="+", default=["runs/det_easy_none.pt", "runs/det_hard_none.pt"])
    a.add_argument("--out", default="")
    args = a.parse_args()
    torch.set_num_threads(1)
    main(args.hope, args.ckpt, args.out)
