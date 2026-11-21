"""Does a textured sim regime reproduce the real-photo failure inside the sim?

The `hard` detector never saw textured objects. Scored on fresh renders of `hard`
(control: same seed as its val split), `tex` and `texl`, it shows whether object
textures alone break it the way the photos do (README Step 7). Inference only, one
thread, renders go to data/<regime>/evalprobe_* (ignored).

    python texture_shift.py --out runs/texture_shift.json
"""
import argparse
import json

import numpy as np
import torch

import ap as AP
import detector as D
import gen_dataset as GD
from train_det import predict

N = 300
SEED = 10_000          # gen_dataset's val seed: the `hard` renders equal the first N val images


def main(out):
    ck = torch.load("runs/det_hard_none.pt", map_location="cpu", weights_only=False)
    model = D.build(ck)
    model.load_state_dict(ck["model"])
    one = lambda x: np.c_[x[:, :1], np.zeros(len(x)), x[:, 2:]]
    res = {"model": "runs/det_hard_none.pt", "n": N, "seed": SEED, "regimes": {}}
    for regime in ("hard", "tex", "texl"):
        GD.generate(regime, N, 192, SEED, "evalprobe", "tilt")
        imgs = np.load(f"data/{regime}/evalprobe_images.npy")
        gts = np.load(f"data/{regime}/evalprobe_boxes.npy")[:, :6].astype(np.float64)
        d = predict(model, imgs, 192, k=10)
        r, ra = AP.evaluate(d, gts), AP.evaluate(one(d), one(gts), n_classes=1)
        res["regimes"][regime] = {"mAP": round(float(r["mAP"]), 4), "agnostic_mAP": round(float(ra["mAP"]), 4)}
        print(regime, res["regimes"][regime])
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="")
    args = a.parse_args()
    torch.set_num_threads(1)
    main(args.out)
