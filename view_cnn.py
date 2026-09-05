"""Draw the detector's own output next to ground truth.

Every number in this repo is an average over 2000 images. An average cannot show
you a failure mode -- it can only tell you one exists. This is the figure that
shows which objects it misses and what a false positive actually looks like.

    python view_cnn.py --ckpt runs/det_hard_none.pt --regime hard --n 12
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")               # headless: no display on this box
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import torch

import common as C
import detector as D
from train_det import MEAN, STD

ROOT = os.path.dirname(os.path.abspath(__file__))
COL = {0: "tab:red", 1: "tab:blue", 2: "tab:green"}


def main(ckpt, regime, n, thr, out):
    c = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = D.Detector(w=c["width"]); model.load_state_dict(c["model"]); model.eval()
    imgs, boxes, meta = C.load_split(ROOT, regime, "val")
    gb = C.group_boxes(boxes, len(imgs))

    x = ((imgs[:n].astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)
    with torch.no_grad():
        hm, wh, off = model(torch.from_numpy(np.ascontiguousarray(x)))
        bx, cl, sc = D.decode(hm, wh, off, k=10)

    cols = 4
    rows = int(np.ceil(n / cols)) * 2
    fig, ax = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.3 * rows))
    ax = np.atleast_2d(ax)
    for i in range(n):
        r, cc = (i // cols) * 2, i % cols
        # left: image + gt (dashed) + prediction (solid)
        a = ax[r, cc]
        a.imshow(imgs[i]); a.set_xticks([]); a.set_yticks([])
        for b in gb[i]:
            a.add_patch(mp.Rectangle((b[1], b[2]), b[3] - b[1], b[4] - b[2],
                                     fill=False, ec="w", lw=1.8, ls="--"))
        k = 0
        for b, cl_, s in zip(bx[i], cl[i], sc[i]):
            if s < thr:
                continue
            k += 1
            a.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                     fill=False, ec=COL[int(cl_)], lw=1.6))
            a.text(b[0], b[1] - 2, f"{C.CLASSES[int(cl_)][:3]} {s:.2f}",
                   color=COL[int(cl_)], fontsize=6.5,
                   bbox=dict(fc="k", alpha=0.55, pad=0.6, lw=0))
        a.set_title(f"{len(gb[i])} gt / {k} pred", fontsize=8)
        # right: the heatmap the boxes were read off, max over classes
        a = ax[r + 1, cc]
        a.imshow(hm[i].max(0).values.numpy(), cmap="magma", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([])
        a.set_title("centre heatmap (max over classes)", fontsize=7)
    for j in range(n, (rows // 2) * cols):
        ax[(j // cols) * 2, j % cols].axis("off")
        ax[(j // cols) * 2 + 1, j % cols].axis("off")
    fig.suptitle(f"{regime} val -- dashed white = ground truth, solid = prediction "
                 f"(score > {thr})", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", default="runs/det_hard_none.pt")
    a.add_argument("--regime", choices=C.REGIMES, default="hard")
    a.add_argument("--n", type=int, default=8)
    a.add_argument("--thr", type=float, default=0.3)
    a.add_argument("--out", default="out/cnn_detections.png")
    a = a.parse_args()
    main(a.ckpt, a.regime, a.n, a.thr, a.out)
