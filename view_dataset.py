"""Draw the stored boxes back onto the stored images.

Block 1's version projected labels through the analytic camera model to prove the
pixel<->world convention. Here the labels came straight out of the segmentation
buffer, so what needs proving is different: that the boxes are tight, that the
class colours are right, and that visible_frac actually tracks what is hidden.
Occluded boxes are drawn dashed and annotated with their fraction.

    python view_dataset.py --regime hard --n 12
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import common as C

ROOT = os.path.dirname(os.path.abspath(__file__))
COLORS = {0: "#e6194b", 1: "#3cb44b", 2: "#4363d8"}


def main(regime, split, n, seed, out):
    imgs, boxes, meta = C.load_split(ROOT, regime, split)
    per_img = C.group_boxes(boxes, len(imgs))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs), size=min(n, len(imgs)), replace=False)

    cols = 4
    rows_n = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.1 * cols, 3.1 * rows_n))
    for ax, i in zip(np.ravel(axes), idx):
        ax.imshow(imgs[i])
        for b in per_img[i]:
            cls, x0, y0, x1, y1, vis = int(b[0]), b[1], b[2], b[3], b[4], b[5]
            occluded = vis < 0.9
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=COLORS[cls], lw=1.4,
                                   linestyle="--" if occluded else "-"))
            tag = C.CLASSES[cls][:3]
            if occluded:
                tag += f" {vis:.2f}"
            ax.text(x0, y0 - 2, tag, color=COLORS[cls], fontsize=6.5,
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.6))
        ax.set_title(f"#{i}  n={len(per_img[i])}", fontsize=7)
        ax.axis("off")
    for ax in np.ravel(axes)[len(idx):]:
        ax.axis("off")

    fig.suptitle(f"{regime}/{split}  {meta['img_size']}px  camera '{meta['camera']}'  "
                 f"solid = unoccluded, dashed = visible_frac < 0.9", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=C.REGIMES, default="hard")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "labels.png"))
    a = ap.parse_args()
    main(a.regime, a.split, a.n, a.seed, a.out)
