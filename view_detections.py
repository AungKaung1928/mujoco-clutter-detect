"""Draw baseline detections against ground truth.

The tables say the classical pipeline reaches mAP 0.53 on `hard`. This says what
0.53 looks like, which is the part a table cannot show: which objects it finds
cleanly, which it merges, and which classes it confuses.

    python view_detections.py --regime hard --method bgsub+ws
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import baseline_cv as B
import common as C

ROOT = os.path.dirname(os.path.abspath(__file__))
COL = {0: "#e6194b", 1: "#3cb44b", 2: "#4363d8"}


def main(regime, method, n, seed, out):
    tr_i, tr_b, meta = C.load_split(ROOT, regime, "train")
    va_i, va_b, _ = C.load_split(ROOT, regime, "val")
    size = meta["img_size"]

    Xb, Xf, Xi, _ = B.run_split(tr_i, method, size, limit=2000)
    y = B.label_against_gt(Xb, Xi, tr_b[:, 2:6].astype(np.float64),
                           tr_b[:, 0].astype(np.int64), tr_b[:, 1].astype(np.int64))
    clf = B.ShapeClassifier()
    clf.fit(Xf, y)

    per = C.group_boxes(va_b, len(va_i))
    idx = np.random.default_rng(seed).choice(len(va_i), n, replace=False)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.4 * cols, 3.4 * np.ceil(n / cols)))
    for ax, i in zip(np.ravel(axes), idx):
        img = np.asarray(va_i[i])
        ax.imshow(img); ax.axis("off")
        for b in per[i]:
            ax.add_patch(Rectangle((b[1], b[2]), b[3] - b[1], b[4] - b[2],
                                   fill=False, ec="lime", lw=1.8, alpha=0.85))
        bx, ft = B.detect_one(img, method, size)
        if len(bx):
            cls, sc = clf.predict(ft)
            for b, c, s in zip(bx, cls, sc):
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, ec=COL[int(c)], lw=1.2, ls="--"))
                ax.text(b[0], b[3] + 8, f"{C.CLASSES[int(c)][:3]} {s:.2f}",
                        color=COL[int(c)], fontsize=6.5,
                        bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.5))
        ax.set_title(f"#{i}  gt={len(per[i])}  det={len(bx)}", fontsize=7.5)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(f"{regime} / {method}   green solid = ground truth, "
                 f"dashed = detection (colour is predicted class)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--regime", default="hard", choices=C.REGIMES)
    a.add_argument("--method", default="bgsub+ws")
    a.add_argument("--n", type=int, default=8)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--out", default=os.path.join(ROOT, "out", "detections.png"))
    a = a.parse_args()
    main(a.regime, a.method, a.n, a.seed, a.out)
