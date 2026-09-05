"""End-to-end checks of ap.py against the real dataset.

The unit tests in test_ap.py prove the arithmetic on cases small enough to do on
paper. They cannot catch a mistake that only appears at scale -- a wrong axis, a
silently dropped image, a class index off by one. These checks use the actual
val split and rely on the fact that a metric with known inputs has known outputs:

  * feed the ground truth back as detections -> AP must be exactly 1.0
  * degrade those detections in a controlled way -> AP must fall predictably
  * feed noise -> AP must be ~0

If any of those is off, the number a detector produces later is meaningless.
"""
import time

import numpy as np

import ap
import common as C

ROOT = "."
rng = np.random.default_rng(0)

imgs, boxes, meta = C.load_split(ROOT, "hard", "val")
n_img = len(imgs)
gts = boxes[:, [0, 1, 2, 3, 4, 5]].astype(np.float64)      # img, cls, x0,y0,x1,y1
vf = boxes[:, 6].astype(np.float64)
print(f"val split: {n_img} images, {len(gts)} objects, "
      f"{len(gts)/n_img:.2f}/img, classes {list(C.CLASSES)}\n")


def as_dets(g, score=None):
    s = np.ones((len(g), 1)) if score is None else np.asarray(score).reshape(-1, 1)
    return np.hstack([g, s])


def run(name, dets, expect=None):
    t0 = time.perf_counter()
    r = ap.evaluate(dets, gts)
    dt = time.perf_counter() - t0
    tag = ""
    if expect is not None:
        ok = abs(r["mAP"] - expect) < 1e-9
        tag = f"   {'ok' if ok else 'FAIL'} (want {expect:.4f})"
    print(f"  {name:38s} mAP {r['mAP']:.4f}   AP50 {r['AP50']:.4f}   "
          f"AP75 {r['AP75']:.4f}   {dt*1e3:6.0f} ms{tag}")
    return r


print("identity and noise")
run("ground truth as detections", as_dets(gts), expect=1.0)

d = gts.copy()
d[:, 1] = rng.permutation(d[:, 1])
run("correct boxes, shuffled classes", as_dets(d))

d = gts.copy()
lo = np.array([0, 0, 0, 0], np.float64)
d[:, 2:6] = rng.uniform(0, meta["img_size"], size=(len(d), 4))
d[:, 2:6] = np.hstack([np.minimum(d[:, 2:4], d[:, 4:6]), np.maximum(d[:, 2:4], d[:, 4:6])])
run("random boxes, correct classes", as_dets(d))

print("\ncontrolled degradation: shift every box by d pixels")
print("  a w-wide box shifted by d has IoU (w-d)/(w+d); median w here is 28 px")
for shift in (0, 1, 2, 4, 7, 12):
    d = gts.copy()
    d[:, [2, 4]] += shift
    pred = (28 - shift) / (28 + shift)
    r = ap.evaluate(as_dets(d), gts)
    print(f"    shift {shift:2d} px   predicted IoU {pred:.3f}   "
          f"mAP {r['mAP']:.4f}   AP50 {r['AP50']:.4f}")

print("\nrecall control: keep only a fraction of the ground truth as detections")
for frac in (1.0, 0.75, 0.5, 0.25):
    k = int(len(gts) * frac)
    sel = rng.permutation(len(gts))[:k]
    r = ap.evaluate(as_dets(gts[sel]), gts)
    print(f"    keep {frac:4.0%}   mAP {r['mAP']:.4f}   "
          f"recall {np.nanmean(r['recall']):.4f}")

print("\nscore ordering: identical detection sets, only the ranking differs")
# ground truth plus an equal number of random false positives. The set of boxes
# is the same in both rows; only which half scores higher changes.
fp = gts.copy()
fp[:, 2:6] = rng.uniform(0, meta["img_size"] - 20, size=(len(fp), 4))
fp[:, 4:6] = fp[:, 2:4] + rng.uniform(8, 30, size=(len(fp), 2))
good_high = np.vstack([as_dets(gts, 0.9 * np.ones(len(gts))),
                       as_dets(fp, 0.1 * np.ones(len(fp)))])
good_low = np.vstack([as_dets(gts, 0.1 * np.ones(len(gts))),
                      as_dets(fp, 0.9 * np.ones(len(fp)))])
r_hi = ap.evaluate(good_high, gts)
r_lo = ap.evaluate(good_low, gts)
print(f"    hits ranked above the misses   mAP {r_hi['mAP']:.4f}   (want 1.0000)")
print(f"    hits ranked below the misses   mAP {r_lo['mAP']:.4f}   (want 0.5000)")
print("    same boxes, same recall, half the AP. A detector whose confidence does")
print("    not rank its own hits above its own misses is scored as if it missed.")

print("\nsize strata (terciles of this dataset, not COCO's absolute 32/96 px)")
scale = ap.box_scale(gts[:, 2:6])
edges = [0.0, *np.percentile(scale, [33.3, 66.7]), 1e9]
d = gts.copy()
d[:, [2, 4]] += 3
strata = ap.evaluate_size_strata(as_dets(d), gts, edges)
for (lo_e, hi_e), r in strata.items():
    n = int(((scale >= lo_e) & (scale < hi_e)).sum())
    print(f"    {lo_e:5.1f}-{min(hi_e, 999):5.1f} px  n={n:5d}   mAP {r['mAP']:.4f}"
          f"   AP50 {r['AP50']:.4f}")
print("    a fixed 3 px shift hurts small boxes most -- IoU is scale-relative,")
print("    which is exactly why a single mAP number hides where a detector fails")

print("\nrecall by visibility (recall, not AP -- see the docstring in ap.py)")
vis_edges = [0.0, 0.5, 0.9, 1.01]
rv = ap.recall_by_visibility(as_dets(gts), gts, vf, vis_edges)
for (lo_e, hi_e), r in rv.items():
    print(f"    visible_frac {lo_e:.2f}-{hi_e:.2f}   n={r['n_gt']:5d}   "
          f"recall@0.5 {r['recall']:.4f}")
