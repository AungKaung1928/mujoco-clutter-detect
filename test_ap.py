"""Hand-computable checks on ap.py.

Every expected value below is derived on paper, not captured from a previous run
of this code. A test that records whatever the implementation happened to print
proves only that the implementation is deterministic.

Run:  python test_ap.py
"""
import numpy as np

import ap

FAILED = []


def check(name, got, want, tol=1e-9, note=""):
    ok = (np.isnan(got) and np.isnan(want)) or abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:44s} got {got:.6f}  want {want:.6f}"
          f"{'   ' + note if note else ''}")
    if not ok:
        FAILED.append(name)


def det(img, cls, box, score):
    return [img, cls, *box, score]


def gt(img, cls, box):
    return [img, cls, *box]


B = [0.0, 0.0, 10.0, 10.0]          # the reference box, area 100
B2 = [50.0, 50.0, 60.0, 60.0]       # disjoint from B


def ap1(dets, gts, **kw):
    """AP at IoU 0.5 only, single class -- the number that can be done by hand."""
    r = ap.evaluate(dets, gts, n_classes=1, iou_thrs=[0.5], **kw)
    return float(r["ap"][0, 0])


print("\n101-point interpolation, single class, IoU 0.50")

# 1. one gt, one perfect detection.
check("perfect detection", ap1([det(0, 0, B, 0.9)], [gt(0, 0, B)]), 1.0)

# 2. a false positive that arrives AFTER full recall costs nothing.
#    rc = [1, 1], pr = [1, 0.5]; every recall level samples index 0.
#    This is counter-intuitive and it is correct: AP asks what precision was
#    achievable at each recall level, and recall 1.0 was reached at precision 1.0.
check("fp after full recall",
      ap1([det(0, 0, B, 0.9), det(0, 0, B2, 0.8)], [gt(0, 0, B)]), 1.0,
      note="costs nothing")

# 3. the same false positive BEFORE the hit costs half.
#    rc = [0, 1], pr = [0, 0.5] -> monotonised to [0.5, 0.5].
check("fp before the hit",
      ap1([det(0, 0, B2, 0.9), det(0, 0, B, 0.8)], [gt(0, 0, B)]), 0.5,
      note="same fp, half the AP")

# 4. half the ground truth found, no false positives.
#    rc = [0.5], pr = [1]; 51 of the 101 recall levels are reachable.
check("1 of 2 found", ap1([det(0, 0, B, 0.9)], [gt(0, 0, B), gt(1, 0, B)]),
      51 / 101, note="not 0.5 -- 51/101")

# 5. a duplicate detection is a false positive, and it is only paid for when it
#    outranks a later true positive.
#    tp/fp = T,F,T / F,T,F -> rc = [.5,.5,1], pr = [1,.5,2/3] -> mono [1,2/3,2/3]
#    AP = (51*1 + 50*(2/3)) / 101
check("duplicate outranking a later tp",
      ap1([det(0, 0, B, 0.9), det(0, 0, B, 0.8), det(1, 0, B, 0.7)],
          [gt(0, 0, B), gt(1, 0, B)]),
      (51 * 1.0 + 50 * (2 / 3)) / 101)
check("same two hits, no duplicate",
      ap1([det(0, 0, B, 0.9), det(1, 0, B, 0.7)], [gt(0, 0, B), gt(1, 0, B)]), 1.0)

# 6. the threshold is inclusive. det [0,0,10,5] vs gt [0,0,10,10]:
#    inter 50, union 100, IoU exactly 0.50.
half = [0.0, 0.0, 10.0, 5.0]
check("IoU exactly 0.50 at thr 0.50", ap1([det(0, 0, half, 0.9)], [gt(0, 0, B)]), 1.0)
r = ap.evaluate([det(0, 0, half, 0.9)], [gt(0, 0, B)], n_classes=1, iou_thrs=[0.55])
check("IoU exactly 0.50 at thr 0.55", float(r["ap"][0, 0]), 0.0)

# 7. degenerate inputs.
check("no detections", ap1([], [gt(0, 0, B)]), 0.0)
r = ap.evaluate([det(0, 0, B, 0.9)], [gt(0, 1, B)], n_classes=2, iou_thrs=[0.5])
check("class with no gt is nan not zero", float(r["ap"][0, 0]), float("nan"))
check("mAP skips the absent class", r["mAP"], 0.0,
      note="class 1 missed entirely")

# 8. matching cannot cross images: identical boxes in different images.
check("no cross-image matching",
      ap1([det(0, 0, B, 0.9)], [gt(1, 0, B)]), 0.0)

print("\nignore semantics (this is what stratified AP depends on)")

# Two ground truths, one of them outside the stratum. Two detections, one on
# each, and the one on the ignored object scores HIGHER.
dets = [det(0, 0, B2, 0.9), det(0, 0, B, 0.8)]
gts = [gt(0, 0, B2), gt(0, 0, B)]
check("ignored gt: det on it is neither hit nor miss",
      ap1(dets, gts, gt_ignore=[True, False]), 1.0)
# The wrong way to do it -- delete the out-of-stratum ground truth instead of
# ignoring it -- turns that detection into a false positive ahead of the real
# hit, which is case 3 above.
check("  same data with the gt deleted instead",
      ap1([det(0, 0, B2, 0.9), det(0, 0, B, 0.8)], [gt(0, 0, B)]), 0.5,
      note="the bug this guards against")

print("\nmonotonisation")
# Without monotonising, rc=[.5,.5,1] pr=[1,.5,2/3] would sample 2/3 -> lower AP.
# Checked implicitly by case 5; assert the raw (non-monotone) value differs.
raw = (51 * 1.0 + 50 * 0.5) / 101
got = ap1([det(0, 0, B, 0.9), det(0, 0, B, 0.8), det(1, 0, B, 0.7)],
          [gt(0, 0, B), gt(1, 0, B)])
print(f"  {'ok  ' if abs(got - raw) > 1e-6 else 'FAIL'}  "
      f"{'monotone AP differs from raw':44s} got {got:.6f}  raw would be {raw:.6f}")
if abs(got - raw) <= 1e-6:
    FAILED.append("monotonisation")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all hand-computed cases pass")
