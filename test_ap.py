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

print("\nmatching order, COCO's (pycocotools evaluateImg)")
# A det overlapping an ignored gt at IoU 1.0 and a real gt at IoU 0.6 must take
# the real one: real ground truth is offered first, and the scan stops before the
# ignored ones once a real match exists. Listing the ignored gt first in the input
# must not change that.
check("real gt before a better ignored one",
      ap1([det(0, 0, B, 0.9)], [gt(0, 0, B), gt(0, 0, [2.5, 0.0, 12.5, 10.0])],
          gt_ignore=[True, False]), 1.0, note="matching the ignored gt would give 0")
# Equal IoU (0.6) with two gts: the later one wins, as in pycocotools, which
# leaves the earlier gt free for the second det: two hits.
check("equal IoU: later gt wins the tie",
      ap1([det(0, 0, [2.5, 0.0, 12.5, 10.0], 0.9), det(0, 0, B, 0.8)],
          [gt(0, 0, B), gt(0, 0, [5.0, 0.0, 15.0, 10.0])]), 1.0,
      note="first-wins would give 51/101")
# A det on an ignored gt is dropped from the curve, not counted as a hit:
# remaining order FP, TP over 1 positive -> precision 0.5 everywhere.
check("det on ignored gt is not a hit",
      ap1([det(0, 0, B2, 0.9), det(0, 0, [100.0, 100.0, 110.0, 110.0], 0.8), det(0, 0, B, 0.7)],
          [gt(0, 0, B2), gt(0, 0, B)], gt_ignore=[True, False]), 0.5)
# exact monotone value of the case below: rc [.5, .5, 1], pr [1, .5, 2/3] -> [1, 2/3, 2/3]
check("monotone AP, exact",
      ap1([det(0, 0, B, 0.9), det(0, 0, B, 0.8), det(1, 0, B, 0.7)], [gt(0, 0, B), gt(1, 0, B)]),
      (51 * 1.0 + 50 * 2 / 3) / 101)

print("\nsummaries")
r = ap.evaluate([det(0, 0, B, 0.9)], [gt(0, 0, B)], n_classes=2, iou_thrs=[0.5])
check("mAP over present classes only", r["mAP"], 1.0, note="absent class as 0 would give 0.5")
# IoU 0.74 passes the five COCO thresholds 0.50..0.70 and fails 0.75..0.95
r = ap.evaluate([det(0, 0, [0.0, 0.0, 10.0, 7.4], 0.9)], [gt(0, 0, B)], n_classes=1)
check("COCO thresholds: mAP at IoU 0.74", r["mAP"], 0.5)
check("COCO thresholds: AP75 at IoU 0.74", r["AP75"], 0.0)

print("\nstrata")
# size: a large unmatched det must not count against the small band
gts2 = [gt(0, 0, [0.0, 0.0, 4.0, 4.0]), gt(0, 0, [20.0, 20.0, 40.0, 40.0])]
dets2 = [det(0, 0, [100.0, 100.0, 120.0, 120.0], 0.9), det(0, 0, [0.0, 0.0, 4.0, 4.0], 0.8)]
st = ap.evaluate_size_strata(dets2, gts2, [0, 10, 100], n_classes=1)
check("small band ignores a large false positive", float(st[(0, 10)]["ap"][0, 0]), 1.0)
check("large band: missed gt, large fp", float(st[(10, 100)]["ap"][0, 0]), 0.0)
# visibility: bands are [lo, hi); a gt at exactly 0.5 belongs to the upper band
vb = ap.recall_by_visibility([det(0, 0, B, 0.9)], [gt(0, 0, B), gt(0, 0, B2)], [0.5, 0.2],
                             [0.0, 0.5, 1.01], n_classes=1)
check("visibility 0.5 is in [0.5, 1.01)", vb[(0.5, 1.01)]["recall"], 1.0)
check("visibility band [0, 0.5) holds only the missed gt", vb[(0.0, 0.5)]["recall"], 0.0)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all hand-computed cases pass")
