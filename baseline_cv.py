"""Classical detection baseline: segment, connect, describe, classify.

This exists before the network, not after it, because block 1 showed that a
carelessly built baseline inflates the model's apparent win -- there, half the
CNN's advantage turned out to be a single calibration scalar the baseline was
never given. So the baseline here gets everything a classical pipeline can
fairly have, including a fitted classifier.

Two segmenters, and the contrast between them is the point:

  otsu   Global Otsu threshold on greyscale. Reasonable-looking: objects and the
         table differ in brightness, so split the histogram. It works when the
         table happens to be dark and the objects bright. Under `hard`, table
         value is uniform(0.25, 0.85) and object value uniform(0.45, 1.0), so the
         two distributions overlap and the split lands inside the objects. It
         does not raise an error when this happens. It returns a mask.

  bgsub  Estimate the table per image, subtract it, threshold the residual.
         The table is smooth and covers most of the frame, so a heavily
         downscaled median is a good model of it, and the residual is large
         wherever something is sitting on the table regardless of hue or
         brightness.

  bgsub+ws  The same, then split touching objects with a watershed.
         `bgsub` alone was measured first and its masks were clean -- the failure
         was not the threshold. Objects that touch in the image become one
         region: 242 merged regions in 300 images, which caps recall at roughly
         0.53 no matter how the threshold is tuned. The classical answer is a
         distance transform, whose local maxima sit at the centre of each object
         even when the objects share a blob, used as watershed seeds. The
         watershed itself runs on the colour image, where two touching objects of
         different hue have a real edge between them to find.

Scores are fitted, not constant. Step 2a showed that the same box set scores
mAP 1.00 or 0.50 purely on whether the confidence ranks hits above misses, so a
detector that returns 1.0 for everything throws away half its AP for free.
Components are described by nine shape features and classified by multinomial
logistic regression -- linear, on hand-designed features, which is the classical
convention (a linear model on HOG, on SIFT, on shape moments). A fourth class,
background, lets the classifier suppress its own false positives, which is what
a real detector's objectness score does.

Nothing is thresholded before AP is computed. AP integrates over every score
cut-off, so discarding low-confidence detections in advance only removes recall
that the metric would have credited.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

import ap as AP
import common as C

ROOT = os.path.dirname(os.path.abspath(__file__))
BG_CLASS = len(C.CLASSES)          # 3
MIN_AREA = 10                      # px, below the smallest labelled object
FEATS = ["log_area", "scale", "aspect", "extent", "circularity", "solidity",
         "cy_norm", "top_fill", "elongation"]


# ---------------------------------------------------------------- segmentation

def seg_otsu(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu cannot know which side of the split is the object, so take whichever
    # side is the minority -- objects cover ~11% of the frame.
    if m.mean() > 127:
        m = 255 - m
    return m


def _bg_median(img, small=48, k=21):
    s = cv2.resize(img, (small, small), interpolation=cv2.INTER_AREA)
    b = cv2.medianBlur(s, k)
    return cv2.resize(b, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)


def _bg_morph(img, small=48, k=11):
    s = cv2.resize(img, (small, small), interpolation=cv2.INTER_AREA)
    e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    b = cv2.morphologyEx(cv2.morphologyEx(s, cv2.MORPH_OPEN, e), cv2.MORPH_CLOSE, e)
    return cv2.resize(b, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)


def seg_bgsub(img, bg="median"):
    b = _bg_median(img) if bg == "median" else _bg_morph(img)
    d = np.abs(img.astype(np.int16) - b.astype(np.int16)).mean(2).astype(np.uint8)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def split_watershed(img, mask, sigma=4.0, peak=11, min_dist=3.0):
    """Separate touching objects. Returns (label image with labels 2..n+1, n).

    The distance transform of a blob peaks once per object, because the centre of
    an object is further from the outside than the neck where two objects meet.
    Those peaks seed a watershed, and the watershed runs on the colour image,
    where two touching objects of different hue have a real edge to snap to.

    **A region with only one seed is left exactly as it was.** Running the
    watershed over every region instead was measured first and it was worse: it
    redraws the boundary of regions that were already correct, and the fraction of
    objects recovered at IoU >= 0.9 collapsed from 47.3% to 6.3% while the badly
    merged fraction only fell from 38% to 25%. Splitting is a repair, and a repair
    applied to something that is not broken is damage.
    """
    if mask.max() == 0:
        return np.ones_like(mask, np.int32), 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if sigma > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), sigma)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak, peak))
    peaks = ((dist >= cv2.dilate(dist, k)) & (dist > min_dist)).astype(np.uint8)
    n_seed, seeds = cv2.connectedComponents(peaks)
    n_cc, cc = cv2.connectedComponents(mask)

    out = np.zeros(mask.shape, np.int32)
    nxt = 2
    img = np.ascontiguousarray(img)
    for c in range(1, n_cc):
        comp = cc == c
        ids = np.unique(seeds[comp & (seeds > 0)])
        if len(ids) <= 1:
            out[comp] = nxt                      # untouched
            nxt += 1
            continue
        mk = np.zeros(mask.shape, np.int32)
        mk[~comp] = 1                            # everything outside is background
        for j, sid in enumerate(ids):
            mk[(seeds == sid) & comp] = nxt + j
        cv2.watershed(img, mk)
        for j in range(len(ids)):
            out[(mk == nxt + j) & comp] = nxt + j
        nxt += len(ids)
    return out, nxt - 2


def describe_labels(label_img, n, img_size):
    B, F = [], []
    for lab in range(2, n + 2):
        m = ((label_img == lab).astype(np.uint8)) * 255
        if m.max() == 0:
            continue
        b, f = describe(m, img_size)
        if len(b):
            B.append(b); F.append(f)
    if not B:
        return np.zeros((0, 4), np.float32), np.zeros((0, len(FEATS)), np.float32)
    return np.vstack(B), np.vstack(F)


def clean(m):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3)


# ------------------------------------------------------------------- features

def describe(mask, img_size):
    """Connected regions -> (boxes (N,4), features (N,9))."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes, feats = [], []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 2 or h < 2:
            continue
        per = max(cv2.arcLength(c, True), 1e-6)
        hull = max(cv2.contourArea(cv2.convexHull(c)), 1e-6)
        mu = cv2.moments(c)
        if mu["m00"] > 1e-6:
            cxx = mu["mu20"] / mu["m00"]
            cyy = mu["mu02"] / mu["m00"]
            cxy = mu["mu11"] / mu["m00"]
            t = np.sqrt(max((cxx - cyy) ** 2 + 4 * cxy ** 2, 0))
            l1, l2 = (cxx + cyy + t) / 2, (cxx + cyy - t) / 2
            elong = float(np.sqrt(max(l1, 1e-9) / max(l2, 1e-9)))
        else:
            elong = 1.0
        sub = mask[y:y + h, x:x + w] > 0
        half = max(h // 2, 1)
        top = sub[:half].sum() / max(sub.sum(), 1)
        boxes.append([x, y, x + w, y + h])
        feats.append([
            np.log(a + 1.0),
            np.sqrt(a) / img_size,
            h / w,
            a / (w * h),
            4 * np.pi * a / (per ** 2),
            a / hull,
            (y + h / 2) / img_size,
            top,
            min(elong, 20.0),
        ])
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0, len(FEATS)), np.float32)
    return np.asarray(boxes, np.float32), np.asarray(feats, np.float32)


def segment_image(img, method, bg="median"):
    return clean(seg_otsu(img) if method == "otsu" else seg_bgsub(img, bg))


def detect_one(img, method, img_size, bg="median"):
    m = segment_image(img, method, bg)
    if method == "bgsub+ws":
        lab, n = split_watershed(img, m)
        return describe_labels(lab, n, img_size)
    return describe(m, img_size)


def run_split(imgs, method, img_size, bg="median", limit=None):
    """-> boxes (M,4), feats (M,9), img_idx (M,), seconds per image"""
    n = len(imgs) if limit is None else min(limit, len(imgs))
    B, F, I = [], [], []
    t0 = time.perf_counter()
    for i in range(n):
        b, f = detect_one(np.asarray(imgs[i]), method, img_size, bg)
        if len(b):
            B.append(b); F.append(f); I.append(np.full(len(b), i))
    dt = (time.perf_counter() - t0) / n
    if not B:
        return (np.zeros((0, 4), np.float32), np.zeros((0, len(FEATS)), np.float32),
                np.zeros(0, np.int64), dt)
    return np.vstack(B), np.vstack(F), np.concatenate(I), dt


# ------------------------------------------------------------------- labelling

def label_against_gt(boxes, img_idx, gt_boxes, gt_img, gt_cls, thr=0.5):
    """Each region gets the class of the ground truth it overlaps, or background.
    Greedy, one ground truth per region, largest IoU first -- the same rule the
    metric uses, so the classifier is trained on the matching it is scored by."""
    lab = np.full(len(boxes), BG_CLASS, np.int64)
    for img in np.unique(img_idx):
        di = np.flatnonzero(img_idx == img)
        gi = np.flatnonzero(gt_img == img)
        if not len(gi):
            continue
        iou = C.boxes_iou(boxes[di], gt_boxes[gi])
        taken = np.zeros(len(gi), bool)
        for _ in range(min(len(di), len(gi))):
            k = np.argmax(np.where(taken[None, :], -1, iou))
            r, c = divmod(int(k), iou.shape[1])
            if iou[r, c] < thr:
                break
            lab[di[r]] = int(gt_cls[gi[c]])
            taken[c] = True
            iou[r, :] = -1
    return lab


# ------------------------------------------------------------------ classifier

class ShapeClassifier:
    """Multinomial logistic regression on standardised shape features.

    Linear on purpose. A hand-designed feature vector plus a linear decision rule
    is what the classical pipeline is; making it an MLP would quietly turn the
    baseline into a small neural network and destroy the comparison it exists for.
    """

    def __init__(self, n_feat=len(FEATS), n_cls=BG_CLASS + 1):
        self.w = torch.zeros(n_feat, n_cls, requires_grad=True)
        self.b = torch.zeros(n_cls, requires_grad=True)
        self.mu = self.sd = None

    def fit(self, X, y, epochs=400, lr=0.1, seed=0):
        torch.manual_seed(seed)
        X = np.asarray(X, np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.from_numpy((X - self.mu) / self.sd)
        yt = torch.from_numpy(np.asarray(y, np.int64))
        # class-balanced loss: background regions outnumber objects and an
        # unweighted fit would predict background for everything
        cnt = np.bincount(y, minlength=self.b.shape[0]).astype(np.float32)
        wgt = torch.from_numpy((cnt.sum() / np.maximum(cnt, 1)) / len(cnt))
        opt = torch.optim.Adam([self.w, self.b], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xt @ self.w + self.b, yt, weight=wgt)
            loss.backward()
            opt.step()
        return float(loss.item())

    def predict(self, X):
        Xt = torch.from_numpy(((np.asarray(X, np.float32) - self.mu) / self.sd))
        with torch.no_grad():
            p = torch.softmax(Xt @ self.w + self.b, 1).numpy()
        cls = p[:, :BG_CLASS].argmax(1)
        return cls, p[np.arange(len(p)), cls]        # score = P(chosen class)


# ------------------------------------------------------------------ evaluation

def detections(boxes, img_idx, cls, score):
    return np.column_stack([img_idx, cls, boxes, score]).astype(np.float64)


def class_agnostic(d, g):
    d, g = d.copy(), g.copy()
    d[:, 1] = 0
    g[:, 1] = 0
    return AP.evaluate(d, g, n_classes=1)


def report(name, dets, gts, vf, scale_edges, n_img, ms):
    r = AP.evaluate(dets, gts)
    ra = class_agnostic(dets, gts)
    print(f"\n{name}")
    print(f"  detections            {len(dets)} ({len(dets)/n_img:.2f}/img)   "
          f"gt {len(gts)} ({len(gts)/n_img:.2f}/img)")
    print(f"  mAP@[.5:.95]          {r['mAP']:.4f}        AP50 {r['AP50']:.4f}   "
          f"AP75 {r['AP75']:.4f}")
    print(f"  class-agnostic mAP    {ra['mAP']:.4f}        AP50 {ra['AP50']:.4f}"
          f"   <- localisation only")
    print(f"  detection rate @.5    {np.nanmean(ra['recall']):.4f}"
          f"        <- found at all, class ignored")
    print("  per class             " + "   ".join(
        f"{c} {v:.3f}" for c, v in zip(C.CLASSES, r["ap_per_class"])))
    st = AP.evaluate_size_strata(dets, gts, scale_edges)
    print("  by size (terciles)    " + "   ".join(
        f"{lo:.0f}-{min(hi,999):.0f}px {v['mAP']:.3f}" for (lo, hi), v in st.items()))
    rv = AP.recall_by_visibility(dets, gts, vf, [0.0, 0.5, 0.9, 1.01])
    print("  recall by visibility  " + "   ".join(
        f"{lo:.1f}-{hi:.1f} {v['recall']:.3f} (n={v['n_gt']})" for (lo, hi), v in rv.items()))
    print(f"  latency               {ms:.2f} ms/img")
    return {"mAP": r["mAP"], "AP50": r["AP50"], "AP75": r["AP75"],
            "agnostic_mAP": ra["mAP"], "agnostic_AP50": ra["AP50"],
            "det_rate": float(np.nanmean(ra["recall"])),
            "ap_per_class": [float(v) for v in r["ap_per_class"]],
            "size_strata": {f"{lo:.1f}-{hi:.1f}": v["mAP"] for (lo, hi), v in st.items()},
            "vis_recall": {f"{lo:.1f}-{hi:.1f}": v["recall"] for (lo, hi), v in rv.items()},
            "ms_per_img": ms, "n_dets": len(dets)}


def main(regime, methods, bg, n_fit):
    tr_i, tr_b, meta = C.load_split(ROOT, regime, "train")
    va_i, va_b, _ = C.load_split(ROOT, regime, "val")
    size = meta["img_size"]
    gts = va_b[:, :6].astype(np.float64)
    vf = va_b[:, 6].astype(np.float64)
    edges = [0.0, *np.percentile(AP.box_scale(gts[:, 2:6]), [33.3, 66.7]), 1e9]
    out = {}
    print(f"=== regime {regime}, val {len(va_i)} images, {len(gts)} objects ===")
    for method in methods:
        Xb, Xf, Xi, _ = run_split(tr_i, method, size, bg, limit=n_fit)
        y = label_against_gt(Xb, Xi, tr_b[:, 2:6].astype(np.float64),
                             tr_b[:, 0].astype(np.int64), tr_b[:, 1].astype(np.int64))
        clf = ShapeClassifier()
        loss = clf.fit(Xf, y)
        hist = np.bincount(y, minlength=BG_CLASS + 1)
        print(f"\n[{method}] fitted on {n_fit} train images: {len(Xf)} regions, "
              f"labels {dict(zip(list(C.CLASSES) + ['bg'], hist.tolist()))}, loss {loss:.4f}")
        Vb, Vf, Vi, dt = run_split(va_i, method, size, bg)
        cls, sc = clf.predict(Vf)
        out[method] = report(f"[{method}] {regime}", detections(Vb, Vi, cls, sc),
                             gts, vf, edges, len(va_i), dt * 1e3)
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--regime", choices=C.REGIMES, default="hard")
    a.add_argument("--methods", nargs="+", default=["otsu", "bgsub", "bgsub+ws"])
    a.add_argument("--bg", choices=["median", "morph"], default="median")
    a.add_argument("--n-fit", type=int, default=3000)
    a.add_argument("--save", default="")
    a = a.parse_args()
    res = main(a.regime, a.methods, a.bg, a.n_fit)
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        json.dump(res, open(a.save, "w"), indent=2)
        print(f"\nwrote {a.save}")
