"""COCO-style average precision, written out rather than imported.

`pycocotools` is one function call away. It is not used here on purpose: every
detection interview question is about what this file does, and calling a library
teaches none of it. The semantics below follow pycocotools closely enough that
the numbers are comparable, and every place they differ is marked.

The three parts people get wrong, in order:

  1. **Matching is greedy, per image, in score order.** A detection may only
     claim a ground truth in its own image, and once claimed the ground truth is
     gone. So a second detection on the same object is a false positive, not a
     duplicate credit. Because matching never crosses images, sorting detections
     globally by score and sorting them per image give identical matches -- which
     is why this file sorts once, globally, and pycocotools sorts per image.

  2. **AP is 101-point interpolated, over a monotonised precision curve.** Raw
     precision is not monotonic; COCO forces it non-increasing from the right
     before sampling. Skip that and AP is systematically low. Sample the raw
     curve at 11 points instead (VOC 2007) and it is systematically high. These
     are different numbers for the same detector, which is why "mAP 0.62" means
     nothing without the convention attached.

  3. **Ignore regions are not the same as deleted regions.** When a stratum is
     evaluated -- small objects only, say -- ground truth outside the stratum is
     marked ignore, not removed. A detection that matches an ignored ground truth
     counts as neither a hit nor a miss; deleting the ground truth instead would
     turn every correct detection of a large object into a false positive and
     the small-object AP would collapse for no real reason.
"""
import numpy as np

import common as C

# COCO's ten thresholds. arange with a float step is fragile at the endpoint, so
# the count is stated explicitly.
IOU_THRS = np.linspace(0.50, 0.95, 10)
REC_THRS = np.linspace(0.00, 1.00, 101)

DET_COLS = ["img_idx", "cls", "x0", "y0", "x1", "y1", "score"]
GT_COLS = ["img_idx", "cls", "x0", "y0", "x1", "y1"]


def _nanmean(a):
    """np.nanmean over an all-NaN slice is a warning and a NaN; a class that is
    absent from the dataset should simply not contribute, silently."""
    a = np.asarray(a, np.float64)
    m = ~np.isnan(a)
    return float(a[m].mean()) if m.any() else float("nan")


def _match(iou, gt_ig, thr):
    """Greedy matching for one (image, class), detections already in score order.

    iou   : (D, G)
    gt_ig : (G,) bool, ground truth to be ignored rather than scored
    returns dt_gt (D,) index of matched gt or -1
    """
    d_n, g_n = iou.shape
    taken = np.zeros(g_n, bool)
    dt_gt = np.full(d_n, -1, np.int64)
    # Non-ignored ground truth is offered first, so a detection only falls back
    # to an ignored one when nothing real is available.
    order = np.concatenate([np.flatnonzero(~gt_ig), np.flatnonzero(gt_ig)])
    for d in range(d_n):
        best, best_iou, seen_real = -1, thr - 1e-12, False
        for g in order:
            if taken[g]:
                continue
            if seen_real and gt_ig[g]:
                break                      # already have a real match; stop before the ignores
            if iou[d, g] < best_iou:
                continue
            best, best_iou = g, iou[d, g]
            seen_real = not gt_ig[g]
        if best >= 0:
            taken[best] = True
            dt_gt[d] = best
    return dt_gt


def _ap_from_curve(tp, fp, n_pos):
    """Cumulative hits/misses in score order -> 101-point interpolated AP."""
    if n_pos == 0:
        return np.nan                      # class absent: undefined, not zero
    if len(tp) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rc = ctp / n_pos
    pr = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
    # monotonise precision from the right
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    idx = np.searchsorted(rc, REC_THRS, side="left")
    q = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0)
    return float(q.mean())


def evaluate(dets, gts, n_classes=len(C.CLASSES), gt_ignore=None, dt_ignore=None,
             iou_thrs=IOU_THRS):
    """dets (D,7), gts (G,6). Returns {'ap': (T, K)} plus convenience summaries.

    gt_ignore (G,) / dt_ignore (D,) implement stratified evaluation. A detection
    flagged in dt_ignore is dropped only if it went unmatched -- matching it to a
    real ground truth still counts, exactly as COCO treats an out-of-area-range
    detection.
    """
    dets = np.asarray(dets, np.float64).reshape(-1, 7)
    gts = np.asarray(gts, np.float64).reshape(-1, 6)
    gt_ig = np.zeros(len(gts), bool) if gt_ignore is None else np.asarray(gt_ignore, bool)
    dt_ig0 = np.zeros(len(dets), bool) if dt_ignore is None else np.asarray(dt_ignore, bool)

    n_t = len(iou_thrs)
    ap = np.full((n_t, n_classes), np.nan)
    rec = np.full((n_t, n_classes), np.nan)

    for k in range(n_classes):
        dk = np.flatnonzero(dets[:, 1] == k)
        gk = np.flatnonzero(gts[:, 1] == k)
        n_pos = int((~gt_ig[gk]).sum())
        if n_pos == 0:
            continue
        # global score order; matching never crosses images so this is equivalent
        # to pycocotools' per-image sort
        dk = dk[np.argsort(-dets[dk, 6], kind="stable")]

        # IoU is independent of the threshold, so build it once per image and
        # reuse it across all ten. This is the whole cost of the metric.
        by_img = {}
        for img in np.unique(np.concatenate([dets[dk, 0], gts[gk, 0]])) if len(dk) else np.unique(gts[gk, 0]):
            di = dk[dets[dk, 0] == img]
            gi = gk[gts[gk, 0] == img]
            iou = C.boxes_iou(dets[di, 2:6], gts[gi, 2:6]) if len(di) and len(gi) \
                else np.zeros((len(di), len(gi)))
            by_img[img] = (di, gi, iou)

        for ti, thr in enumerate(iou_thrs):
            tp = np.zeros(len(dk), bool)
            ig = dt_ig0[dk].copy()
            pos = {d: i for i, d in enumerate(dk)}
            n_hit = 0
            for img, (di, gi, iou) in by_img.items():
                if len(di) == 0:
                    continue
                dt_gt = _match(iou, gt_ig[gi], thr)
                for j, d in enumerate(di):
                    p = pos[d]
                    if dt_gt[j] >= 0:
                        if gt_ig[gi[dt_gt[j]]]:
                            ig[p] = True          # matched an ignored gt: neither hit nor miss
                        else:
                            tp[p] = True
                            n_hit += 1
                    # unmatched keeps whatever dt_ignore said about it
            keep = ~ig
            ap[ti, k] = _ap_from_curve(tp[keep], ~tp[keep], n_pos)
            rec[ti, k] = n_hit / n_pos

    return {
        "ap": ap, "recall": rec, "iou_thrs": np.asarray(iou_thrs),
        "mAP": _nanmean(ap),
        "AP50": _nanmean(ap[0]),
        "AP75": _nanmean(ap[len(iou_thrs) // 2]),
        "ap_per_class": np.array([_nanmean(ap[:, k]) for k in range(n_classes)]),
        "recall_per_class": np.array([_nanmean(rec[:, k]) for k in range(n_classes)]),
    }


def box_scale(boxes):
    """sqrt of box area, in pixels. COCO's absolute small/medium/large cut-offs
    (32 and 96 px) were chosen for ~640 px images; at 192 px every object here
    would be 'small' and the stratification would carry no information. So size
    strata are defined as terciles of this dataset's own distribution."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    return np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None))


def evaluate_size_strata(dets, gts, edges, n_classes=len(C.CLASSES)):
    """Full AP per size band, with COCO ignore semantics on both sides. Valid
    because a detection's size is something the detector actually reports."""
    dets, gts = np.asarray(dets, np.float64), np.asarray(gts, np.float64)
    ds, gs = box_scale(dets[:, 2:6]), box_scale(gts[:, 2:6])
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[(lo, hi)] = evaluate(
            dets, gts, n_classes,
            gt_ignore=~((gs >= lo) & (gs < hi)),
            dt_ignore=~((ds >= lo) & (ds < hi)),
        )
    return out


def recall_by_visibility(dets, gts, visible_frac, edges, iou_thr=0.5,
                         n_classes=len(C.CLASSES)):
    """Recall only -- deliberately not AP.

    A visibility band can be applied to ground truth but not to a detection: the
    detector never reports how occluded it thinks an object was. So unmatched
    detections cannot be attributed to a band, precision is undefined, and any
    'AP for heavily occluded objects' computed this way would be an artefact of
    where the other bands' false positives landed. Recall is well defined and is
    the honest thing to report.
    """
    dets, gts = np.asarray(dets, np.float64), np.asarray(gts, np.float64)
    vf = np.asarray(visible_frac, np.float64)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (vf >= lo) & (vf < hi)
        r = evaluate(dets, gts, n_classes, gt_ignore=~sel, iou_thrs=[iou_thr])
        out[(lo, hi)] = {"recall": float(np.nanmean(r["recall"])), "n_gt": int(sel.sum())}
    return out
