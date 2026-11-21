"""eval_real_hope checks; all but the last need no dataset.

Wrong crop geometry or a wrong class map would not crash; it would only make
the real-photo numbers meaningless. Expected values are derived by hand.

Run:  python test_real_hope.py [unzipped hope/val]
With the HOPE path, the real labels go through load() and back into ap.evaluate as
detections: mAP must be 1.0, so a low detector score on the photos is not the harness.
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np

import common as C
import eval_real_hope as E

FAILED = []


def check(name, got, want, tol=1e-6):
    ok = got == want if isinstance(want, (tuple, type(None))) else bool(np.allclose(got, want, atol=tol))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:44s} got {got}  want {want}")
    if not ok:
        FAILED.append(name)


print("\ncrop window: the square the 45 deg sim camera sees, clipped to the frame")
# HOPE D415: 2 * 1386.99 * tan(22.5 deg) = 1149 px > 1080 px frame height -> 1080, centred
check("HOPE 1920x1080, fy 1386.99", E.crop_window(1920, 1080, 1386.99), (420, 0, 1080))
# fy 1000: 2 * 1000 * tan(22.5 deg) = 828.4 -> 828; x0 = (4000 - 828) // 2, y0 = (3000 - 828) // 2
check("4000x3000, fy 1000", E.crop_window(4000, 3000, 1000.0), (1586, 1086, 828))

print("\nBOP [x, y, w, h] -> crop pixels at 192")
s = 192 / 1080
check("inside", E.to_crop([520, 100, 108, 54], 420, 0, 1080), [100 * s, 100 * s, 208 * s, 154 * s])
check("straddles the left edge: clipped", E.to_crop([400, 0, 40, 40], 420, 0, 1080), [0, 0, 20 * s, 40 * s])
check("fully left of the crop: dropped", E.to_crop([0, 0, 100, 100], 420, 0, 1080), None)

print("\nclass map and ignore on a synthetic one-image scene")
with tempfile.TemporaryDirectory() as root:
    sc = os.path.join(root, "000001")
    os.makedirs(os.path.join(sc, "rgb"))
    cv2.imwrite(os.path.join(sc, "rgb", "000000.png"), np.zeros((1080, 1920, 3), np.uint8))
    objs = [(3, 0.9), (1, 0.9), (2, 0.9), (6, 0.05)]   # Butter box, AlphabetSoup can, BBQSauce bottle, box 5 % visible
    bbox = [520, 100, 108, 54]
    json.dump({"0": [{"obj_id": o} for o, _ in objs]}, open(os.path.join(sc, "scene_gt.json"), "w"))
    json.dump({"0": [{"bbox_visib": bbox, "visib_fract": v} for _, v in objs]}, open(os.path.join(sc, "scene_gt_info.json"), "w"))
    json.dump({"0": {"cam_K": [1390.53, 0, 964.957, 0, 1386.99, 522.586, 0, 0, 1]}}, open(os.path.join(sc, "scene_camera.json"), "w"))
    imgs, gts, ign, meta = E.load(root)
check("image resized to 192", tuple(imgs.shape), (1, 192, 192, 3))
check("box counted as box", gts[~ign][:, 1].tolist(), [C.CLS_ID["box"], C.CLS_ID["cylinder"]])
# bottle and the 5 %-visible box: one ignored entry per class each -> 4
check("ignored entries (2 objects x 2 classes)", int(ign.sum()), 4)
check("ignored entries cover both classes", sorted(gts[ign][:, 1].tolist()), [0, 0, 1, 1])
check("box coordinates in crop pixels", gts[0, 2:], [100 * s, 100 * s, 208 * s, 154 * s])

import gap_real_hope as G
print("\ngap_real_hope test-time corrections")
rng = np.random.default_rng(0)
x = rng.integers(0, 256, (2, 32, 32, 3)).astype(np.uint8)
y = G.match_colour(x, [120, 110, 100], [40, 45, 50]).astype(np.float64)
check("colour match: per-image channel means", y.mean(axis=(1, 2)), [[120, 110, 100]] * 2, tol=0.6)
check("colour match: per-image channel stds", y.std(axis=(1, 2)), [[40, 45, 50]] * 2, tol=0.6)
sharp = np.zeros((1, 64, 64, 3), np.uint8)
sharp[:, ::2] = 255                                   # stripes: huge Laplacian variance
check("blur lowers Laplacian variance", bool(G.lap_var(G.blur(sharp, 1.0))[0] < 0.1 * G.lap_var(sharp)[0]), True)
check("sigma pick: target huge -> smallest sigma", G.pick_sigma(x, 1e9), G.SIGMAS[0])
check("sigma pick: target ~0 -> largest sigma", G.pick_sigma(x, 1e-6), G.SIGMAS[-1])
g = np.array([[0, 0, 0, 0, 10, 10], [0, 1, 20, 20, 30, 30], [1, 0, 0, 0, 10, 10]], float)
d = np.array([[0, 0, 0, 0, 10, 5, .9], [0, 0, 20, 20, 30, 30, .9], [2, 0, 0, 0, 10, 10, .9]], float)
# half-overlap 50/100 = 0.5; exact match 1.0; image 2 has no objects -> 0
check("best IoU per detection, own image only", G.best_iou(d, g), [0.5, 1.0, 0.0])

if len(sys.argv) > 1:
    import ap as AP
    print("\nreal HOPE labels fed back as detections")
    imgs, gts, ign, meta = E.load(sys.argv[1])
    dets = np.c_[gts[~ign], np.ones((~ign).sum())]
    check("objects scored (README: 275 boxes + 310 cans)", int((~ign).sum()), 585)
    check("labels as detections: mAP", AP.evaluate(dets, gts, gt_ignore=ign)["mAP"], 1.0)
    on_ignored = np.r_[dets, np.c_[gts[ign], np.full(ign.sum(), 0.99)]]
    check("+ detections on ignored objects: not false positives",
          AP.evaluate(on_ignored, gts, gt_ignore=ign)["mAP"], 1.0)

print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
