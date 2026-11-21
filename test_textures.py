"""gen_dataset.label_images: the texl texture bank (no rendering, no GL).

Run:  python test_textures.py
"""
import sys

import cv2
import numpy as np

import gen_dataset as GD

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:52s} {detail}")
    if not ok:
        FAILED.append(name)


a = GD.label_images(seed=7, n=4)
b = GD.label_images(seed=7, n=4)
ims = [cv2.imread(p) for p in a]
check("n files, 64 x 64 RGB", len(ims) == 4 and all(i.shape == (64, 64, 3) for i in ims))
check("same seed, same images", all(np.array_equal(cv2.imread(p), cv2.imread(q)) for p, q in zip(a, b)))
check("every channel 0 or 255 except anti-aliased text edges",
      all(np.isin(i, (0, 255)).mean() > 0.9 for i in ims), f"{min(np.isin(i, (0, 255)).mean() for i in ims):.3f}")
lap = [cv2.Laplacian(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for i in ims]
check("high-contrast detail on every image", min(lap) > 1000, f"min Laplacian variance {min(lap):.0f}")
print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
sys.exit(1 if FAILED else 0)
