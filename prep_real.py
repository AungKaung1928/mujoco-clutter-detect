"""Real-camera step 1: turn phone photos into images the detector was trained on.

The renderer's `tilt` camera sees 45 degrees on both axes of a 192 x 192 square.
A phone main camera sees about 53 degrees across the short side of a 4:3 frame,
so a resize alone would show every object smaller than the detector has ever
seen it. The fix is geometric, not a guess: crop the centred square whose field
of view is 45 degrees, then resize. The angle comes from the photo's own EXIF
35 mm-equivalent focal length, so a different phone gets a different crop.

35 mm-equivalent is defined on the diagonal of a 36 x 24 mm frame (43.27 mm),
whatever the photo's aspect ratio. The short side of the equivalent frame is
therefore 43.27 / sqrt(1 + aspect^2): 25.96 mm at 4:3, not the 24 mm of 3:2.
Using 24 on a phone photo would understate its field of view by four degrees.

The output is PNG written by OpenCV, which carries no EXIF at all, so GPS and
device metadata never reach the repo. Raw photos stay in data/ (gitignored).

    python prep_real.py --src data/real_raw --out data/real/png
"""
import argparse
import glob
import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

SIM_FOV_DEG = 45.0          # scene.xml, camera "tilt", fovy
IMG_SIZE = 192              # the detector's input, data/*/meta.json
DIAG_35MM = math.hypot(36.0, 24.0)
EXIF_IFD = 0x8769
TAG_FOCAL_35MM = 41989      # FocalLengthIn35mmFilm
FALLBACK_FOCAL_35MM = 26.0  # typical phone main camera; flagged in the manifest


def short_side_fov_deg(focal_35mm, aspect):
    """Field of view across the short side of a frame with aspect = long/short."""
    short_mm = DIAG_35MM / math.sqrt(1.0 + aspect * aspect)
    return math.degrees(2.0 * math.atan(short_mm / 2.0 / focal_35mm))


def crop_side(short_px, phone_fov_deg, target_fov_deg=SIM_FOV_DEG):
    """Pixels of the centred square that sees target_fov_deg (pinhole model)."""
    ratio = math.tan(math.radians(target_fov_deg / 2)) / math.tan(math.radians(phone_fov_deg / 2))
    if ratio > 1.0:
        raise ValueError(f"phone sees {phone_fov_deg:.1f} deg, narrower than the "
                         f"sim's {target_fov_deg:.1f}; cropping cannot widen it")
    return round(short_px * ratio)


def focal_35mm(im):
    """EXIF 35 mm-equivalent focal length, or None. Chat apps strip it."""
    f = im.getexif().get_ifd(EXIF_IFD).get(TAG_FOCAL_35MM)
    return float(f) if f else None


def prepare(im, focal, size=IMG_SIZE):
    """PIL image as shot -> (size x size x 3 uint8 RGB, info dict)."""
    im = ImageOps.exif_transpose(im).convert("RGB")   # pixels upright, as the photographer saw them
    w, h = im.size
    short, long_ = min(w, h), max(w, h)
    fov = short_side_fov_deg(focal, long_ / short)
    s = crop_side(short, fov)
    x0, y0 = (w - s) // 2, (h - s) // 2
    sq = np.asarray(im.crop((x0, y0, x0 + s, y0 + s)))
    out = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    return out, {"orig_wh": [w, h], "landscape": w >= h, "phone_fov_deg": round(fov, 2),
                 "crop_px": s, "crop_xy": [x0, y0]}


def main(src, out_dir, fallback):
    paths = sorted(p for p in glob.glob(os.path.join(src, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not paths:
        raise SystemExit(f"no photos in {src}")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            f = focal_35mm(im)
            img, info = prepare(im, f if f else fallback)
        name = f"{i:04d}.png"
        cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        info.update(out=name, src=os.path.basename(p),
                    focal_35mm=f if f else fallback, focal_from_exif=bool(f))
        manifest.append(info)
        flags = ("" if f else "  NO EXIF FOCAL, fallback") + ("" if info["landscape"] else "  PORTRAIT")
        print(f"{name}  {os.path.basename(p)}  fov {info['phone_fov_deg']:.1f}  "
              f"crop {info['crop_px']} px{flags}")
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    n_fb = sum(not m["focal_from_exif"] for m in manifest)
    print(f"{len(manifest)} images -> {out_dir}; {n_fb} used the fallback focal length")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--src", default="data/real_raw")
    a.add_argument("--out", default="data/real/png")
    a.add_argument("--fallback-focal", type=float, default=FALLBACK_FOCAL_35MM)
    args = a.parse_args()
    main(args.src, args.out, args.fallback_focal)
