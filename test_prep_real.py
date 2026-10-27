"""prep_real checks that need no photos.

The failure modes do not crash: a field-of-view formula with the wrong sensor
size crops a few percent too much and every object shrinks; an ignored EXIF
orientation feeds the detector a sideways scene; and EXIF that survives into
the output publishes the GPS of the room the photos were taken in. Expected
values are derived by hand from the definitions, not recorded from a run.

Run:  python test_prep_real.py
"""
import io
import json
import math
import os
import tempfile

import numpy as np
from PIL import Image

import prep_real as P


def test_fov_3_2_is_the_textbook_24mm_case():
    # A 3:2 frame is the 36 x 24 mm frame itself, so the short side is 24 mm.
    for f in (18.0, 26.0, 50.0):
        want = math.degrees(2 * math.atan(12.0 / f))
        assert abs(P.short_side_fov_deg(f, 1.5) - want) < 1e-9


def test_fov_4_3_short_side_is_wider_than_3_2():
    # 43.2666 / (5/3) = 25.96 mm, half 12.98 -> 2 atan(12.98/26) = 53.06 deg at 26 mm.
    assert abs(P.short_side_fov_deg(26.0, 4 / 3) - 53.06) < 0.01


def test_crop_side_identities():
    assert P.crop_side(1000, 45.0) == 1000                  # same FOV: keep everything
    assert P.crop_side(1000, 90.0) == round(1000 * math.tan(math.radians(22.5)))  # = 414
    try:
        P.crop_side(1000, 40.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a narrower phone FOV must refuse, not upscale silently")


def _jpeg(img, exif=None):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95, exif=exif if exif is not None else Image.Exif())
    buf.seek(0)
    return Image.open(buf)


def test_centre_survives_and_shape():
    # 4:3 landscape at 26 mm; a 20 px red square in the centre must stay red at the centre.
    a = np.zeros((300, 400, 3), np.uint8)
    a[140:160, 190:210] = (255, 0, 0)
    out, info = P.prepare(_jpeg(Image.fromarray(a)), 26.0)
    assert out.shape == (P.IMG_SIZE, P.IMG_SIZE, 3) and out.dtype == np.uint8
    assert info["crop_px"] == round(300 * math.tan(math.radians(22.5))
                                    / math.tan(math.radians(P.short_side_fov_deg(26.0, 4 / 3) / 2)))
    c = out[P.IMG_SIZE // 2, P.IMG_SIZE // 2]
    assert c[0] > 200 and c[1] < 60 and c[2] < 60, c


def test_exif_orientation_is_applied():
    # Stored landscape with Orientation=6 (rotate 90 CW to view): the scene is portrait.
    a = np.zeros((300, 400, 3), np.uint8)
    a[:, :200] = 255                                  # left half white as stored
    ex = Image.Exif()
    ex[0x0112] = 6
    out, info = P.prepare(_jpeg(Image.fromarray(a), ex), 26.0)
    assert info["orig_wh"] == [300, 400] and not info["landscape"]
    # Stored left half becomes the top half after a 90 deg clockwise turn.
    assert out[10, P.IMG_SIZE // 2].mean() > 200 and out[-10, P.IMG_SIZE // 2].mean() < 50


def test_output_carries_no_exif_and_focal_is_read():
    ex = Image.Exif()
    ex.get_ifd(P.EXIF_IFD)[P.TAG_FOCAL_35MM] = 28
    ex[0x0110] = "test-phone"                         # Model
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "raw"), os.path.join(d, "png")
        os.makedirs(src)
        Image.fromarray(np.full((300, 400, 3), 128, np.uint8)).save(
            os.path.join(src, "a.jpg"), exif=ex)
        Image.fromarray(np.full((300, 400, 3), 128, np.uint8)).save(os.path.join(src, "b.jpg"))
        P.main(src, out, P.FALLBACK_FOCAL_35MM)
        with Image.open(os.path.join(out, "0000.png")) as im:
            assert len(im.getexif()) == 0 and not im.info.get("exif")
        with open(os.path.join(out, "manifest.json")) as fh:
            m = json.load(fh)
        assert m[0]["focal_35mm"] == 28 and m[0]["focal_from_exif"]
        assert m[1]["focal_35mm"] == P.FALLBACK_FOCAL_35MM and not m[1]["focal_from_exif"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} prep_real checks passed")
