# Real-camera test set — capture protocol

Every number in the README is measured on rendered images. This file fixes how
the first real-camera test set is captured, before any photo is taken, so the
protocol cannot drift toward whatever makes the detector look good.

## What the renderer produces (the target to match)

Read from `gen_dataset.py` and `scene.xml`, `hard` regime:

| | sim |
|---|---|
| classes | `box`, `cylinder`, `sphere` only |
| objects per image | 3–6, centres inside a 0.24 m square, ≥ 0.075 m apart |
| box | near-cube, side 40–70 mm, any rotation about z |
| cylinder | upright, Ø 36–60 mm, height 44–80 mm |
| sphere | Ø 40–64 mm |
| camera | `tilt`: 0.34 m above the table, 0.42 m back from the square's centre, 39° down, optical axis through the centre |
| field of view | 45° vertical, square image |
| image | 192 × 192 RGB |
| appearance | one matte colour per object (random hue), plain grey table, one moving light |

## Objects

About nine household objects, three per class, sized to the table above, each a
different colour. Matte and single-coloured where possible; small print is
acceptable, glass and polished metal are not (the renderer has no reflections).
Typical picks: phone charger cube, Rubik's cube, die / wooden block; small
drink bottle, spice jar, small can or cup; ping-pong ball, golf ball, mandarin.

The same objects are reused across all photos. Each photo changes which 3–6
objects are present, where they sit, how close they are, their rotation, and
the light.

## Camera

- Phone resting on a fixed support, lens 0.34 m above the surface and 0.42 m
  back from the centre of the object square, pointed at that centre.
- Landscape orientation. Nothing else in frame: no hands, cables, other items.
- Surface: plain light paper or a plain table. Note the surface per photo.

### Matching the field of view

A typical phone main camera is ~26 mm equivalent: about 53° vertical in a 4:3
landscape frame, against the sim's 45°. `prep_real.py` crops a centred square of
side

    s = H · tan(45° / 2) / tan(vfov_phone / 2)        (≈ 0.83 H at 53°)

then resizes it to 192 × 192. `vfov_phone` comes from the photo's EXIF
`FocalLengthIn35mmFilm`, not from a guess. EXIF (including GPS) is stripped
from every image before anything is committed (OpenCV writes PNG with no EXIF;
EXIF orientation is applied first, so portrait shots come out upright).

## Procedure

1. Five test photos first; check angle, object sizes in pixels against the sim's
   (median ≈ 28 px at 192 px) and the crop, then fix the setup.
2. 100 photos. Move at least two objects between shots; change the light
   (ceiling / lamp / window) every ~20 photos.
3. Label boxes in Label Studio, export, convert to the `{split}_boxes.npy`
   column layout in `common.BOX_COLS`. `visible_frac`, `world_x/y` and `yaw`
   are unknown for real photos and stored as NaN.

## Splits — fixed before any result is seen

- **dev, 30 photos**: failures are inspected here and drive any change to the
  renderer or training.
- **test, 70 photos**: run once, at the end. That is the reported number.

Split by capture order in blocks (not random per photo), so near-duplicate
consecutive shots cannot land on both sides.

## What the result will and will not show

It measures transfer to one phone camera, about nine objects and a few rooms.
It does not cover other cameras, cluttered backgrounds, objects outside the
three primitive shapes, or a moving robot.
