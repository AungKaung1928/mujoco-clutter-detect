"""Object-on-object occlusion labelling, on hand-built scenes. Needs MuJoCo rendering.

test_distractors covers a distractor in front of a labelled object. This covers the
common case in every regime: one labelled object in front of another, and one
hidden completely. Both must be re-rendered alone so visible_frac is measured, not
assumed to be 1.

Run:  MUJOCO_GL=egl python test_occlusion.py
"""
import sys

import mujoco
import numpy as np

import common as C
import gen_dataset as G

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:52s} {detail}")
    if not ok:
        FAILED.append(name)


model = G.make_model(192)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 192, 192)
bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(C.N_SLOTS)]
gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}") for i in range(C.N_SLOTS)]


def scene(objs):
    """objs: [(xy, half size)] boxes; slot 0 is the far one, the tilted camera sits at y = -0.42."""
    for b in bids:
        G.park(model, b)
    placed = []
    for s, (xy, h) in enumerate(objs):
        model.geom_type[gids[s]] = mujoco.mjtGeom.mjGEOM_BOX
        model.geom_size[gids[s]] = [h, h, h]
        placed.append((np.asarray(xy, float), h, 0.0))
        G.place(model, bids[s], *placed[-1])
    mujoco.mj_forward(model, data)
    return G.visibility(renderer, model, data, "tilt", bids, gids, placed, [])


print("\none labelled box partly in front of another")
vis_px, vis_box, full_px, n = scene([((0.0, 0.0), 0.03), ((0.02, -0.08), 0.03)])
check("back box gets a solo render", full_px[0] is not None, f"full_px {full_px[0]}")
frac = vis_px[0] / full_px[0] if full_px[0] else float("nan")
check("back box visible fraction strictly between 0 and 1", 0.0 < frac < 0.95, f"{vis_px[0]} / {full_px[0]} = {frac:.3f}")
check("both slots placed back after the solo renders",
      np.allclose(model.body_pos[bids[0]][:2], [0.0, 0.0]) and np.allclose(model.body_pos[bids[1]][:2], [0.02, -0.08]))

print("\na small box hidden completely behind a large one")
vis_px, vis_box, full_px, n = scene([((0.0, 0.0), 0.012), ((0.0, -0.075), 0.035)])
check("hidden box has no visible pixels", vis_box[0] is None and vis_px[0] == 0, f"vis_px {vis_px[0]}")
check("hidden box is still measured alone", full_px[0] is not None and full_px[0] > 0, f"full_px {full_px[0]}")

print("\nlabel filter (12 px and 10 % floors, common.py)")
box = (0.0, 0.0, 10.0, 10.0)
check("unoccludable, 100 px: kept, fraction 1", G.label_fraction(100, box, None) == 1.0)
check("11 px: dropped", G.label_fraction(11, box, None) is None)
check("12 px: kept", G.label_fraction(12, box, None) == 1.0)
check("20 of 250 px (8 %): dropped", G.label_fraction(20, box, 250) is None)
check("30 of 250 px (12 %): kept at 0.12", G.label_fraction(30, box, 250) == 0.12)
check("hidden (no box): dropped", G.label_fraction(0, None, 140) is None)

print(f"\n{'ALL PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
