"""Distractor occlusion in regime texd, on one hand-built scene. Needs MuJoCo rendering.

A cube sits at the origin; an unlabeled ellipsoid stands between it and the
tilted camera. The cube's visible fraction must come from a solo render with
the distractor parked. Before the fix, distractors were not in the overlap test,
so this cube kept full_px None and was labelled 100 % visible.

Run:  python test_distractors.py
"""
import sys

import mujoco
import numpy as np

import common as C
import gen_dataset as G

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:56s} {detail}")
    if not ok:
        FAILED.append(name)


model = G.make_model(192, "texd")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 192, 192)
ids = lambda kind, names: [mujoco.mj_name2id(model, kind, n) for n in names]
bids = ids(mujoco.mjtObj.mjOBJ_BODY, [f"obj{i}" for i in range(C.N_SLOTS)])
gids = ids(mujoco.mjtObj.mjOBJ_GEOM, [f"g{i}" for i in range(C.N_SLOTS)])
dbid, = ids(mujoco.mjtObj.mjOBJ_BODY, ["dis0"])
dgid, = ids(mujoco.mjtObj.mjOBJ_GEOM, ["dg0"])
check("texd model has distractor slots", dbid >= 0 and dgid >= 0)

for b in bids:
    G.park(model, b)
model.geom_type[gids[0]] = mujoco.mjtGeom.mjGEOM_BOX
model.geom_size[gids[0]] = [0.03, 0.03, 0.03]
placed = [(np.array([0.0, 0.0]), 0.03, 0.0)]
G.place(model, bids[0], *placed[0])
model.geom_type[dgid] = mujoco.mjtGeom.mjGEOM_ELLIPSOID
model.geom_size[dgid] = [0.05, 0.04, 0.06]
dxy = np.array([0.0, -0.10])                # between the cube and the camera at y = -0.42
G.place(model, dbid, dxy, 0.06, np.nan)
mujoco.mj_forward(model, data)

dis = [(dbid, dgid, dxy, 0.06)]
vis_px, vis_box, full_px, n = G.visibility(renderer, model, data, "tilt", bids, gids, placed, dis)
frac = vis_px[0] / full_px[0] if full_px[0] else float("nan")
check("occluded cube gets a solo render", full_px[0] is not None and n == 1, f"full_px {full_px[0]}")
check("visible fraction well under 1", frac < 0.6, f"{vis_px[0]} / {full_px[0]} = {frac:.3f}")
check("distractor restored after the solo render", np.allclose(model.body_pos[dbid][:2], dxy))

_, _, full_old, _ = G.visibility(renderer, model, data, "tilt", bids, gids, placed, [])
check("without distractors in the test, no solo render (the old bug)", full_old[0] is None)

rng = np.random.default_rng(0)
ell = [G.sample_distractor(rng) for _ in range(400)]
ell = [sz for t, sz, _ in ell if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID]
ratio = min(max(sz) / min(sz) for sz in ell)
check("no ellipsoid distractor is near-spherical (ratio >= 1.8)", ratio >= 1.8 - 1e-9, f"min ratio {ratio:.3f}, n {len(ell)}")

print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
