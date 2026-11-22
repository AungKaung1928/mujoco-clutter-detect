"""world_x / world_y labels checked through the overhead camera. Needs MuJoCo rendering.

The overhead camera is the one view with an exact pixel<->world map
(common.top_world_to_pixel, any height above the table). From the labels and the
sampled shape alone, the tight box of each object's mask is predicted in closed
form: a box or cylinder spans the extremes of its top (z = 2*half_h) and bottom
(z = 0) outlines, because an off-axis object shows the side facing the camera; a
sphere spans its two tangent rays, f*tan(atan(x/d) +- asin(r/hypot(x, d))). Each
predicted edge must match the rendered segmentation box within 1 px. A scene.xml
camera edit, a wrong height in place(), a wrong size or yaw magnitude, or an axis flip
in the projection moves edges by pixels. The yaw sign is invisible here: the
axis-aligned box of a rotated near-square is the same for +yaw and -yaw.

Run:  MUJOCO_GL=egl python test_top_labels.py
"""
import sys

import mujoco
import numpy as np

import common as C
import gen_dataset as G

IMG, SCENES, TOL_PX = 192, 30, 1.0

model = G.make_model(IMG)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, IMG, IMG)
bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(C.N_SLOTS)]
gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}") for i in range(C.N_SLOTS)]
cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")

failed = []
if not (np.isclose(model.cam_pos[cam][2], C.TOP_CAM_HEIGHT) and np.isclose(model.cam_fovy[cam], C.FOVY_DEG)):
    failed.append(f"scene.xml top camera z={model.cam_pos[cam][2]} fovy={model.cam_fovy[cam]} "
                  f"!= common.py {C.TOP_CAM_HEIGHT} / {C.FOVY_DEG}")

F = IMG / 2 / np.tan(np.radians(C.FOVY_DEG) / 2)


def predicted_box(cls, xy, size, half_h, yaw):
    x, y = xy
    if cls == "sphere":
        d, r = C.TOP_CAM_HEIGHT - half_h, size[0]
        uu = [IMG / 2 + F * np.tan(np.arctan2(x, d) + sg * np.arcsin(r / np.hypot(x, d))) for sg in (-1, 1)]
        vv = [IMG / 2 - F * np.tan(np.arctan2(y, d) + sg * np.arcsin(r / np.hypot(y, d))) for sg in (-1, 1)]
    else:
        if cls == "box":
            c, s_ = np.cos(yaw), np.sin(yaw)
            outline = [(x + c * a - s_ * b, y + s_ * a + c * b)
                       for a in (-size[0], size[0]) for b in (-size[1], size[1])]
        else:   # cylinder: per-axis extremes of a circle are its 4 axis points
            r = size[0]
            outline = [(x + r, y), (x - r, y), (x, y + r), (x, y - r)]
        px = [C.top_world_to_pixel(px_, py_, IMG, z) for z in (0.0, 2 * half_h) for px_, py_ in outline]
        uu, vv = [p[0] for p in px], [p[1] for p in px]
    # the segmentation box cannot leave the frame (an object near y = +0.12 m pokes out)
    return tuple(np.clip([min(uu), min(vv), max(uu), max(vv)], 0, IMG))


rng = np.random.default_rng(0)
errs, n_cls = [], dict.fromkeys(C.CLASSES, 0)
for scene in range(SCENES):
    pts = G.sample_positions(rng, int(rng.integers(3, C.N_SLOTS + 1)))
    objs = [G.sample_object(rng) for _ in pts]
    for s in range(C.N_SLOTS):
        if s < len(pts):
            cls, gtype, size, half_h, yaw = objs[s]
            model.geom_type[gids[s]] = gtype
            model.geom_size[gids[s]] = size
            G.place(model, bids[s], pts[s], half_h, yaw)
        else:
            G.park(model, bids[s])
    mujoco.mj_forward(model, data)
    seg = G.seg_ids(renderer, model, data, "top")
    for s, (cls, _, _, half_h, _) in enumerate(objs):
        box = C.mask_to_box(seg == gids[s])
        if box is None:
            failed.append(f"scene {scene} slot {s} {cls}: not visible from the top")
            continue
        e = np.abs(np.subtract(box, predicted_box(cls, pts[s], objs[s][2], half_h, objs[s][4]))).max()
        errs.append(e)
        n_cls[cls] += 1
        if e > TOL_PX:
            failed.append(f"scene {scene} slot {s} {cls} at ({pts[s][0]:+.3f}, {pts[s][1]:+.3f}): "
                          f"worst box edge {e:.2f} px from prediction")

print(f"  {len(errs)} objects {n_cls}, worst box edge vs prediction: "
      f"max {max(errs):.2f} px, median {np.median(errs):.2f} px (tolerance {TOL_PX} px)")
for f in failed[:10]:
    print("  FAIL ", f)
print("ALL PASSED" if not failed else f"FAILED: {len(failed)}")
sys.exit(1 if failed else 0)
