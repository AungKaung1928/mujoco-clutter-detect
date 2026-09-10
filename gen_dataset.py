"""Render a labelled multi-object detection dataset from MuJoCo.

Labels are read out of the renderer's segmentation buffer, not annotated and not
projected by hand. That makes them exact under perspective, exact for a rotated
box silhouette, and exact about what is hidden behind what.

The occlusion label needs a second pass. The segmentation buffer says which
object won each pixel; it cannot say how many pixels an object *would* have had
alone. So for each object we re-render it with the others parked away and divide.
That is N extra renders per scene, which is the dominant cost -- so we skip it
for any object whose visible box overlaps nothing, because such an object is
provably unoccluded. Most objects in a scene are.

Usage (from the repo root, with the virtualenv active):
    python gen_dataset.py --regime hard --n 200 --smoke
    python gen_dataset.py --regime easy --n 12000
    python gen_dataset.py --regime hard --n 12000
"""
import argparse
import json
import os
import time

import numpy as np
import mujoco

import common as C

ROOT = os.path.dirname(os.path.abspath(__file__))
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


def make_model(img_size):
    xml = open(os.path.join(ROOT, "scene.xml")).read()
    xml = xml.replace('offwidth="256"', f'offwidth="{img_size}"')
    xml = xml.replace('offheight="256"', f'offheight="{img_size}"')
    return mujoco.MjModel.from_xml_string(xml)


def sample_object(rng):
    """One object: class, mjGeom size triple, and its half-height above the table."""
    cls = C.CLASSES[rng.integers(len(C.CLASSES))]
    if cls == "box":
        hx = rng.uniform(0.020, 0.035)
        hy = hx * rng.uniform(0.85, 1.15)
        hz = rng.uniform(0.020, 0.035)
        size, half_h, gtype = [hx, hy, hz], hz, mujoco.mjtGeom.mjGEOM_BOX
    elif cls == "cylinder":
        r = rng.uniform(0.018, 0.030)
        hz = rng.uniform(0.022, 0.040)
        size, half_h, gtype = [r, hz, 0.0], hz, mujoco.mjtGeom.mjGEOM_CYLINDER
    else:
        r = rng.uniform(0.020, 0.032)
        size, half_h, gtype = [r, 0.0, 0.0], r, mujoco.mjtGeom.mjGEOM_SPHERE
    yaw = rng.uniform(0, np.pi / 2) if C.YAW_OBSERVABLE[cls] else np.nan
    return cls, int(gtype), size, half_h, yaw


def sample_positions(rng, n):
    """Rejection sampling so objects can crowd and occlude but never interpenetrate."""
    pts = []
    for _ in range(400):
        if len(pts) == n:
            break
        p = rng.uniform(-C.XY_RANGE, C.XY_RANGE, 2)
        if all(np.linalg.norm(p - q) >= C.MIN_SEP for q in pts):
            pts.append(p)
    return pts


def sample_appearance(model, rng, regime, gids, table_gid, light_id):
    if regime == "easy":
        for k, g in enumerate(gids):
            model.geom_rgba[g] = [*C.hsv_to_rgb(k / len(C.CLASSES) % 1.0, 0.85, 0.8), 1.0]
        model.geom_rgba[table_gid] = [0.75, 0.75, 0.72, 1.0]
        model.light_pos[light_id] = [0.0, 0.0, 1.2]
        model.light_diffuse[light_id] = [0.7, 0.7, 0.7]
    else:
        for g in gids:
            model.geom_rgba[g] = [
                *C.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.5, 1.0), rng.uniform(0.45, 1.0)), 1.0]
        g0 = rng.uniform(0.25, 0.85)
        model.geom_rgba[table_gid] = [g0, g0 * rng.uniform(0.9, 1.1), g0 * rng.uniform(0.9, 1.1), 1.0]
        model.light_pos[light_id] = [rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.3), rng.uniform(0.9, 1.4)]
        d = rng.uniform(0.4, 1.0)
        model.light_diffuse[light_id] = [d, d, d]


def place(model, bid, xy, half_h, yaw):
    model.body_pos[bid] = [xy[0], xy[1], half_h]
    if np.isnan(yaw):
        model.body_quat[bid] = [1.0, 0.0, 0.0, 0.0]
    else:
        model.body_quat[bid] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def park(model, bid):
    model.body_pos[bid] = [C.PARK_XY, C.PARK_XY, 0.0]
    model.body_quat[bid] = [1.0, 0.0, 0.0, 0.0]


def seg_ids(renderer, model, data, cam):
    """Instance-id image: which geom won each pixel, -1 for background."""
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam)
    s = renderer.render()
    renderer.disable_segmentation_rendering()
    return np.where(s[..., 1] == GEOM, s[..., 0], -1)


def generate(regime, n, img_size, seed, split, cam, do_occlusion=True):
    model = make_model(img_size)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, img_size, img_size)
    rng = np.random.default_rng(seed)

    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(C.N_SLOTS)]
    gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}") for i in range(C.N_SLOTS)]
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "l0")

    outdir = os.path.join(ROOT, "data", regime)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, split)

    imgs = np.lib.format.open_memmap(
        base + "_images.npy", mode="w+", dtype=np.uint8, shape=(n, img_size, img_size, 3))
    rows = []
    n_dropped = n_solo = 0

    t0 = time.perf_counter()
    for i in range(n):
        k = int(rng.integers(3, C.N_SLOTS + 1))
        pts = sample_positions(rng, k)
        k = len(pts)
        objs = [sample_object(rng) for _ in range(k)]

        for s in range(C.N_SLOTS):
            if s < k:
                cls, gtype, size, half_h, yaw = objs[s]
                model.geom_type[gids[s]] = gtype
                model.geom_size[gids[s]] = size
                place(model, bids[s], pts[s], half_h, yaw)
            else:
                park(model, bids[s])
        sample_appearance(model, rng, regime, gids[:k], table_gid, light_id)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam)
        imgs[i] = renderer.render()
        seg = seg_ids(renderer, model, data, cam)

        vis_px, vis_box = [], []
        for s in range(k):
            m = seg == gids[s]
            vis_px.append(int(m.sum()))
            vis_box.append(C.mask_to_box(m))

        # An object whose visible box touches no other visible box cannot have
        # been occluded, so it needs no second render.
        present = [s for s in range(k) if vis_box[s] is not None]
        need_solo = set()
        for a_i, a in enumerate(present):
            for b in present[a_i + 1:]:
                if C.boxes_iou([vis_box[a]], [vis_box[b]])[0, 0] > 0:
                    need_solo.add(a)
                    need_solo.add(b)
        # a completely invisible object is fully occluded; measure it anyway
        need_solo |= {s for s in range(k) if vis_box[s] is None}

        full_px = dict.fromkeys(range(k), None)
        for s in sorted(need_solo):
            for t in range(k):
                if t != s:
                    park(model, bids[t])
            mujoco.mj_forward(model, data)
            full_px[s] = int((seg_ids(renderer, model, data, cam) == gids[s]).sum())
            n_solo += 1
            for t in range(k):
                if t != s:
                    place(model, bids[t], pts[t], objs[t][3], objs[t][4])
            mujoco.mj_forward(model, data)

        for s in range(k):
            cls, _, _, _, yaw = objs[s]
            fp = full_px[s] if full_px[s] is not None else vis_px[s]
            frac = float(vis_px[s] / fp) if fp > 0 else 0.0
            if vis_box[s] is None or vis_px[s] < C.MIN_VISIBLE_PX or frac < C.MIN_VISIBLE_FRAC:
                n_dropped += 1
                continue
            x0, y0, x1, y1 = vis_box[s]
            rows.append([i, C.CLS_ID[cls], x0, y0, x1, y1, frac,
                         pts[s][0], pts[s][1], yaw])

        if (i + 1) % 200 == 0:
            r = (i + 1) / (time.perf_counter() - t0)
            print(f"  {i+1}/{n}  {r:6.1f} img/s", flush=True)

    dt = time.perf_counter() - t0
    imgs.flush()
    boxes = np.asarray(rows, dtype=np.float32)
    np.save(base + "_boxes.npy", boxes)

    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({
            "regime": regime, "img_size": img_size, "camera": cam,
            "classes": list(C.CLASSES), "xy_range_m": C.XY_RANGE,
            "min_sep_m": C.MIN_SEP, "yaw_fold": C.YAW_FOLD,
            "box_cols": C.BOX_COLS,
            "min_visible_px": C.MIN_VISIBLE_PX, "min_visible_frac": C.MIN_VISIBLE_FRAC,
        }, f, indent=2)

    occ = boxes[:, 6]
    print(f"\n{regime}/{split}: {n} imgs at {img_size}px, camera '{cam}', in {dt:.1f}s "
          f"({n/dt:.1f} img/s), {imgs.nbytes/1e6:.0f} MB")
    print(f"  objects kept        {len(boxes)}  ({len(boxes)/n:.2f}/img), dropped {n_dropped}")
    print(f"  solo re-renders     {n_solo}  ({n_solo/n:.2f}/img)")
    print(f"  visible_frac        median {np.median(occ):.3f}   "
          f"<0.9 (occluded) {100*(occ < 0.9).mean():.1f}%   <0.5 {100*(occ < 0.5).mean():.1f}%")
    side = np.sqrt((boxes[:, 4] - boxes[:, 2]) * (boxes[:, 5] - boxes[:, 3]))
    print(f"  box size (sqrt area) median {np.median(side):.1f} px   "
          f"p5 {np.percentile(side,5):.1f}   p95 {np.percentile(side,95):.1f}")
    for ci, cn in enumerate(C.CLASSES):
        print(f"  class {cn:9s}   {(boxes[:,1]==ci).sum():6d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=C.REGIMES, required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--img-size", type=int, default=192)
    ap.add_argument("--camera", default="tilt", choices=["tilt", "top"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="train split only, no val")
    a = ap.parse_args()

    generate(a.regime, a.n, a.img_size, a.seed, "train", a.camera)
    if not a.smoke:
        generate(a.regime, max(200, a.n // 6), a.img_size, a.seed + 10_000, "val", a.camera)
