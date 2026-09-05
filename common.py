"""Scene constants, projection, and box geometry for the clutter-detection task.

Block 1 regressed one pose from a top-down view. Block 2 detects a variable
number of objects of three classes from a tilted view. Two things change and
both are deliberate:

  1. The camera is tilted. Top-down, objects sitting on a flat table can never
     overlap in the image, so occlusion is structurally impossible and detection
     collapses into "find the blobs". Tilting brings back perspective,
     depth-dependent scale, and real occlusion. The overhead camera stays in the
     scene because it is the only view with an exact affine pixel<->world map,
     which is what verifies labels and what the policy blocks will need.

  2. Labels come from the segmentation buffer, not from projected geometry. A
     rendered instance mask is exact -- it already accounts for perspective, for
     the silhouette of a rotated box, and for what is hidden behind what. This
     is the concrete reason robot-learning work starts in simulation: the label
     is not annotated, it is read out.
"""
import json
import os

import numpy as np

# --- scene constants, must match scene.xml ---
TOP_CAM_HEIGHT = 0.45      # m
FOVY_DEG = 45.0
N_SLOTS = 6                # object bodies present in the XML
PARK_XY = 10.0             # m, where unused slots go (outside both FOVs)

# object centres are sampled inside this square (m)
XY_RANGE = 0.12
# rejection-sampling floor between centres, so objects touch but do not interpenetrate
MIN_SEP = 0.075

CLASSES = ("box", "cylinder", "sphere")
CLS_ID = {c: i for i, c in enumerate(CLASSES)}

# A sphere has no observable yaw and a cylinder's yaw about z is unobservable too.
# Only the box carries an orientation label. Pretending otherwise would train the
# network to fit noise.
YAW_OBSERVABLE = {"box": True, "cylinder": False, "sphere": False}

# Same 90 deg fold as block 1: a square-topped box is only observable modulo 90 deg.
YAW_FOLD = 4

# label columns of {split}_boxes.npy
BOX_COLS = ["img_idx", "cls", "x0", "y0", "x1", "y1", "visible_frac",
            "world_x", "world_y", "yaw_rad"]

MIN_VISIBLE_PX = 12        # below this the object is not a detectable target
MIN_VISIBLE_FRAC = 0.10    # below this it is occluded past usefulness


def yaw_to_vec(yaw):
    return np.stack([np.sin(YAW_FOLD * yaw), np.cos(YAW_FOLD * yaw)], -1)


def vec_to_yaw(v):
    t = np.arctan2(v[..., 0], v[..., 1]) / YAW_FOLD
    return np.mod(t, np.pi / 2)


def yaw_err_deg(a, b):
    d = np.mod(np.asarray(a) - np.asarray(b), np.pi / 2)
    d = np.minimum(d, np.pi / 2 - d)
    return np.degrees(d)


# --- overhead projection, unchanged from block 1, kept for label verification ---

def top_px_per_m(img_size, obj_half_height):
    d = TOP_CAM_HEIGHT - obj_half_height
    half_extent = d * np.tan(np.radians(FOVY_DEG) / 2)
    return (img_size / 2) / half_extent


def top_world_to_pixel(x, y, img_size, obj_half_height):
    s = top_px_per_m(img_size, obj_half_height)
    return img_size / 2 + np.asarray(x) * s, img_size / 2 - np.asarray(y) * s


# --- box geometry ---

def boxes_iou(a, b):
    """a: (N,4), b: (M,4), both x0,y0,x1,y1 -> (N,M) IoU."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def mask_to_box(mask):
    """Tight axis-aligned box of a boolean mask -> (x0,y0,x1,y1) or None.
    Half-open in the max: x1 is one past the last occupied column."""
    cols = mask.any(0)
    rows = mask.any(1)
    if not cols.any():
        return None
    xs = np.flatnonzero(cols)
    ys = np.flatnonzero(rows)
    return float(xs[0]), float(ys[0]), float(xs[-1] + 1), float(ys[-1] + 1)


# --- appearance ---

def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


REGIMES = ("easy", "hard")


def load_split(root, regime, split):
    p = os.path.join(root, "data", regime)
    imgs = np.load(os.path.join(p, split + "_images.npy"), mmap_mode="r")
    boxes = np.load(os.path.join(p, split + "_boxes.npy"))
    with open(os.path.join(p, "meta.json")) as f:
        meta = json.load(f)
    return imgs, boxes, meta


def group_boxes(boxes, n_images):
    """(M,10) flat table -> list of n_images arrays. The flat table is what gets
    saved (one .npy, memmap-friendly); this is what the training loop wants."""
    out = [[] for _ in range(n_images)]
    idx = boxes[:, 0].astype(np.int64)
    for i, row in zip(idx, boxes):
        out[i].append(row[1:])
    return [np.asarray(v, dtype=np.float32).reshape(-1, len(BOX_COLS) - 1) for v in out]
