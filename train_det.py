"""Train the anchor-free detector and score it with the same AP harness the
classical baseline was scored with.

`report` is imported from baseline_cv rather than reimplemented. Two copies of a
metric drift, and the comparison this project exists to make would silently stop
being a comparison.

Augmentation is a flag here rather than a separate script, because step 4 is an
ablation: the only honest way to attribute a difference to augmentation is to
change nothing else.

    --aug none          simulator's own randomisation only
    --aug photo         brightness / contrast / channel gain / gamma / noise
    --aug geom          horizontal flip + translation
    --aug both

Horizontal flip is legal here and vertical flip is not. The tilt camera sits at
(0, -0.42, 0.34) looking along +y, so the scene is mirror-symmetric about x = 0
and a flipped image is a scene that could have occurred. Flipping vertically
would put the far edge of the table nearer than the front edge, which no camera
pose produces -- the network would be trained on physically impossible evidence.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import common as C
import ap as AP
import detector as D
from baseline_cv import report, detections

ROOT = os.path.dirname(os.path.abspath(__file__))
MEAN, STD = 0.45, 0.25


# --------------------------------------------------------------- augmentation

def photometric(img, rng):
    """uint8 HWC -> float32 HWC in [0,1]. Appearance only, geometry untouched."""
    x = img.astype(np.float32) / 255.0
    x = x * rng.uniform(0.7, 1.3)                                  # brightness
    m = x.mean()
    x = (x - m) * rng.uniform(0.7, 1.3) + m                        # contrast
    x = x * rng.uniform(0.85, 1.15, size=3).astype(np.float32)     # channel gain
    x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)                 # gamma
    x = x + rng.normal(0, 0.02, x.shape).astype(np.float32)        # sensor noise
    return np.clip(x, 0, 1)


def geometric(x, boxes, rng, size):
    """Flip and translate together, boxes carried along. Boxes that leave the
    frame are dropped, not clipped to a sliver: a 3-pixel remnant of a cube is
    not a cube, and training the size head on it teaches a lie."""
    if rng.random() < 0.5:
        x = x[:, ::-1]
        b = boxes.copy()
        boxes = b.copy()
        boxes[:, 1] = size - b[:, 3]
        boxes[:, 3] = size - b[:, 1]
    dx, dy = rng.integers(-14, 15), rng.integers(-14, 15)
    x = np.roll(x, (dy, dx), axis=(0, 1))
    # roll wraps; overwrite the wrapped strip with the edge pixel instead
    if dx > 0:
        x[:, :dx] = x[:, dx:dx + 1]
    elif dx < 0:
        x[:, dx:] = x[:, dx - 1:dx]
    if dy > 0:
        x[:dy] = x[dy:dy + 1]
    elif dy < 0:
        x[dy:] = x[dy - 1:dy]

    b = boxes.copy()
    b[:, 1] += dx; b[:, 3] += dx
    b[:, 2] += dy; b[:, 4] += dy
    a0 = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])
    np.clip(b[:, [1, 3]], 0, size, out=b[:, [1, 3]])
    np.clip(b[:, [2, 4]], 0, size, out=b[:, [2, 4]])
    a1 = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])
    keep = (a1 > 0.5 * np.maximum(a0, 1e-6)) & (b[:, 3] - b[:, 1] > 4) & (b[:, 4] - b[:, 2] > 4)
    return np.ascontiguousarray(x), b[keep]


class DetData(Dataset):
    def __init__(self, imgs, boxes, size, aug="none", seed=0):
        self.imgs = imgs
        self.gb = C.group_boxes(boxes, len(imgs))
        self.size = size
        self.grid = size // D.STRIDE
        self.aug = aug
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch) * 100_003 + i)
        b = self.gb[i]
        if self.aug in ("photo", "both"):
            x = photometric(self.imgs[i], rng)
        else:
            x = self.imgs[i].astype(np.float32) / 255.0
        if self.aug in ("geom", "both"):
            x, b = geometric(x, b, rng, self.size)
        hm, ind, wh, off, mask = D.encode(b[:, 1:5], b[:, 0], self.grid)
        x = (x - MEAN) / STD
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(hm), torch.from_numpy(ind),
                torch.from_numpy(wh), torch.from_numpy(off), torch.from_numpy(mask))


# ------------------------------------------------------------------ inference

@torch.no_grad()
def predict(model, imgs, size, bs=64, k=10, thr=0.02):
    model.eval()
    out_b, out_c, out_s, out_i = [], [], [], []
    for s in range(0, len(imgs), bs):
        x = imgs[s:s + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2).copy())
        hm, wh, off = model(x)
        bx, cl, sc = D.decode(hm, wh, off, k=k)
        for j in range(len(bx)):
            m = sc[j] > thr
            n = int(m.sum())
            if n == 0:
                continue
            out_b.append(bx[j][m].numpy())
            out_c.append(cl[j][m].numpy())
            out_s.append(sc[j][m].numpy())
            out_i.append(np.full(n, s + j))
    if not out_b:
        return np.zeros((0, 7))
    return detections(np.concatenate(out_b), np.concatenate(out_i),
                      np.concatenate(out_c), np.concatenate(out_s))


@torch.no_grad()
def latency(model, imgs, n=200):
    """Batch 1, the only number an edge deployment can use. Includes decode,
    because a heatmap is not a detection."""
    model.eval()
    x = torch.from_numpy(((imgs[:n].astype(np.float32) / 255.0 - MEAN) / STD)
                         .transpose(0, 3, 1, 2).copy())
    for i in range(10):
        D.decode(*model(x[i:i + 1]))
    t0 = time.perf_counter()
    for i in range(n):
        D.decode(*model(x[i:i + 1]))
    return (time.perf_counter() - t0) / n * 1e3


def quick_map(model, imgs, gts, size, n=800):
    d = predict(model, imgs[:n], size)
    g = gts[gts[:, 0] < n]
    if len(d) == 0:
        return 0.0
    return AP.evaluate(d, g[:, :6].astype(np.float64))["mAP"]


# ------------------------------------------------------------------- training

def eval_on(model, regime, tag):
    """Score a trained model on a regime's val split. Called for the training
    regime and for the other one: a model trained on `easy` and tested on `hard`
    is the whole point of step 4, and it costs no training time."""
    va_i, va_b, meta = C.load_split(ROOT, regime, "val")
    gts = va_b[:, :6].astype(np.float64)
    vf = va_b[:, 6].astype(np.float64)
    edges = [0.0, *np.percentile(AP.box_scale(gts[:, 2:6]), [33.3, 66.7]), 1e9]
    dets = predict(model, va_i, meta["img_size"])
    ms = latency(model, va_i)
    return report(tag, dets, gts, vf, edges, len(va_i), ms)


def train(regime, aug, epochs, bs, lr, width, seed, tune_n, out_json, out_ckpt,
          fit_n=0, cross=True):
    torch.manual_seed(seed)
    tr_i, tr_b, meta = C.load_split(ROOT, regime, "train")
    va_i, va_b, _ = C.load_split(ROOT, regime, "val")
    size = meta["img_size"]

    # last `tune_n` training images are held out to choose the epoch budget and
    # watch for divergence. val is touched exactly once, at the end.
    n_fit = len(tr_i) - tune_n
    if fit_n:
        n_fit = min(n_fit, fit_n)          # step 4 runs a reduced, but identical, budget
    tr_i = np.asarray(tr_i)                      # 1.3 GB, fits; memmap random access does not
    fit_b = tr_b[tr_b[:, 0] < n_fit]
    tune_b = tr_b[tr_b[:, 0] >= n_fit].copy()
    tune_b[:, 0] -= n_fit
    tune_i = tr_i[n_fit:]

    ds = DetData(tr_i[:n_fit], fit_b, size, aug=aug, seed=seed)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

    model = D.Detector(w=width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.3)

    print(f"regime {regime}  aug {aug}  params {model.n_params():,}  "
          f"fit {n_fit} tune {tune_n} val {len(va_i)}  {epochs} epochs x {len(dl)} steps")
    hist = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        ds.epoch = ep
        model.train()
        agg = np.zeros(4)
        te = time.perf_counter()
        for x, hm, ind, wh, off, mask in dl:
            p_hm, p_wh, p_off = model(x)
            l_hm = D.focal_loss(p_hm, hm)
            l_wh = D.reg_l1(p_wh, ind, wh, mask)
            l_off = D.reg_l1(p_off, ind, off, mask)
            loss = l_hm + 0.1 * l_wh + l_off
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            agg += [loss.item(), l_hm.item(), l_wh.item(), l_off.item()]
        agg /= len(dl)
        dt = time.perf_counter() - te
        line = (f"  ep {ep+1:3d}/{epochs}  loss {agg[0]:.4f}  hm {agg[1]:.4f}  "
                f"wh {agg[2]:.3f}  off {agg[3]:.4f}  {n_fit/dt:6.0f} img/s")
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            m = quick_map(model, tune_i, tune_b, size)
            line += f"   tune mAP {m:.4f}"
            hist.append({"epoch": ep + 1, "tune_mAP": float(m), "loss": float(agg[0])})
        print(line, flush=True)
    train_s = time.perf_counter() - t0

    res = eval_on(model, regime, f"[cnn/{aug}] {regime}  (trained here)")
    res.update({"regime": regime, "aug": aug, "epochs": epochs, "params": model.n_params(),
                "train_s": train_s, "history": hist, "width": width, "seed": seed,
                "n_fit": n_fit})
    if cross:
        other = [r for r in C.REGIMES if r != regime][0]
        res["cross"] = eval_on(model, other, f"[cnn/{aug}] {other}  (TRAINED ON {regime})")
        res["cross_regime"] = other

    if out_ckpt:
        torch.save({"model": model.state_dict(), "width": width, "img_size": size}, out_ckpt)
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        json.dump(res, open(out_json, "w"), indent=2)
        print(f"\nwrote {out_json}")
    return res


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--regime", choices=C.REGIMES, default="hard")
    a.add_argument("--aug", choices=["none", "photo", "geom", "both"], default="none")
    a.add_argument("--epochs", type=int, default=30)
    a.add_argument("--bs", type=int, default=32)
    a.add_argument("--lr", type=float, default=2.5e-3)
    a.add_argument("--width", type=int, default=32)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--tune-n", type=int, default=1500)
    a.add_argument("--fit-n", type=int, default=0, help="cap training images (0 = all)")
    a.add_argument("--no-cross", action="store_true")
    a.add_argument("--save", default="")
    a.add_argument("--ckpt", default="")
    a = a.parse_args()
    train(a.regime, a.aug, a.epochs, a.bs, a.lr, a.width, a.seed, a.tune_n, a.save,
          a.ckpt, a.fit_n, not a.no_cross)
