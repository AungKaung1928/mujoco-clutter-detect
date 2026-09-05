"""Step 5: export to ONNX, verify it is the same detector, and measure what the
runtime actually costs.

Block 1 ended on a result that surprised me and is repeated here on a harder
model: the *runtime*, not the architecture, dominated edge latency. A 0.65 ms
PyTorch-eager forward on 8 threads became 0.23 ms under ONNX Runtime on ONE
thread. Same weights, same arithmetic, 2.8x faster on 1/8th of the cores.

So the number a deployment plan needs is not 'ms/img'. It is 'ms/img, under
which runtime, on how many threads' -- and if a report omits the last two, it is
not a latency measurement, it is a rumour.

Verification comes before timing. An export that is 3x faster and 2% wrong is
not an optimisation, and a max-abs-diff on the output tensors is not enough to
prove it: small logit drift can move a heatmap peak by one cell and change a
box. So the check here is end-to-end -- full val mAP through the ONNX graph,
compared against the PyTorch mAP to four decimal places.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import onnxruntime as ort

import common as C
import ap as AP
import detector as D
from baseline_cv import detections
from train_det import MEAN, STD, predict, latency

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=True)
    m = D.Detector(w=c["width"])
    m.load_state_dict(c["model"])
    m.eval()
    return m, c["img_size"]


def export(model, size, path):
    """Batch is fixed at 1. A dynamic batch axis costs nothing here and buys
    nothing either: an edge camera delivers one frame at a time, and a graph
    exported for the shape it will actually see is the graph the runtime can
    plan for."""
    x = torch.zeros(1, 3, size, size)
    torch.onnx.export(model, (x,), path, opset_version=17,
                      input_names=["image"], output_names=["hm", "wh", "off"],
                      dynamo=False)
    return os.path.getsize(path)


def sess(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def onnx_predict(s, imgs, size, k=10, thr=0.02):
    """Same decode function the PyTorch path uses. Two decoders would make the
    comparison meaningless."""
    out_b, out_c, out_s, out_i = [], [], [], []
    for i in range(len(imgs)):
        x = ((imgs[i].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        hm, wh, off = s.run(None, {"image": x})
        bx, cl, sc = D.decode(torch.from_numpy(hm), torch.from_numpy(wh), torch.from_numpy(off), k=k)
        m = sc[0] > thr
        n = int(m.sum())
        if n == 0:
            continue
        out_b.append(bx[0][m].numpy()); out_c.append(cl[0][m].numpy())
        out_s.append(sc[0][m].numpy()); out_i.append(np.full(n, i))
    if not out_b:
        return np.zeros((0, 7))
    return detections(np.concatenate(out_b), np.concatenate(out_i),
                      np.concatenate(out_c), np.concatenate(out_s))


def time_torch(model, x, n, threads):
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    with torch.no_grad():
        for i in range(10):
            model(x[i:i + 1])
        t0 = time.perf_counter()
        for i in range(n):
            model(x[i % len(x):i % len(x) + 1])
        dt = (time.perf_counter() - t0) / n * 1e3
    torch.set_num_threads(old)
    return dt


def time_onnx(path, xs, n, threads):
    s = sess(path, threads)
    for i in range(10):
        s.run(None, {"image": xs[i:i + 1]})
    t0 = time.perf_counter()
    for i in range(n):
        s.run(None, {"image": xs[i % len(xs):i % len(xs) + 1]})
    return (time.perf_counter() - t0) / n * 1e3


def time_decode(model, x, n):
    """Post-processing is part of latency. A heatmap is not a detection."""
    with torch.no_grad():
        h, w, o = model(x[:1])
        for _ in range(10):
            D.decode(h, w, o)
        t0 = time.perf_counter()
        for _ in range(n):
            D.decode(h, w, o)
    return (time.perf_counter() - t0) / n * 1e3


def main(ckpt, regime, path, n_time, n_check):
    model, size = load(ckpt)
    nbytes = export(model, size, path)
    va_i, va_b, _ = C.load_split(ROOT, regime, "val")
    gts = va_b[:, :6].astype(np.float64)

    xs = np.ascontiguousarray(
        ((va_i[:n_time].astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2))
    xt = torch.from_numpy(xs)

    # --- correctness -------------------------------------------------------
    s1 = sess(path, 1)
    with torch.no_grad():
        t_out = [v.numpy() for v in model(xt[:32])]
    o_out = [np.concatenate([s1.run(None, {"image": xs[i:i + 1]})[j] for i in range(32)])
             for j in range(3)]
    diffs = [float(np.abs(a - b).max()) for a, b in zip(t_out, o_out)]

    d_torch = predict(model, va_i[:n_check], size)
    d_onnx = onnx_predict(s1, va_i[:n_check], size)
    g = gts[gts[:, 0] < n_check]
    m_torch = AP.evaluate(d_torch, g)["mAP"]
    m_onnx = AP.evaluate(d_onnx, g)["mAP"]

    # --- latency -----------------------------------------------------------
    lat = {
        "torch_eager_8t": time_torch(model, xt, n_time, 8),
        "torch_eager_4t": time_torch(model, xt, n_time, 4),
        "torch_eager_1t": time_torch(model, xt, n_time, 1),
        "onnxrt_8t": time_onnx(path, xs, n_time, 8),
        "onnxrt_4t": time_onnx(path, xs, n_time, 4),
        "onnxrt_1t": time_onnx(path, xs, n_time, 1),
    }
    dec = time_decode(model, xt, n_time)

    print(f"\n=== ONNX export: {os.path.basename(path)} ({nbytes/1e6:.2f} MB, "
          f"{model.n_params():,} params) ===")
    print(f"  max |torch - onnx| per head   hm {diffs[0]:.2e}  wh {diffs[1]:.2e}  "
          f"off {diffs[2]:.2e}")
    print(f"  mAP on {n_check} val images    torch {m_torch:.4f}   onnx {m_onnx:.4f}   "
          f"{'IDENTICAL' if abs(m_torch-m_onnx) < 5e-5 else 'DIFFERENT -- export is wrong'}")
    print(f"\n  {'runtime':>16} {'ms/img':>8} {'vs baseline':>12}")
    base = 2.02      # classical bgsub+ws, hard, from step 2b
    for k, v in sorted(lat.items(), key=lambda kv: kv[1]):
        print(f"  {k:>16} {v:8.3f} {base/v:11.2f}x")
    print(f"  {'decode only':>16} {dec:8.3f}   <- runtime-independent, add to any row above")
    best = min(lat, key=lat.get)
    print(f"\n  best {best} {lat[best]:.3f} ms + decode {dec:.3f} ms = "
          f"{lat[best]+dec:.3f} ms/img end to end")

    res = {"ckpt": ckpt, "regime": regime, "bytes": nbytes, "params": model.n_params(),
           "max_diff": diffs, "mAP_torch": float(m_torch), "mAP_onnx": float(m_onnx),
           "latency_ms": lat, "decode_ms": dec, "classical_ms": base}
    return res


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", default="runs/det_hard_none.pt")
    a.add_argument("--regime", choices=C.REGIMES, default="hard")
    a.add_argument("--path", default="runs/detector.onnx")
    a.add_argument("--n-time", type=int, default=200)
    a.add_argument("--n-check", type=int, default=500)
    a.add_argument("--save", default="runs/onnx.json")
    a = a.parse_args()
    r = main(a.ckpt, a.regime, a.path, a.n_time, a.n_check)
    if a.save:
        json.dump(r, open(a.save, "w"), indent=2)
        print(f"\nwrote {a.save}")
