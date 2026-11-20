"""Classical bgsub+ws vs the ONNX detector (fp32, INT8 MinMax) in ONE process, one thread.

The README's one-thread comparison put numbers from different sessions side by side
(classical 2.13 ms from step 5, INT8 1.38 ms from step 6). Timing moves between
sessions, so this times all of them on the same 200 hard/val images, in 10
interleaved rounds (order reversed every other round), and reports the median round.

    OMP_NUM_THREADS=1 OPENCV_NUM_THREADS=1 taskset -c 0 python edge_vs_classical.py --out runs/edge_vs_classical.json

Rows: classical = detect_one (segment + boxes) per image; fp32 / int8 = model only,
batch 1 (same as time_onnx); int8 end to end = onnx_predict per image (normalise,
model, decode, threshold), the path the scores come from.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import baseline_cv as BC
import common as C
from export_onnx import onnx_predict, sess
from quantize import preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    imgs, _, meta = C.load_split(ROOT, "hard", "val")
    size = meta["img_size"]
    raw = [np.asarray(imgs[i]) for i in range(a.n)]
    xs = preprocess(np.asarray(imgs[:a.n]))
    fp32 = sess(os.path.join(ROOT, "runs/detector_pre.onnx"), 1)
    int8 = sess(os.path.join(ROOT, "runs/detector_int8_minmax.onnx"), 1)

    def model(s):
        def f():
            for i in range(a.n):
                s.run(None, {"image": xs[i:i + 1]})
        return f

    def classical():
        for im in raw:
            BC.detect_one(im, "bgsub+ws", size)

    runs = {"classical bgsub+ws": classical, "fp32 model": model(fp32), "int8 model": model(int8),
            "int8 end to end": lambda: onnx_predict(int8, raw, size)}
    for f in runs.values():
        f()  # warm-up
    ts = {k: [] for k in runs}
    for r in range(a.rounds):
        for k in (list(runs) if r % 2 == 0 else list(reversed(list(runs)))):
            t0 = time.perf_counter()
            runs[k]()
            ts[k].append((time.perf_counter() - t0) / a.n * 1e3)
    out = {"n_images": a.n, "rounds": a.rounds, "threads": 1,
           "load1": float(open("/proc/loadavg").read().split()[0]),
           "ms_per_img": {k: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
                          for k, v in ts.items()}}
    for k, v in out["ms_per_img"].items():
        print(f"  {k:20s} median {v['median']:.2f} ms  (min {v['min']:.2f}, max {v['max']:.2f})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
