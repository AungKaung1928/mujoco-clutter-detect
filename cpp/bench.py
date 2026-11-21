"""C++ vs Python ONNX Runtime for the detector: same graph, same images, same decode.

Writes cpp/images.u8 + cpp/ref.f32 (Python ORT + detector.decode, k = 10 per image), then
runs cpp/detect_ort and Python's onnx_predict in interleaved rounds (order reversed every
other round), one thread each, and reports the median round. The C++ library must be the
same ONNX Runtime version as the Python wheel, or bit-level parity means nothing.

    OMP_NUM_THREADS=1 taskset -c 0 python cpp/bench.py --out runs/cpp_ort.json   # hard/val
    python cpp/bench.py --synthetic --rounds 1                                      # CI, no data/
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import common as C  # noqa: E402
import detector as D  # noqa: E402
from export_onnx import onnx_predict, sess  # noqa: E402
from quantize import preprocess  # noqa: E402

K = 10


def synthetic(n, size=192, seed=0):
    """Flat-coloured rectangles and discs on a grey table: real peaks, no data/ needed."""
    import cv2
    rng = np.random.default_rng(seed)
    out = np.full((n, size, size, 3), 120, np.uint8)
    for im in out:
        for _ in range(rng.integers(3, 7)):
            c = tuple(int(v) for v in rng.integers(0, 256, 3))
            x, y = (int(v) for v in rng.integers(20, size - 20, 2))
            r = int(rng.integers(8, 18))
            if rng.random() < 0.5:
                cv2.rectangle(im, (x - r, y - r), (x + r, y + r), c, -1)
            else:
                cv2.circle(im, (x, y), r, c, -1)
    return out


def reference(s, imgs):
    """Per image, k rows (flat index, class, score, x0, y0, x1, y1).

    detector.decode with one fixed tie rule: score, then lower flat index. The INT8 graph
    puts the heatmap on a few dozen levels, so the k-th score is often shared and
    torch.topk may return any of the tied cells, in any order. Rows scoring above the k-th
    score must be the same set as torch's; rows at it only have to carry the same score.
    Returns (rows, number of images with a tie at k)."""
    rows, ties = [], 0
    for x in preprocess(imgs):
        hm, wh, off = (torch.from_numpy(a) for a in s.run(None, {"image": x[None]}))
        bx, cl, sc = D.decode(hm, wh, off, k=K)
        flat = (hm * (hm == F.max_pool2d(hm, 3, 1, 1)).float()).view(-1).numpy()
        idx = np.lexsort((np.arange(flat.size), -flat))[:K]
        hw = hm.shape[2] * hm.shape[3]
        pix = idx % hw
        w = hm.shape[3]
        o, sz = off.view(2, -1).numpy()[:, pix], wh.view(2, -1).numpy()[:, pix]
        cx = ((pix % w).astype(np.float32) + o[0]) * D.STRIDE
        cy = ((pix // w).astype(np.float32) + o[1]) * D.STRIDE
        bw, bh = np.maximum(sz[0], 0) * D.STRIDE, np.maximum(sz[1], 0) * D.STRIDE
        r = np.c_[idx, idx // hw, flat[idx], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2].astype(np.float32)
        assert np.array_equal(np.sort(sc[0].numpy()), np.sort(r[:, 2])), "scores differ from detector.decode"
        above = r[:, 2] > r[-1, 2]
        t = np.c_[cl[0].numpy(), sc[0].numpy(), bx[0].numpy()][:int(above.sum())]
        canon = lambda m: m[np.lexsort(m.T[::-1])]          # order-free: ties above the cut come in any order
        assert np.array_equal(canon(t), canon(r[above, 1:])), "rows above the k-th score differ from detector.decode"
        ties += int((flat == r[-1, 2]).sum() > (r[:, 2] == r[-1, 2]).sum())
        rows.append(r)
    return np.stack(rows), ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detector_int8_minmax.onnx")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-time", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--classical", action="store_true", help="also time the classical bgsub+ws pipeline per image")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.chdir(ROOT)
    if a.synthetic:
        imgs = synthetic(a.n)
    else:
        imgs = np.asarray(C.load_split(ROOT, "hard", "val")[0][:a.n])
    s = sess(a.model, 1)
    imgs.tofile(os.path.join(HERE, "images.u8"))
    ref, ties = reference(s, imgs)
    ref.tofile(os.path.join(HERE, "ref.f32"))
    cmd = [os.path.join(HERE, "detect_ort"), a.model, os.path.join(HERE, "images.u8"),
           os.path.join(HERE, "ref.f32"), str(a.n_time)]
    xs = preprocess(imgs)

    def cpp():
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode not in (0, 1):
            sys.exit(f"detect_ort failed: {p.stderr}")
        return json.loads(p.stdout)

    def python():
        tm, te = [], []
        for i in range(len(xs)):
            t0 = time.perf_counter()
            s.run(None, {"image": xs[i:i + 1]})
            tm.append(time.perf_counter() - t0)
        for i in range(len(imgs)):
            t0 = time.perf_counter()
            onnx_predict(s, imgs[i:i + 1], 192)
            te.append(time.perf_counter() - t0)
        q = lambda v, p: float(np.quantile(v, p)) * 1e3
        return {"model_ms": [q(tm, .5), q(tm, .99)], "e2e_ms": [q(te, .5), q(te, .99)]}

    def classical():
        import baseline_cv as BC
        t = []
        for im in imgs:
            t0 = time.perf_counter()
            BC.detect_one(im, "bgsub+ws", imgs.shape[1])
            t.append(time.perf_counter() - t0)
        return {"e2e_ms": [float(np.quantile(t, .5)) * 1e3, float(np.quantile(t, .99)) * 1e3]}

    fs = [cpp, python] + ([classical] if a.classical else [])
    for f in fs[1:]:
        f()  # warm-up; the C++ binary warms itself up
    runs = {f: [] for f in fs}
    for r in range(a.rounds):
        for f in (fs if r % 2 == 0 else fs[::-1]):
            runs[f].append(f())
    c_runs, p_runs = runs[cpp], runs[python]
    last = c_runs[-1]
    med = lambda rs, key, j=None: float(np.median([x[key][j] if j is not None else x[key] for x in rs]))
    out = {"model": a.model, "images": "synthetic" if a.synthetic else f"hard/val[:{a.n}]",
           "images_with_tie_at_k": ties, "rounds": a.rounds, "threads": 1, "ort": last["ort"],
           "load1": float(open("/proc/loadavg").read().split()[0]),
           "parity": {k: last[k] for k in ("max_abs_diff", "class_mismatch", "index_mismatch", "detections_above_thr")},
           "cpp_ms_p50": {k: round(med(c_runs, k, 0), 4) for k in ("model_ms", "decode_ms", "e2e_ms")},
           "cpp_ms_p99": {k: round(med(c_runs, k, 1), 4) for k in ("model_ms", "decode_ms", "e2e_ms")},
           "python_ms_p50": {k: round(med(p_runs, k, 0), 4) for k in ("model_ms", "e2e_ms")},
           "python_ms_p99": {k: round(med(p_runs, k, 1), 4) for k in ("model_ms", "e2e_ms")}}
    if a.classical:
        out["classical_ms_p50"] = {"e2e_ms": round(med(runs[classical], "e2e_ms", 0), 4)}
        out["classical_ms_p99"] = {"e2e_ms": round(med(runs[classical], "e2e_ms", 1), 4)}
    print(json.dumps(out, indent=1))
    ok = all(x["max_abs_diff"] < 1e-4 and x["class_mismatch"] == 0 and x["index_mismatch"] == 0 for x in c_runs)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
