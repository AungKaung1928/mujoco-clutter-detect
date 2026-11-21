"""Per-operator time of each edge graph, ONNX Runtime built-in profiler, 1 thread.

    python edge/profile_ops.py --out runs/op_profile.json

Same 200 hard/val images for every graph, batch 1, after 20 warm-up runs. Reports
the mean ms per image by op type (kernel time only, from the profiler's node
events) and the session-run wall time, so the gap between the two is framework
overhead outside the kernels. For fp32 graphs it also counts the convolution MACs
the kernels execute, from the optimised graph's own tensor shapes.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common as C  # noqa: E402
from quantize import preprocess  # noqa: E402

GRAPHS = ["detector.onnx", "detector_pruned_50.onnx", "detector_int8_minmax.onnx", "detector_int8_dynamic.onnx"]


def profile(path, xs, warm=20):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.enable_profiling = True
    o.profile_file_prefix = os.path.join("/tmp", "ortprof_" + os.path.basename(path))
    s = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
    for i in range(warm):
        s.run(None, {"image": xs[i % len(xs)][None]})
    t = []
    for x in xs:
        t0 = time.perf_counter()
        s.run(None, {"image": x[None]})
        t.append(time.perf_counter() - t0)
    prof = s.end_profiling()
    ev = json.load(open(prof))
    os.remove(prof)
    by_op, runs, macs, seen = defaultdict(float), 0, 0, set()
    for e in ev:
        if e.get("cat") == "Session" and e.get("name") == "model_run":
            runs += 1
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            by_op[e["args"]["op_name"]] += e["dur"]
            a = e["args"]
            if a["op_name"] in ("Conv", "FusedConv") and a["node_index"] not in seen:
                # MACs the kernel executes: weight elements x output pixels, from the shapes
                # of the OPTIMISED graph (after ORT's layout transform pads channel counts)
                seen.add(a["node_index"])
                w = list(a["input_type_shape"][1].values())[0]
                y = list(a["output_type_shape"][0].values())[0]
                macs += int(np.prod(w)) * int(np.prod(y[2:]))
    assert runs == warm + len(xs), runs
    # kernel events cover warm-up too: per-image mean over all runs
    per = {k: round(v / runs / 1e3, 4) for k, v in sorted(by_op.items(), key=lambda kv: -kv[1])}
    return {"run_ms_median": round(float(np.median(t)) * 1e3, 4), "kernel_ms_mean": round(sum(per.values()), 4),
            "conv_macs_executed": macs or None, "by_op_ms": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    imgs = np.asarray(C.load_split(ROOT, "hard", "val")[0][:a.n])
    xs = preprocess(imgs)
    out = {"images": f"hard/val[:{a.n}]", "threads": 1, "ort": ort.__version__,
           "load1": float(open("/proc/loadavg").read().split()[0]), "graphs": {}}
    for g in GRAPHS:
        r = profile(os.path.join(ROOT, "runs", g), xs)
        out["graphs"][g] = r
        top = ", ".join(f"{k} {v:.3f}" for k, v in list(r["by_op_ms"].items())[:4])
        print(f"{g:28s} run {r['run_ms_median']:.3f} ms  kernels {r['kernel_ms_mean']:.3f} ms  [{top}]")
    if a.out:
        with open(os.path.join(ROOT, a.out), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
