"""Step 6a: post-training INT8 quantization of the exported detector, and what it
costs in detections rather than in tensor error.

Static, not dynamic, for a convolutional network. Dynamic quantization stores
INT8 weights and computes activation scales at run time from each tensor's
observed range; on a CNN that leaves the convolutions running in fp32 with an
extra quantize step in front of each one, so it saves bytes and buys no speed.
Static quantization fixes activation scales from a calibration pass, so the
runtime can execute the whole conv chain in INT8 with fused requantization.
Both are measured here; the dynamic row exists to show the difference, not as
a candidate.

QDQ format rather than QOperator. QDQ inserts standard ONNX Quantize/Dequantize
nodes around each op and leaves the operator graph readable and portable; ONNX
Runtime fuses the QDQ pairs into QLinearConv at session creation, so the
executed graph is the same either way. QOperator writes the fused contrib ops
directly and ties the file to one runtime. Activations unsigned, weights
signed per channel: the U8S8 combination is what the x86 INT8 kernels want, and
per-channel weight scales are the single largest accuracy lever in PTQ.

Calibration images come from the *training* split, never from val. Val is what
the INT8 graph is scored on; feeding it the same images it was calibrated on
would make the accuracy drop look smaller than it is on new frames.

Verification is end to end, as in step 5. An INT8 graph is a different network
and a max-abs-diff on its heatmap says nothing about whether the peaks moved. So
the INT8 graph goes through the same `onnx_predict` and the same `ap.evaluate`
the fp32 graph did, and the number that matters is the mAP drop, per class,
and the recall on occluded objects -- the small, low-contrast detections that
quantization noise takes first.

    nice -n 10 python quantize.py --calib 200 --methods minmax percentile entropy dynamic
"""
import argparse
import json
import os
import time

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

import ap as AP
import common as C
from export_onnx import onnx_predict, sess, time_onnx
from train_det import MEAN, STD

ROOT = os.path.dirname(os.path.abspath(__file__))
METHODS = {"minmax": CalibrationMethod.MinMax,
           "percentile": CalibrationMethod.Percentile,
           "entropy": CalibrationMethod.Entropy}
VIS_EDGES = [0.0, 0.5, 0.9, 1.01]


def preprocess(imgs):
    """uint8 NHWC -> float32 NCHW, the exact transform the graph was trained on."""
    x = imgs.astype(np.float32) / 255.0
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2))


class ImageReader(CalibrationDataReader):
    """Feeds `n` images one at a time. The graph has batch fixed at 1."""

    def __init__(self, imgs, n, input_name="image"):
        self.xs = preprocess(np.asarray(imgs[:n]))
        self.i = 0
        self.name = input_name

    def get_next(self):
        if self.i >= len(self.xs):
            return None
        x = self.xs[self.i:self.i + 1]
        self.i += 1
        return {self.name: x}

    def rewind(self):
        self.i = 0


def pre_process(src, dst):
    """Shape inference + graph optimisation before quantization. Without it the
    quantizer sees unfused Conv/BN pairs and unknown intermediate shapes, and
    quantizes them separately."""
    quant_pre_process(src, dst, skip_symbolic_shape=False)
    return dst


def quantize(pre_path, out_path, method, calib_imgs, n_calib, per_channel=True):
    """One INT8 graph. `method` is a key of METHODS or 'dynamic'."""
    t0 = time.perf_counter()
    if method == "dynamic":
        # Unsigned weights: dynamic quantization rewrites Conv as ConvInteger,
        # and the CPU provider implements ConvInteger for uint8 x uint8 only.
        # A signed-weight dynamic graph builds fine and fails at session load.
        quantize_dynamic(pre_path, out_path, per_channel=per_channel,
                         weight_type=QuantType.QUInt8)
    else:
        reader = ImageReader(calib_imgs, n_calib)
        extra = {}
        if method == "percentile":
            extra = {"CalibPercentile": 99.99}
        quantize_static(pre_path, out_path, reader, quant_format=QuantFormat.QDQ,
                        per_channel=per_channel, reduce_range=False,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=METHODS[method], extra_options=extra)
    return time.perf_counter() - t0


def score(path, imgs, gts, vf, size, n_eval, threads=1):
    """End-to-end detections through the graph, scored by ap.py."""
    s = sess(path, threads)
    n = len(imgs) if n_eval <= 0 else min(n_eval, len(imgs))
    dets = onnx_predict(s, imgs[:n], size)
    g_sel = gts[:, 0] < n
    g = gts[g_sel]
    r = AP.evaluate(dets, g)
    rv = AP.recall_by_visibility(dets, g, vf[g_sel], VIS_EDGES)
    return {"n_images": int(n), "n_dets": int(len(dets)),
            "mAP": float(r["mAP"]), "AP50": float(r["AP50"]), "AP75": float(r["AP75"]),
            "ap_per_class": [float(v) for v in r["ap_per_class"]],
            "vis_recall": {f"{lo:.1f}-{hi:.1f}": v["recall"] for (lo, hi), v in rv.items()},
            "vis_n": {f"{lo:.1f}-{hi:.1f}": v["n_gt"] for (lo, hi), v in rv.items()}}


def n_quantized_nodes(path):
    m = onnx.load(path)
    ops = [n.op_type for n in m.graph.node]
    return {"QuantizeLinear": ops.count("QuantizeLinear"),
            "DequantizeLinear": ops.count("DequantizeLinear"),
            "Conv": ops.count("Conv"), "ConvInteger": ops.count("ConvInteger"),
            "MatMulInteger": ops.count("MatMulInteger"), "total": len(ops)}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--onnx", default="runs/detector.onnx")
    a.add_argument("--regime", choices=C.REGIMES, default="hard")
    a.add_argument("--calib", type=int, default=200, help="training images for calibration")
    a.add_argument("--methods", nargs="+", default=["minmax", "percentile", "entropy", "dynamic"])
    a.add_argument("--n-eval", type=int, default=0, help="val images to score, 0 = all")
    a.add_argument("--n-time", type=int, default=200)
    a.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    a.add_argument("--tag", default="", help="suffix for output files, e.g. smoke")
    args = a.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    tr_i, _, meta = C.load_split(ROOT, args.regime, "train")
    va_i, va_b, _ = C.load_split(ROOT, args.regime, "val")
    size = meta["img_size"]
    gts = va_b[:, :6].astype(np.float64)
    vf = va_b[:, 6].astype(np.float64)
    xs = preprocess(np.asarray(va_i[:args.n_time]))
    load1 = float(open("/proc/loadavg").read().split()[0])

    pre = pre_process(args.onnx, f"runs/detector_pre{tag}.onnx")
    print(f"pre-processed graph: {os.path.getsize(pre)/1e6:.2f} MB  {n_quantized_nodes(pre)}")

    # fp32 reference through the SAME pre-processed graph and the SAME subset,
    # so every delta below is quantization and nothing else.
    ref = {"path": pre, "bytes": os.path.getsize(pre),
           "score": score(pre, va_i, gts, vf, size, args.n_eval),
           "latency_ms": {f"{t}t": time_onnx(pre, xs, args.n_time, t) for t in args.threads}}
    print(f"fp32 ORT   mAP {ref['score']['mAP']:.4f} on {ref['score']['n_images']} images   "
          + "  ".join(f"{k} {v:.3f} ms" for k, v in ref["latency_ms"].items()))

    rows = []
    for method in args.methods:
        out = f"runs/detector_int8_{method}{tag}.onnx"
        q_s = quantize(pre, out, method, tr_i, args.calib)
        sc = score(out, va_i, gts, vf, size, args.n_eval)
        lat = {f"{t}t": time_onnx(out, xs, args.n_time, t) for t in args.threads}
        res = {"method": method, "path": out, "bytes": os.path.getsize(out),
               "calib_images": 0 if method == "dynamic" else args.calib,
               "calib_split": "train", "quantize_s": q_s, "nodes": n_quantized_nodes(out),
               "score": sc, "latency_ms": lat, "ref": ref,
               "mAP_drop": ref["score"]["mAP"] - sc["mAP"],
               "ap_drop_per_class": [float(r - q) for r, q in
                                     zip(ref["score"]["ap_per_class"], sc["ap_per_class"])],
               "speedup_1t": ref["latency_ms"].get("1t", float("nan")) / lat.get("1t", float("nan")),
               "measured": {"n_eval": sc["n_images"], "n_time": args.n_time,
                            "load1_at_start": load1, "regime": args.regime,
                            "decode_note": "model only; add decode_ms from runs/onnx.json"}}
        rows.append(res)
        json.dump(res, open(f"runs/quant_{method}{tag}.json", "w"), indent=2)
        print(f"{method:10s} mAP {sc['mAP']:.4f} ({-res['mAP_drop']:+.4f})   "
              + "  ".join(f"{k} {v:.3f} ms" for k, v in lat.items())
              + f"   {res['bytes']/1e6:.2f} MB   quantized in {q_s:.1f} s")

    print(f"\n=== INT8, {args.regime}/val, {ref['score']['n_images']} images"
          f"{' (SMOKE SUBSET)' if args.n_eval else ''}, calibration {args.calib} train images ===")
    hdr = "| graph | mAP | Δ mAP | " + " | ".join(f"ms {k}" for k in ref["latency_ms"]) \
        + " | MB | " + " | ".join(f"AP {c}" for c in C.CLASSES) + " | recall vis<0.5 |"
    print(hdr)
    print("|---" * (hdr.count("|") - 1) + "|")

    def row(name, sc, lat, nbytes, ref_map=None):
        d = "--" if ref_map is None else f"{sc['mAP'] - ref_map:+.4f}"
        return (f"| {name} | {sc['mAP']:.4f} | {d} | "
                + " | ".join(f"{v:.3f}" for v in lat.values())
                + f" | {nbytes/1e6:.2f} | " + " | ".join(f"{v:.3f}" for v in sc["ap_per_class"])
                + f" | {sc['vis_recall']['0.0-0.5']:.3f} (n={sc['vis_n']['0.0-0.5']}) |")
    print(row("fp32 ORT", ref["score"], ref["latency_ms"], ref["bytes"]))
    for r in rows:
        print(row(f"INT8 {r['method']}", r["score"], r["latency_ms"], r["bytes"],
                  ref["score"]["mAP"]))
    print("\nwrote " + ", ".join(f"runs/quant_{r['method']}{tag}.json" for r in rows))


if __name__ == "__main__":
    main()
