"""Step 6c: the mAP-versus-latency curve, from the JSON the other scripts wrote.

Reads runs/onnx.json (fp32 PyTorch and ONNX Runtime), runs/quant_*.json and
runs/prune_*.json, and draws one figure: model latency at ONE thread on the
x axis, mAP on the y axis, every graph a labelled point. One thread because
that is the deployment question step 5 left open -- on a single core the fp32
detector lost to the classical pipeline -- and because a 1-thread number is the
only one that does not depend on how many cores the box happened to be given.

Files tagged `_smoke` are skipped: they were scored on a 100-image subset and
are not comparable.

    python edge_curve.py --out out/edge_curve.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

CLASSICAL_MS = 2.13   # bgsub+ws, hard, step 5 re-timing
CLASSICAL_MAP = 0.5322


def points():
    pts = []
    if os.path.exists("runs/onnx.json"):
        o = json.load(open("runs/onnx.json"))
        pts.append(("fp32 PyTorch", o["latency_ms"]["torch_eager_1t"], o["mAP_torch"],
                    o["params"], o["bytes"]))
        pts.append(("fp32 ORT", o["latency_ms"]["onnxrt_1t"], o["mAP_onnx"],
                    o["params"], o["bytes"]))
    for p in sorted(glob.glob("runs/quant_*.json")):
        if "_smoke" in p:
            continue
        q = json.load(open(p))
        pts.append((f"INT8 {q['method']}", q["latency_ms"]["1t"], q["score"]["mAP"],
                    None, q["bytes"]))
    for p in sorted(glob.glob("runs/prune_*.json")):
        if "_smoke" in p:
            continue
        r = json.load(open(p))
        pct = int(round(100 * r["ratio"]))
        pts.append((f"pruned {pct}%", r["latency_ms"]["onnxrt_1t"], r["score"]["mAP"],
                    r["params"], r["bytes"]))
        if "int8" in r:
            pts.append((f"pruned {pct}% + INT8", r["int8"]["latency_ms"]["onnxrt_1t"],
                        r["int8"]["score"]["mAP"], r["params"], r["int8"]["bytes"]))
    return pts


def table(pts):
    lines = ["| graph | ms/img, 1 thread | mAP | params | MB |", "|---|---|---|---|---|"]
    for name, ms, m, params, nbytes in pts:
        lines.append(f"| {name} | {ms:.3f} | {m:.4f} | "
                     f"{'--' if params is None else f'{params:,}'} | {nbytes/1e6:.2f} |")
    lines.append(f"| classical bgsub+ws | {CLASSICAL_MS:.2f} | {CLASSICAL_MAP:.4f} | -- | -- |")
    return "\n".join(lines)


def draw(pts, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ink, accent, muted = "#2b2f36", "#4f7d93", "#9aa3ad"
    for name, ms, m, _, _ in pts:
        ax.scatter(ms, m, s=42, color=accent if "INT8" in name or "pruned" in name else ink,
                   zorder=3)
        ax.annotate(name, (ms, m), xytext=(6, 4), textcoords="offset points", fontsize=8,
                    color=ink)
    ax.scatter(CLASSICAL_MS, CLASSICAL_MAP, s=42, color=muted, zorder=3)
    ax.annotate("classical bgsub+ws", (CLASSICAL_MS, CLASSICAL_MAP), xytext=(6, 4),
                textcoords="offset points", fontsize=8, color=muted)
    ax.axvline(CLASSICAL_MS, color=muted, lw=0.8, ls="--")
    ax.set_xlabel("model latency, ms per image, 1 thread (decode not included)")
    ax.set_ylabel("mAP@[.5:.95], hard/val")
    ax.set_title("Edge-AI step: what each graph costs on one core", fontsize=10, loc="left")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="out/edge_curve.png")
    args = a.parse_args()
    pts = points()
    if not pts:
        raise SystemExit("no runs/onnx.json, runs/quant_*.json or runs/prune_*.json found")
    print(table(pts))
    print(f"\nwrote {draw(pts, args.out)}")


if __name__ == "__main__":
    main()
