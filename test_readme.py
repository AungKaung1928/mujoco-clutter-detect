"""README headline table and the real-photo table against the committed result files (no model, stdlib only).

Each row of the "graph | mAP | Δ mAP | ms 1 thread | ms 4 threads" table at the top of
README.md must equal the JSON it was copied from. Δ is the exact difference, rounded
(not the difference of the two rounded mAPs: that made two cells 1e-4 off before this test).

Run:  python test_readme.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load(name):
    with open(os.path.join(HERE, "runs", name)) as f:
        return json.load(f)


with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    LINES = f.read().splitlines()


def row(prefix):
    for line in LINES:
        if line.startswith(f"| {prefix}"):
            return [c.strip().strip("*") for c in line.strip().strip("|").split("|")][1:]
    raise AssertionError(f"README row not found: {prefix}")


def delta(m, base):
    return f"{m - base:+.4f}".replace("-", "−")


mm, dyn, ent = load("quant_minmax.json"), load("quant_dynamic.json"), load("quant_entropy.json")
p25, p50 = load("prune_25.json"), load("prune_50.json")
onnx, base = load("onnx.json"), load("baseline_hard.json")
ref = mm["ref"]
B = ref["score"]["mAP"]

expected = {
    "classical `bgsub+ws`": [f"{base['bgsub+ws']['mAP']:.4f}", "—", f"{onnx['classical_ms']:.2f}", "—"],
    "fp32, ONNX Runtime": [f"{B:.4f}", "—", f"{ref['latency_ms']['1t']:.2f}", f"{ref['latency_ms']['4t']:.2f}"],
    "INT8 static, MinMax": [f"{mm['score']['mAP']:.4f}", delta(mm["score"]["mAP"], B),
                            f"{mm['latency_ms']['1t']:.2f}", f"{mm['latency_ms']['4t']:.2f}"],
    "INT8 dynamic": [f"{dyn['score']['mAP']:.4f}", delta(dyn["score"]["mAP"], B),
                     f"{dyn['latency_ms']['1t']:.2f}", f"{dyn['latency_ms']['4t']:.2f}"],
    "pruned 25 % channels": [f"{p25['score']['mAP']:.4f}", delta(p25["score"]["mAP"], B),
                             f"{p25['latency_ms']['onnxrt_1t']:.2f}", "—"],
    "pruned 50 % channels": [f"{p50['score']['mAP']:.4f}", delta(p50["score"]["mAP"], B),
                             f"{p50['latency_ms']['onnxrt_1t']:.2f}", "—"],
    "pruned 50 % + INT8": [f"{p50['int8']['score']['mAP']:.4f}", delta(p50["int8"]["score"]["mAP"], B),
                           f"{p50['int8']['latency_ms']['onnxrt_1t']:.2f}", "—"],
}

print("README headline table")
for prefix, want in expected.items():
    got = row(prefix)[:4]
    check(prefix, got == want, f"README {got} vs JSON {want}")

print("claims next to the table")
mm_line = next(l for l in LINES if l.startswith("| INT8 static, MinMax"))
check("MinMax = Entropy (same SHA-256)", mm["sha256"] == ent["sha256"] and "same SHA-256" in mm_line)

print("Step 7, real photos (runs/real_hope.json)")
hope = load("real_hope.json")
for prefix, ckpt in (("| `easy` (fixed appearance) |", "det_easy_none.pt"),
                     ("| `hard` (randomised hue, table, light) |", "det_hard_none.pt")):
    m = hope["models"][ckpt]
    line = next(l for l in LINES if l.startswith(prefix))
    got = [c.strip() for c in line.strip().strip("|").split("|")][1:]
    want = [f"{m['mAP']:.4f}", f"{m['AP50']:.4f}", f"{m['AP_box']:.4f}", f"{m['AP_cylinder']:.4f}",
            f"{m['recall50_box']:.3f} / {m['recall50_cylinder']:.3f}"]
    check(prefix, got == want, f"README {got} vs JSON {want}")
k100 = load("real_hope_k100.json")
h100 = k100["models"]["det_hard_none.pt"]
sentence = (f"caps recall at {hope['max_recall_at_k']:.3f} here\n({hope['images_over_k']} of {hope['images']} photos",
            f"`hard` reaches mAP {h100['mAP']:.4f}, recall@0.5 {h100['recall50_box']:.3f} box / "
            f"{h100['recall50_cylinder']:.3f} cylinder")
check("top-10 cap and top-100 row", all(p in "\n".join(LINES) for p in sentence) and k100["k"] == 100,
      f"README must contain {sentence}")
h10 = hope["models"]["det_hard_none.pt"]
agn = (f"`hard` gets mAP {h10['agnostic_mAP']:.4f} ({h10['mAP']:.4f} per class) "
       f"and recall@0.5 {h10['agnostic_recall50']:.3f}")
check("class-agnostic row", agn in "\n".join(LINES), f"README must contain {agn!r}")
c100 = k100["models"]["det_hard_none.pt"]
chance = (f"recall@0.5 {h10['chance_agnostic_recall50']:.3f} (top 10) and {c100['chance_agnostic_recall50']:.3f} (top 100)",
          f"class-agnostic mAP {h10['agnostic_mAP']:.4f} against {h10['chance_agnostic_mAP']:.4f} by chance")
hc = h10["dets_above_0.3_by_class"]
claim = f"calls {hc['box']} of {sum(hc.values())} detections `box`"
check("score > 0.3 class counts", claim in " ".join(LINES), f"README must contain {claim!r}")
lo, hi = h10["ci95_agnostic_mAP_minus_chance_by_scene"]
check("scene bootstrap interval", lo < 0 < hi and f"difference between {lo:.4f} and {hi:.4f} (95 %)"
      in " ".join(LINES), f"README must give [{lo}, {hi}] and the interval must contain 0")
check("chance control (boxes from another scene)",
      all(p in "\n".join(LINES) for p in chance) and hope["chance_pairs_same_scene"] == 0,
      f"README must contain {chance}")
boxes, cans = hope["objects"]["box"], hope["objects"]["cylinder"]
# an ignored object is entered once per class, so two entries are one object
per_image = (boxes + cans + hope["ignored_entries"] / 2) / hope["images"]
text = "\n".join(LINES)
check("objects scored and per image",
      f"{boxes} boxes and {cans} cans scored, {per_image:.1f} objects per image" in text)
check("median object size", f"median\n{hope['object_px_median']} px at 192 px" in text)

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
