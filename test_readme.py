"""README headline table against the committed result files (no model, stdlib only).

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

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
