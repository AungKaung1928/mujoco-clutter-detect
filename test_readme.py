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

print("key point: appearance randomisation vs augmentation (runs/ablation.json)")
ab = load("ablation.json")
en, ep, hn, hp = ab["easy/none"], ab["easy/photo"], ab["hard/none"], ab["hard/photo"]
claim = (f"drops {en['mAP']:.4f} → {en['cross']['mAP']:.4f} under appearance shift (detection rate "
         f"{en['cross']['det_rate']:.2f}); photometric augmentation recovers only to {ep['cross']['mAP']:.4f}; "
         f"on top of the randomiser it adds {hp['mAP'] - hn['mAP']:+.4f} for "
         f"{100 * (hp['train_s'] / hn['train_s'] - 1):.0f} % more training time, and the randomised model loses "
         f"only {hn['mAP'] - hn['cross']['mAP']:.3f} going back to `easy`")
check("ablation key point", claim in " ".join(LINES), f"README must contain {claim!r}")

print("key points: classical vs detector, pruning")
det, cv = load("det_hard_none.json"), load("baseline_hard.json")["bgsub+ws"]
spread = lambda apc: max(apc) - min(apc)
mmq, p25, p50 = load("quant_minmax.json"), load("prune_25.json"), load("prune_50.json")
claim = (f"per-class spread falls {spread(cv['ap_per_class']):.3f} → {spread(det['ap_per_class']):.3f} and recall on "
         f"objects under half visible goes {cv['vis_recall']['0.0-0.5']:.3f} → {det['vis_recall']['0.0-0.5']:.3f} "
         f"(n = {mmq['score']['vis_n']['0.0-0.5']})")
check("classical vs detector key point", claim in " ".join(LINES), f"README must contain {claim!r}")
fp32 = mmq["ref"]["latency_ms"]["1t"]
cut = [round(100 * (1 - p["latency_ms"]["onnxrt_1t"] / fp32)) for p in (p25, p50)]
both = p50["int8"]["latency_ms"]["onnxrt_1t"]
claim = (f"25 / 50 % of channels buy {cut[0]} / {cut[1]} % less latency "
         f"({p25['latency_ms']['onnxrt_1t']:.2f} / {p50['latency_ms']['onnxrt_1t']:.2f} ms) for "
         f"−{p25['mAP_drop']:.3f} / −{p50['mAP_drop']:.3f} mAP after fine-tuning; pruned 50 % + INT8 reaches {both:.2f} ms")
check("pruning key point", claim in " ".join(LINES) and both > 0.9 * mmq["latency_ms"]["1t"],
      f"README must contain {claim!r}; and 50 % + INT8 under 10 % faster than INT8")

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
onx = load("real_hope_onnx.json")["models"]
m = lambda n: f"{onx[n]['mAP']:.4f}"
sentence = (f"(mAP {m('detector.onnx')})", f"static INT8 (calibrated on sim images only) gives {m('detector_int8_minmax.onnx')}",
            f"dynamic INT8 {m('detector_int8_dynamic.onnx')}.")
check("real-photo ONNX rows", all(p in " ".join(" ".join(LINES).split()) for p in sentence)
      and onx["detector.onnx"] == {**onx["det_hard_none.pt"]}, f"README must contain {sentence}; fp32 ONNX row must equal torch row")
h10 = hope["models"]["det_hard_none.pt"]
agn = (f"`hard` gets mAP {h10['agnostic_mAP']:.4f} ({h10['mAP']:.4f} per class) "
       f"and recall@0.5 {h10['agnostic_recall50']:.3f}")
check("class-agnostic row", agn in "\n".join(LINES), f"README must contain {agn!r}")
c100 = k100["models"]["det_hard_none.pt"]
chance = (f"recall@0.5 {h10['chance_agnostic_recall50']:.3f} (top 10) and {c100['chance_agnostic_recall50']:.3f} (top 100)",
          f"class-agnostic mAP {h10['agnostic_mAP']:.4f} against {h10['chance_agnostic_mAP']:.4f} by chance")
hc = h10["dets_above_0.3_by_class"]
check("top-100 vs top-10 mAP under 0.001",
      all(abs(k100["models"][m]["mAP"] - hope["models"][m]["mAP"]) < 0.001 for m in hope["models"])
      and "keeping 100 peaks instead of 10 changes mAP by under 0.001" in " ".join(LINES), "")
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

print("Step 7, test-time corrections (runs/real_hope_gap.json)")
gap = load("real_hope_gap.json")
gh = gap["models"]["hard"]
check("gap raw row reproduces real_hope.json",
      [gh["rows"]["raw"][k] for k in ("mAP", "agnostic_mAP")] == [h10["mAP"], h10["agnostic_mAP"]])
for name, label in (("raw", "as taken"), ("colour", "colour matched (per-channel mean/std of the sim set)"),
                    ("blur", f"blurred (σ {gh['blur_sigma']} px, Laplacian variance {gh['blurred_laplacian_var_median']})"),
                    ("both", "blurred, then colour matched")):
    r = gh["rows"][name]
    row = (f"| {label} | {r['mAP']:.4f} | {r['agnostic_mAP']:.4f} | "
           f"{r['recall50_box']:.3f} / {r['recall50_cylinder']:.3f} |")
    check(f"gap row {name}", row in LINES, f"README must contain {row!r}")
stats_claim = (f"median Laplacian variance {gap['real']['laplacian_var_median']} against\n"
               f"{gh['sim']['laplacian_var_median']} for the `hard` sim set, and more saturated "
               f"({gap['real']['saturation_mean']} against {gh['sim']['saturation_mean']})")
check("gap image statistics", stats_claim in text, f"README must contain {stats_claim!r}")
d = gh["rows"]["blur"]["mAP"] - gh["rows"]["raw"]["mAP"]
check("blur moves mAP by", f"blur moves mAP by {d:.4f}" in " ".join(LINES)
      and gh["rows"]["colour"]["mAP"] < gh["rows"]["raw"]["mAP"], f"{d:.4f}; colour must be below raw")
cf = gh["confidence"]
lo, hi = cf["ci95_hit_minus_chance_by_scene"]
conf = (f"median {cf['real']['top_score_median']:.3f} (", f"{cf['sim']['top_score_median']:.3f} on the\nsim set), it keeps "
        f"{cf['real']['dets_above_thr_per_image']:.2f} detections above {cf['thr']} per photo ({cf['sim']['dets_above_thr_per_image']:.2f} in sim) "
        f"at a median side of\n{cf['real']['det_side_px_median']} px", f"{cf['hit50']:.3f} of those sit",
        f"against {cf['chance_hit50']:.3f} for the same boxes", f"{lo:.3f} to {hi:.3f} by scene bootstrap")
check("confident-detection hit rate", all(p in text for p in conf) and lo > 0, f"README must contain {conf}; CI above 0")

print("Step 7, render detail check (runs/render_detail.json)")
rd = load("render_detail.json")
st = {k: v["laplacian_var_median"] for k, v in rd["sets"].items()}
claim = (f"Median Laplacian variance is {st['tex']} for `tex` and\n{st['texd']} for `texd` against "
         f"{rd['real']['laplacian_var_median']} on the photos (`hard` {st['hard']})",
         f"(pixel noise) reaches {st['hard_aug_photo']}", f"overshoots\n({rd['sets']['tex']['saturation_mean']} against "
         f"{rd['real']['saturation_mean']})")
check("render detail check", all(p in text for p in claim) and rd["real"] == gap["real"]
      and max(st.values()) < rd["real"]["laplacian_var_median"] / 2, f"README must contain {claim}")

print("C++ detector (runs/cpp_ort.json)")
co = load("cpp_ort.json")
cp, pp, par = co["cpp_ms_p50"], co["python_ms_p50"], co["parity"]
claims = (f"same ORT {co['ort']} as the Python wheel",
          f"All {int(co['images'].split(':')[1].rstrip(']')) * 10:,} top-10 rows ({par['detections_above_thr']:,} above the 0.02 threshold)",
          f"{co['images_with_tie_at_k']} of 200 images have a tie",
          f"model only: C++ {cp['model_ms']:.3f} ms, Python {pp['model_ms']:.3f} ms",
          f"End to end: C++ {cp['e2e_ms']:.3f} ms, Python {pp['e2e_ms']:.3f} ms; decode alone is {cp['decode_ms']:.3f} ms",
          f"drops from {pp['e2e_ms'] - pp['model_ms']:.2f} ms to {cp['e2e_ms'] - cp['model_ms']:.2f} ms")
check("C++ detector paragraph", all(c in text for c in claims) and par["max_abs_diff"] == 0
      and par["class_mismatch"] == par["index_mismatch"] == 0, f"README must contain {claims}")

print("C++ vs classical (runs/cpp_ort_classical*.json)")
cc = [load(f"cpp_ort_classical{s}.json") for s in ("", "_r2", "_r3")]
trio = lambda k: " / ".join(f"{r[k]['e2e_ms']:.3f}" for r in cc)
claims = (f"C++ INT8 end to end {trio('cpp_ms_p50')} ms, classical {trio('classical_ms_p50')} ms, "
          f"Python INT8 end to end {trio('python_ms_p50')} ms",)
check("C++ vs classical paragraph", all(c in text for c in claims)
      and all(r["cpp_ms_p50"]["e2e_ms"] < r["classical_ms_p50"]["e2e_ms"] < r["python_ms_p50"]["e2e_ms"] for r in cc)
      and all(r["parity"]["max_abs_diff"] == 0 for r in cc), f"README must contain {claims}")

print("Step 7, detail by region (runs/real_hope_gap.json, runs/render_detail.json)")
O, B = "laplacian_var_objects_median", "laplacian_var_background_median"
rs, hs, es = gap["real_split"], gap["models"]["hard"]["sim_split"], gap["models"]["easy"]["sim_split"]
sp = rd["split"]
region = (f"{rs[O]} / {rs[B]} and the `hard` sim set {hs[O]} / {hs[B]}: the background gap\n"
          f"({rs[B] / hs[B]:.0f}×) dwarfs the object gap ({rs[O] / hs[O]:.1f}×)",
          f"distractors lift the background to {sp['texd'][B]}, `tex` leaves it at {sp['tex'][B]}",
          f"reaches {es[O]} inside its boxes on edges alone")
check("detail by region", all(p in text for p in region) and rd["real_split"] == rs
      and rs[B] / hs[B] > rs[O] / hs[O], f"README must contain {region}")

rn = gap["real"]["noise_sigma_median"]
an = rd["sets"]["hard_aug_photo"]["noise_sigma_median"]
noise = (f"σ at {rn} grey levels at most (sim {gap['models']['hard']['sim']['noise_sigma_median']}), and white noise that size adds at most 20σ² = "
         f"{20 * rn ** 2:.0f} to the\nLaplacian variance of {gap['real']['laplacian_var_median']}",
         f"matches that noise (σ {an}) and\nstill stops at {rd['sets']['hard_aug_photo']['laplacian_var_median']}")
check("noise vs texture", all(p in text for p in noise) and 20 * rn ** 2 < gap["real"]["laplacian_var_median"] / 2,
      f"README must contain {noise}")

bm = gap["models"]["hard"]["background_masked"]
gr, bl = bm["background_grey"], bm["background_blurred"]
raw = gap["models"]["hard"]["rows"]["raw"]
masked = (f"`hard` reaches mAP {gr['mAP']:.4f} (class-agnostic {gr['agnostic_mAP']:.4f}) and {gr['hit50_above_0.3']:.3f} of its confident boxes",
          f"blurred, {bl['mAP']:.4f} and {bl['hit50_above_0.3']:.3f}; as taken, {raw['mAP']:.4f} and {gap['models']['hard']['confidence']['hit50']:.3f}")
check("background-masked diagnostic", all(p in text for p in masked) and gr["mAP"] < 0.02,
      f"README must contain {masked}; and with no background mAP must stay near 0")

fl = bm["background_grey_objects_mean_colour"]
sim_agn = load("det_hard_none.json")["agnostic_mAP"]
flat = (f"lifts class-agnostic mAP from {gr['agnostic_mAP']:.4f}\nto {fl['agnostic_mAP']:.4f} and confident boxes on an object from "
        f"{gr['hit50_above_0.3']:.3f} to {fl['hit50_above_0.3']:.3f}", f"still score far below the sim's own\n{sim_agn:.4f}")
check("objects flattened diagnostic", all(p in text for p in flat) and fl["agnostic_mAP"] > 2 * gr["agnostic_mAP"]
      and fl["agnostic_mAP"] < 0.5 * sim_agn, f"README must contain {flat}")

texl = (f"Detail inside the object boxes rises to {sp['texl'][O]} against\n{sp['tex'][O]} for `tex` and {rs[O]} on the photos",)
check("texl detail", all(p in text for p in texl) and sp["tex"][O] < sp["texl"][O] < rs[O] * 1.5, f"README must contain {texl}")

ts = load("texture_shift.json")["regimes"]
shift = (f"the `hard` model scores class-agnostic mAP {ts['hard']['agnostic_mAP']:.4f} on\n`hard` renders",
         f"{ts['tex']['agnostic_mAP']:.4f} on `tex` and {ts['texl']['agnostic_mAP']:.4f} on `texl`")
check("texture shift in sim", all(p in text for p in shift) and abs(ts["hard"]["agnostic_mAP"] - 0.91) < 0.02
      and ts["texl"]["agnostic_mAP"] < 0.2, f"README must contain {shift}")

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
