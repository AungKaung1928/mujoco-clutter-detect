"""The tracked fp32 ONNX graph must still be the tracked `hard` checkpoint (no dataset needed).

runs/detector.onnx is what the ONNX, INT8 and C++ rows of the README start from;
runs/det_hard_none.pt is what the accuracy and real-photo rows use. If one is
retrained or re-exported without the other, every row still prints but describes
a different network. Same seeded batch through both, per head; the tolerance is
about 10x the largest difference export_onnx.py recorded (runs/onnx.json: 1.1e-5).

Run:  python test_onnx_parity.py
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

import detector as D

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-4

torch.set_num_threads(1)
ck = torch.load(os.path.join(HERE, "runs", "det_hard_none.pt"), map_location="cpu", weights_only=False)
model = D.build(ck)
model.load_state_dict(ck["model"])
model.eval()
x = np.random.default_rng(0).normal(size=(4, 3, 192, 192)).astype(np.float32)
with torch.no_grad():
    ref = [t.numpy() for t in model(torch.from_numpy(x))]
sess = ort.InferenceSession(os.path.join(HERE, "runs", "detector.onnx"), providers=["CPUExecutionProvider"])
got = [np.concatenate(h) for h in zip(*(sess.run(["hm", "wh", "off"], {"image": x[i:i + 1]}) for i in range(len(x))))]
diffs = [float(np.abs(a - b).max()) for a, b in zip(ref, got)]
recorded = json.load(open(os.path.join(HERE, "runs", "onnx.json")))["max_diff"]
# all(d < TOL), not max(diffs) < TOL: builtin max skips a NaN that is not first.
ok = all(d < TOL for d in diffs) and all(r < TOL for r in recorded)
print(f"  {'ok  ' if ok else 'FAIL'}  tracked detector.onnx vs det_hard_none.pt, 4 seeded inputs: "
      f"max |diff| hm/wh/off {diffs[0]:.1e} / {diffs[1]:.1e} / {diffs[2]:.1e} (< {TOL})")
sys.exit(0 if ok else 1)
