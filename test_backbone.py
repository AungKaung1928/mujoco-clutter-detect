"""FrozenBackboneDetector checks that need no download (weights=None).

Run:  python test_backbone.py
"""
import sys

import torch

import detector as D

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:52s} {detail}")
    if not ok:
        FAILED.append(name)


m = D.FrozenBackboneDetector(weights=None)
x = torch.randn(2, 3, 192, 192)
ref = [t.shape for t in D.Detector(32)(x)]
out = [t.shape for t in m(x)]
check("heads have the scratch detector's shapes (stride 4)", out == ref, str(out))
check("backbone frozen: no grad", all(not p.requires_grad for p in m.backbone_parameters()))
m.train()
check("train() keeps backbone BatchNorm in eval", not m.b4.training and not m.b16.training and m.s2.training)
before = [p.clone() for p in m.backbone_parameters()]
opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
hm, wh, off = m(x)
(hm.sum() + wh.sum() + off.sum()).backward()
opt.step()
check("one optimiser step leaves the backbone unchanged",
      all(torch.equal(a, b) for a, b in zip(before, m.backbone_parameters())))
ck = {"backbone": "mnv3", "width": 32, "model": m.state_dict()}
b = D.build(ck)
b.load_state_dict(ck["model"])
check("build() restores a mnv3 checkpoint", isinstance(b, D.FrozenBackboneDetector))
check("build() defaults to the scratch detector", isinstance(D.build({"width": 32}), D.Detector))

print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
