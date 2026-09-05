"""Step 4: does augmentation still buy anything when the simulator already
randomises the scene?

This is not a rhetorical question. Photometric augmentation exists because real
datasets are captured under one set of lights and deployed under another. A
simulator with domain randomisation already varies colour, lighting position and
lighting intensity at render time -- the `hard` regime does exactly that. If
augmentation still helps there, it is adding variation the randomiser does not
cover. If it does not, then every hour spent on an augmentation pipeline for a
randomised simulator is an hour spent on nothing, and that is worth knowing
before block 5 builds a bigger one.

Two by four, one budget:

              trained on easy            trained on hard
  none        appearance fixed           randomiser only
  photo       augmentation only          randomiser + augmentation
  geom        flip + shift               flip + shift
  both

Each model is scored on BOTH regimes' val splits. The easy->hard column is the
sim-to-real question in miniature: `easy` is a simulator that was never
randomised, `hard` is the world it has to survive. Augmentation and domain
randomisation are two ways to buy the same invariance, and this measures which
one is doing the work.

Budget is reduced and IDENTICAL across all eight cells. An ablation compares
conditions; it does not need the best model, it needs eight comparable ones.
"""
import argparse
import json
import os
import time

import common as C
import train_det as T

AUGS = ["none", "photo", "geom", "both"]


def main(regimes, augs, epochs, fit_n, out):
    res = {}
    t0 = time.perf_counter()
    for regime in regimes:
        for aug in augs:
            key = f"{regime}/{aug}"
            print(f"\n{'='*70}\n{key}   ({len(res)+1}/{len(regimes)*len(augs)}, "
                  f"{time.perf_counter()-t0:.0f}s elapsed)\n{'='*70}", flush=True)
            res[key] = T.train(regime, aug, epochs, 32, 2.5e-3, 32, 0, 1500,
                               "", "", fit_n=fit_n, cross=True)
            json.dump(res, open(out, "w"), indent=2)      # written after every cell
    table(res, regimes, augs)
    print(f"\ntotal {(time.perf_counter()-t0)/60:.1f} min   wrote {out}")
    return res


def table(res, regimes, augs):
    print(f"\n{'='*70}\nstep 4 -- mAP@[.5:.95] on each val split\n{'='*70}")
    print(f"{'trained on':>12} {'aug':>6} | {'easy val':>9} {'hard val':>9} | "
          f"{'drop':>7} | {'train s':>8}")
    for regime in regimes:
        for aug in augs:
            r = res.get(f"{regime}/{aug}")
            if r is None:
                continue
            own, cross = r["mAP"], r.get("cross", {}).get("mAP", float("nan"))
            e, h = (own, cross) if regime == "easy" else (cross, own)
            print(f"{regime:>12} {aug:>6} | {e:9.4f} {h:9.4f} | "
                  f"{e-h:+7.4f} | {r['train_s']:8.0f}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--regimes", nargs="+", default=list(C.REGIMES))
    a.add_argument("--augs", nargs="+", default=AUGS)
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--fit-n", type=int, default=6000)
    a.add_argument("--out", default="runs/ablation.json")
    a = a.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    main(a.regimes, a.augs, a.epochs, a.fit_n, a.out)
