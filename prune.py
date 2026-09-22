"""Step 6b: structured channel pruning of the detector, with the channels actually
removed.

Structured, not unstructured, because the target is CPU latency. Zeroing 50%
of individual weights leaves every convolution the same shape and the same
cost; a dense kernel does not skip zeros. Removing whole output channels makes
the next layer's input narrower too, and the MACs go down in the shape of the
network rather than on paper. The saliency is the BatchNorm scale gamma of each
conv+BN block (Liu et al. 2017, "network slimming"): a channel whose gamma is
near zero is one the network has already learned to ignore.

Masking is not pruning. A mask proves an accuracy claim, not a latency claim,
so this file does the surgery explicitly: a physically narrower `SlimDetector`
is built, the surviving weights are copied into it, and every consumer of a
pruned tensor has its input channels sliced to match. The FPN makes that
non-trivial and is the reason the wiring is spelt out:

    c1       -> c2[0]                          (input)
    c2[0]    -> c2[1]
    c2[1] x2 -> c3[0], l2                      (two consumers)
    c3[0]    -> c3[1]
    c3[1] x3 -> c4[0], l3
    c4[0]    -> c4[1]
    c4[1] x4 -> p4
    s2       -> hm, wh, off                    (three heads)

`s3` is not pruned: its output is summed with `l2(x2)`, so removing one of its
channels would require removing the same channel from `l2`, whose output is
not BN-scaled and has no saliency of its own. The three lateral 1x1 convs and
the three heads keep their output widths. With ratio 0 the pruned network must
reproduce the original to floating-point precision; `test_edge.py` checks it,
because a rewiring error here does not crash, it trains to a low loss with the
wrong channels talking to each other.

    nice -n 10 python prune.py --ratios 0.25 0.5 --finetune-epochs 3
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ap as AP
import common as C
import detector as D
from export_onnx import export, load, sess, time_onnx, time_torch
from train_det import DetData, MEAN, STD, predict

ROOT = os.path.dirname(os.path.abspath(__file__))

# blocks whose OUTPUT channels can be removed, in forward order
PRUNABLE = ["c1", "c2.0", "c2.1", "c3.0", "c3.1", "c4.0", "c4.1", "s2"]
MIN_KEEP = 4


class SlimDetector(nn.Module):
    """The Detector's topology with explicit channel counts per block.

    `ch` maps block name -> output channels. `Detector(w, head)` is the special
    case ch = {c1: w/2, c2.0: w, c2.1: w, c3.0: 2w, c3.1: 2w, c4.0: 4w, c4.1: 4w,
    s3: 2w, s2: head}; `full_channels` builds that dict.
    """

    def __init__(self, ch):
        super().__init__()
        self.ch = dict(ch)
        cb = D.conv_bn
        self.c1 = cb(3, ch["c1"], s=2)
        self.c2 = nn.Sequential(cb(ch["c1"], ch["c2.0"], s=2), cb(ch["c2.0"], ch["c2.1"]))
        self.c3 = nn.Sequential(cb(ch["c2.1"], ch["c3.0"], s=2), cb(ch["c3.0"], ch["c3.1"]))
        self.c4 = nn.Sequential(cb(ch["c3.1"], ch["c4.0"], s=2), cb(ch["c4.0"], ch["c4.1"]))
        lat = ch["s3"]                       # the FPN's shared lateral width
        self.l3 = nn.Conv2d(ch["c3.1"], lat, 1)
        self.l2 = nn.Conv2d(ch["c2.1"], lat, 1)
        self.p4 = nn.Conv2d(ch["c4.1"], lat, 1)
        self.s3 = cb(lat, lat)
        self.s2 = cb(lat, ch["s2"])
        self.hm = nn.Conv2d(ch["s2"], D.N_CLS, 1)
        self.wh = nn.Conv2d(ch["s2"], 2, 1)
        self.off = nn.Conv2d(ch["s2"], 2, 1)
        self.hm.bias.data.fill_(-4.6)

    forward = D.Detector.forward
    n_params = D.Detector.n_params


def full_channels(w=32, head=64):
    return {"c1": w // 2, "c2.0": w, "c2.1": w, "c3.0": w * 2, "c3.1": w * 2,
            "c4.0": w * 4, "c4.1": w * 4, "s3": w * 2, "s2": head}


def _block(model, name):
    """The conv_bn Sequential for a PRUNABLE name."""
    if "." in name:
        a, i = name.split(".")
        return getattr(model, a)[int(i)]
    return getattr(model, name)


def gammas(model):
    return {n: _block(model, n)[1].weight.detach().abs().clone() for n in PRUNABLE}


def select_channels(model, ratio, mode="global"):
    """name -> sorted LongTensor of channels to keep.

    global: one threshold across all prunable BN gammas, so layers the network
            uses less lose more (the network-slimming recipe).
    layer:  remove the same fraction from every block.
    Every block keeps at least MIN_KEEP channels.
    """
    g = gammas(model)
    keep = {}
    if ratio <= 0:
        return {n: torch.arange(len(v)) for n, v in g.items()}
    if mode == "global":
        allg = torch.cat(list(g.values()))
        thr = torch.quantile(allg, ratio)
        for n, v in g.items():
            idx = torch.nonzero(v > thr).flatten()
            if len(idx) < MIN_KEEP:
                idx = torch.topk(v, MIN_KEEP).indices
            keep[n] = torch.sort(idx).values
    else:
        for n, v in g.items():
            k = max(MIN_KEEP, int(round(len(v) * (1 - ratio))))
            keep[n] = torch.sort(torch.topk(v, k).indices).values
    return keep


def _copy_conv_bn(src, dst, keep_out, keep_in):
    conv_s, bn_s = src[0], src[1]
    conv_d, bn_d = dst[0], dst[1]
    conv_d.weight.data.copy_(conv_s.weight.data[keep_out][:, keep_in])
    for attr in ("weight", "bias", "running_mean", "running_var"):
        getattr(bn_d, attr).data.copy_(getattr(bn_s, attr).data[keep_out])
    bn_d.num_batches_tracked.copy_(bn_s.num_batches_tracked)
    bn_d.eps, bn_d.momentum = bn_s.eps, bn_s.momentum


def _copy_conv(src, dst, keep_out, keep_in):
    dst.weight.data.copy_(src.weight.data[keep_out][:, keep_in])
    if src.bias is not None:
        dst.bias.data.copy_(src.bias.data[keep_out])


def prune_detector(model, ratio, mode="global"):
    """Detector (or SlimDetector) -> a narrower SlimDetector with copied weights."""
    model.eval()
    keep = select_channels(model, ratio, mode)
    old = getattr(model, "ch", None) or full_channels(
        model.c2[0][0].out_channels, model.s2[0].out_channels)
    ch = {n: len(keep[n]) for n in PRUNABLE}
    ch["s3"] = old["s3"]
    slim = SlimDetector(ch).eval()
    all_in = lambda n: torch.arange(n)      # noqa: E731

    _copy_conv_bn(model.c1, slim.c1, keep["c1"], all_in(3))
    _copy_conv_bn(model.c2[0], slim.c2[0], keep["c2.0"], keep["c1"])
    _copy_conv_bn(model.c2[1], slim.c2[1], keep["c2.1"], keep["c2.0"])
    _copy_conv_bn(model.c3[0], slim.c3[0], keep["c3.0"], keep["c2.1"])
    _copy_conv_bn(model.c3[1], slim.c3[1], keep["c3.1"], keep["c3.0"])
    _copy_conv_bn(model.c4[0], slim.c4[0], keep["c4.0"], keep["c3.1"])
    _copy_conv_bn(model.c4[1], slim.c4[1], keep["c4.1"], keep["c4.0"])
    lat = all_in(old["s3"])
    _copy_conv(model.l2, slim.l2, lat, keep["c2.1"])
    _copy_conv(model.l3, slim.l3, lat, keep["c3.1"])
    _copy_conv(model.p4, slim.p4, lat, keep["c4.1"])
    _copy_conv_bn(model.s3, slim.s3, lat, lat)
    _copy_conv_bn(model.s2, slim.s2, keep["s2"], lat)
    for h in ("hm", "wh", "off"):
        _copy_conv(getattr(model, h), getattr(slim, h),
                   all_in(getattr(model, h).out_channels), keep["s2"])
    return slim, keep


# ------------------------------------------------------------------- MAC count

def count_macs(model, size=192, in_ch=3):
    """Multiply-accumulates of one forward at batch 1, convolutions only.

    BatchNorm folds into the conv at inference, ReLU and the two nearest
    upsamples are memory ops, and the sigmoid is 3x48x48. Counting only convs
    understates the total by well under 1%, and is what every pruning paper
    reports, so the numbers are comparable.
    """
    macs = [0]

    def hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1]
        macs[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * k

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(torch.zeros(1, in_ch, size, size))
    for h in hs:
        h.remove()
    return macs[0]


# -------------------------------------------------------------------- finetune

def finetune(model, regime, epochs, bs=32, lr=5e-4, seed=0, tune_n=1500, fit_n=0):
    """The training loop of train_det.py, on the pruned network, short.

    Same dataset class, same losses, same fit/tune split. No augmentation:
    step 4 measured it at +0.0007 mAP on this regime.
    """
    if epochs <= 0:
        return {"epochs": 0, "history": [], "train_s": 0.0}
    torch.manual_seed(seed)
    tr_i, tr_b, meta = C.load_split(ROOT, regime, "train")
    size = meta["img_size"]
    n_fit = len(tr_i) - tune_n
    if fit_n:
        n_fit = min(n_fit, fit_n)
    tr_i = np.asarray(tr_i[:n_fit])
    fit_b = tr_b[tr_b[:, 0] < n_fit]
    dl = DataLoader(DetData(tr_i, fit_b, size, aug="none", seed=seed),
                    batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    hist, t0 = [], time.perf_counter()
    for ep in range(epochs):
        dl.dataset.epoch = ep
        model.train()
        agg, te = 0.0, time.perf_counter()
        for x, hm, ind, wh, off, mask in dl:
            p_hm, p_wh, p_off = model(x)
            loss = D.focal_loss(p_hm, hm) + 0.1 * D.reg_l1(p_wh, ind, wh, mask) \
                + D.reg_l1(p_off, ind, off, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            agg += loss.item()
        dt = time.perf_counter() - te
        hist.append({"epoch": ep + 1, "loss": agg / len(dl), "img_s": n_fit / dt})
        print(f"  finetune ep {ep + 1}/{epochs}  loss {agg / len(dl):.4f}  {n_fit / dt:6.0f} img/s",
              flush=True)
    model.eval()
    return {"epochs": epochs, "history": hist, "train_s": time.perf_counter() - t0, "lr": lr}


# ------------------------------------------------------------------------ main

def evaluate_model(model, imgs, gts, vf, size, n_eval):
    n = len(imgs) if n_eval <= 0 else min(n_eval, len(imgs))
    dets = predict(model, imgs[:n], size)
    g_sel = gts[:, 0] < n
    g = gts[g_sel]
    r = AP.evaluate(dets, g)
    rv = AP.recall_by_visibility(dets, g, vf[g_sel], [0.0, 0.5, 0.9, 1.01])
    return {"n_images": int(n), "n_dets": int(len(dets)), "mAP": float(r["mAP"]),
            "AP50": float(r["AP50"]), "AP75": float(r["AP75"]),
            "ap_per_class": [float(v) for v in r["ap_per_class"]],
            "vis_recall": {f"{lo:.1f}-{hi:.1f}": v["recall"] for (lo, hi), v in rv.items()}}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", default="runs/det_hard_none.pt")
    a.add_argument("--regime", choices=C.REGIMES, default="hard")
    a.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5])
    a.add_argument("--mode", choices=["global", "layer"], default="global")
    a.add_argument("--finetune-epochs", type=int, default=3)
    a.add_argument("--fit-n", type=int, default=0, help="cap fine-tune images, 0 = all")
    a.add_argument("--n-eval", type=int, default=0, help="val images to score, 0 = all")
    a.add_argument("--n-time", type=int, default=200)
    a.add_argument("--int8", action="store_true", help="also static-quantize the pruned graph")
    a.add_argument("--calib", type=int, default=200)
    a.add_argument("--tag", default="")
    args = a.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    base, size = load(args.ckpt)
    va_i, va_b, _ = C.load_split(ROOT, args.regime, "val")
    gts = va_b[:, :6].astype(np.float64)
    vf = va_b[:, 6].astype(np.float64)
    xs = np.ascontiguousarray(
        ((np.asarray(va_i[:args.n_time]).astype(np.float32) / 255.0 - MEAN) / STD)
        .transpose(0, 3, 1, 2))
    xt = torch.from_numpy(xs)
    load1 = float(open("/proc/loadavg").read().split()[0])

    base_macs = count_macs(base, size)
    base_sc = evaluate_model(base, va_i, gts, vf, size, args.n_eval)
    print(f"base   params {base.n_params():>8,}  MACs {base_macs/1e6:7.1f} M  "
          f"mAP {base_sc['mAP']:.4f} on {base_sc['n_images']} images")

    for ratio in args.ratios:
        slim, keep = prune_detector(base, ratio, args.mode)
        ch = {n: len(k) for n, k in keep.items()}
        macs = count_macs(slim, size)
        pre = evaluate_model(slim, va_i, gts, vf, size, args.n_eval)
        print(f"\nratio {ratio:.2f} ({args.mode})  params {slim.n_params():>8,} "
              f"({100*slim.n_params()/base.n_params():.0f}%)  MACs {macs/1e6:7.1f} M "
              f"({100*macs/base_macs:.0f}%)  mAP before finetune {pre['mAP']:.4f}")
        print("  channels kept: " + "  ".join(f"{n} {ch[n]}" for n in PRUNABLE))
        ft = finetune(slim, args.regime, args.finetune_epochs, fit_n=args.fit_n)
        post = evaluate_model(slim, va_i, gts, vf, size, args.n_eval)

        stem = f"runs/detector_pruned_{int(ratio*100)}{tag}"
        torch.save({"model": slim.state_dict(), "ch": slim.ch, "img_size": size,
                    "ratio": ratio, "mode": args.mode}, stem + ".pt")
        nbytes = export(slim, size, stem + ".onnx")
        lat = {"torch_1t": time_torch(slim, xt, args.n_time, 1),
               "onnxrt_1t": time_onnx(stem + ".onnx", xs, args.n_time, 1),
               "onnxrt_4t": time_onnx(stem + ".onnx", xs, args.n_time, 4)}
        res = {"ratio": ratio, "mode": args.mode, "channels": ch, "params": slim.n_params(),
               "params_base": base.n_params(), "macs": macs, "macs_base": base_macs,
               "score_before_finetune": pre, "finetune": ft, "score": post,
               "score_base": base_sc, "mAP_drop": base_sc["mAP"] - post["mAP"],
               "bytes": nbytes, "latency_ms": lat, "onnx": stem + ".onnx", "ckpt": stem + ".pt",
               "measured": {"n_eval": post["n_images"], "n_time": args.n_time,
                            "load1_at_start": load1, "regime": args.regime}}
        if args.int8:
            from quantize import pre_process, quantize, score as q_score
            tr_i, _, _ = C.load_split(ROOT, args.regime, "train")
            q_pre = pre_process(stem + ".onnx", stem + "_pre.onnx")
            q_out = stem + "_int8.onnx"
            q_s = quantize(q_pre, q_out, "minmax", tr_i, args.calib)
            res["int8"] = {"path": q_out, "bytes": os.path.getsize(q_out), "quantize_s": q_s,
                           "score": q_score(q_out, va_i, gts, vf, size, args.n_eval),
                           "latency_ms": {"onnxrt_1t": time_onnx(q_out, xs, args.n_time, 1),
                                          "onnxrt_4t": time_onnx(q_out, xs, args.n_time, 4)}}
            print(f"  + INT8: mAP {res['int8']['score']['mAP']:.4f}  "
                  f"1t {res['int8']['latency_ms']['onnxrt_1t']:.3f} ms  "
                  f"{res['int8']['bytes']/1e6:.2f} MB")
        json.dump(res, open(f"runs/prune_{int(ratio*100)}{tag}.json", "w"), indent=2)
        print(f"  after finetune  mAP {post['mAP']:.4f} ({post['mAP']-base_sc['mAP']:+.4f})  "
              f"ORT 1t {lat['onnxrt_1t']:.3f} ms  4t {lat['onnxrt_4t']:.3f} ms  "
              f"torch 1t {lat['torch_1t']:.3f} ms  {nbytes/1e6:.2f} MB")
        print(f"  wrote runs/prune_{int(ratio*100)}{tag}.json")


if __name__ == "__main__":
    main()
