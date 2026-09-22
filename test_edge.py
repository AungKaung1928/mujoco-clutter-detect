"""Step 6 checks that need no data and no trained weights.

Three failure modes that do not crash: an INT8 graph that quantizes nothing (all
Q/DQ pairs skipped, so the 'speedup' is the fp32 graph timed twice), a channel
rewiring that connects the wrong channels (trains to a low loss, detects
nothing), and a MAC counter that reports the wrong unit. Each is checked against
a value derived by hand or an identity that must hold exactly.

Run:  python test_edge.py
"""
import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch

import detector as D
import prune as P
from export_onnx import export
from quantize import ImageReader, pre_process, quantize


def _tiny_export(w, tmp, size=64):
    torch.manual_seed(0)
    m = D.Detector(w=w, head=16).eval()
    p = os.path.join(tmp, "tiny.onnx")
    export(m, size, p)
    return m, p


def test_static_int8_roundtrip():
    """fp32 -> pre-process -> static INT8 with 4 calibration images -> runs, same
    output shapes, decode works, and the graph really contains quantized convs."""
    with tempfile.TemporaryDirectory() as tmp:
        m, p = _tiny_export(8, tmp)
        pre = pre_process(p, os.path.join(tmp, "pre.onnx"))
        rng = np.random.default_rng(0)
        calib = rng.integers(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        q = os.path.join(tmp, "int8.onnx")
        quantize(pre, q, "minmax", calib, 4)

        ops = [n.op_type for n in onnx.load(q).graph.node]
        assert ops.count("QuantizeLinear") >= 3 and ops.count("DequantizeLinear") >= 3, ops
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        s = ort.InferenceSession(q, o, providers=["CPUExecutionProvider"])
        x = ImageReader(calib, 1).get_next()["image"]
        hm, wh, off = s.run(None, {"image": x})
        with torch.no_grad():
            ref = [t.numpy() for t in m(torch.from_numpy(x))]
        assert hm.shape == ref[0].shape and wh.shape == ref[1].shape and off.shape == ref[2].shape
        assert 0.0 <= hm.min() and hm.max() <= 1.0
        b, c, sc = D.decode(torch.from_numpy(hm), torch.from_numpy(wh), torch.from_numpy(off), k=5)
        assert b.shape == (1, 5, 4) and c.shape == (1, 5) and sc.shape == (1, 5)


def test_dynamic_int8_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        _, p = _tiny_export(8, tmp)
        pre = pre_process(p, os.path.join(tmp, "pre.onnx"))
        q = os.path.join(tmp, "dyn.onnx")
        quantize(pre, q, "dynamic", None, 0)
        s = ort.InferenceSession(q, providers=["CPUExecutionProvider"])
        out = s.run(None, {"image": np.zeros((1, 3, 64, 64), np.float32)})
        assert out[0].shape == (1, D.N_CLS, 16, 16)


def test_prune_ratio_zero_is_identity():
    """The rewiring must reproduce the original network exactly when nothing is
    removed. This is the check that catches a wrong channel index."""
    torch.manual_seed(1)
    m = D.Detector(w=16, head=32).eval()
    # give BN non-trivial statistics so a permutation would show
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_()
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.normal_()
            mod.bias.data.normal_()
    slim, keep = P.prune_detector(m, 0.0)
    assert all(len(k) == n for k, n in zip(keep.values(), [8, 16, 16, 32, 32, 64, 64, 32]))
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a, b = m(x), slim(x)
    for ta, tb in zip(a, b):
        assert torch.allclose(ta, tb, atol=1e-5), (ta - tb).abs().max()
    assert slim.n_params() == m.n_params()


def test_prune_removes_channels_and_keeps_shapes():
    torch.manual_seed(2)
    m = D.Detector(w=16, head=32).eval()
    for mode in ("global", "layer"):
        slim, keep = P.prune_detector(m, 0.5, mode)
        assert slim.n_params() < 0.6 * m.n_params(), (mode, slim.n_params(), m.n_params())
        assert P.count_macs(slim, 64) < 0.6 * P.count_macs(m, 64)
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            a, b = m(x), slim(x)
        for ta, tb in zip(a, b):
            assert ta.shape == tb.shape
        # the unpruned lateral width and every head are untouched
        assert slim.l2.out_channels == m.l2.out_channels
        assert slim.hm.out_channels == D.N_CLS
        for n in P.PRUNABLE:
            assert len(keep[n]) >= P.MIN_KEEP
            assert torch.all(keep[n][1:] > keep[n][:-1])     # sorted, unique


def test_pruned_network_survives_export():
    with tempfile.TemporaryDirectory() as tmp:
        m = D.Detector(w=16, head=32).eval()
        slim, _ = P.prune_detector(m, 0.25)
        p = os.path.join(tmp, "slim.onnx")
        export(slim, 64, p)
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        out = s.run(None, {"image": np.zeros((1, 3, 64, 64), np.float32)})
        assert out[0].shape == (1, D.N_CLS, 16, 16)


def test_mac_counter_matches_hand_count():
    """Conv2d(3 -> 8, 3x3, pad 1) on a 16x16 image: every output element costs
    3 * 9 multiply-accumulates; there are 8 * 16 * 16 of them."""
    conv = torch.nn.Conv2d(3, 8, 3, padding=1)
    assert P.count_macs(conv, 16) == 8 * 16 * 16 * 3 * 9
    strided = torch.nn.Conv2d(4, 6, 3, stride=2, padding=1)
    assert P.count_macs(strided, 16, in_ch=4) == 6 * 8 * 8 * 4 * 9
    one = torch.nn.Conv2d(5, 7, 1)
    assert P.count_macs(one, 16, in_ch=5) == 7 * 16 * 16 * 5


def test_full_channels_matches_detector():
    m = D.Detector(w=32, head=64)
    ch = P.full_channels(32, 64)
    assert m.c1[0].out_channels == ch["c1"] and m.c4[1][0].out_channels == ch["c4.1"]
    assert m.s3[0].out_channels == ch["s3"] and m.s2[0].out_channels == ch["s2"]
    slim = P.SlimDetector(ch)
    assert slim.n_params() == m.n_params()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok  {f.__name__}")
    print(f"{len(fns)} passed")
