"""Target encoding and decoding are inverses. Verified before training, because
a bug here does not crash -- it produces a model that trains to a low loss and
predicts boxes in the wrong place, and the only symptom is a bad mAP that looks
like a modelling problem.
"""
import numpy as np
import torch

import detector as D


def _dense(ind, val, grid, c=2):
    """scatter the compact (ind, value) targets back into a dense map, which is
    what decode() reads from a trained model."""
    m = torch.zeros(1, c, grid * grid)
    for i, v in zip(ind, val):
        m[0, :, i] = torch.tensor(v)
    return m.view(1, c, grid, grid)


def test_roundtrip():
    """encode -> perfect prediction -> decode must return the input boxes."""
    grid, stride = 48, D.STRIDE
    boxes = np.array([[10.0, 12.0, 38.0, 41.0],      # generic
                      [100.5, 100.5, 128.5, 130.5],  # centre lands mid-cell
                      [4.0, 4.0, 20.0, 20.0]])       # near the frame edge
    cls = np.array([0, 2, 1])
    hm, ind, wh, off, mask = D.encode(boxes, cls, grid)
    assert mask[:3].sum() == 3 and mask[3:].sum() == 0

    h = torch.from_numpy(hm)[None]
    b, c, s = D.decode(h, _dense(ind[:3], wh[:3], grid), _dense(ind[:3], off[:3], grid), k=3)
    got = b[0].numpy()
    order = np.argsort(got[:, 0])
    ref = boxes[np.argsort(boxes[:, 0])]
    assert np.allclose(got[order], ref, atol=1e-3), (got[order], ref)
    assert sorted(c[0].tolist()) == sorted(cls.tolist())
    assert np.allclose(s[0].numpy(), 1.0)


def test_peak_is_one_and_unique():
    grid = 48
    hm, ind, _, _, _ = D.encode(np.array([[40.0, 40.0, 68.0, 68.0]]), np.array([1]), grid)
    assert hm[1].max() == 1.0
    assert (hm[1] == 1.0).sum() == 1
    assert hm[0].max() == 0.0 and hm[2].max() == 0.0     # other classes untouched
    cy, cx = np.unravel_index(hm[1].argmax(), hm[1].shape)
    assert cy * grid + cx == ind[0]


def test_gaussian_is_soft_not_hot():
    """The cell next to the peak must carry most of the peak's value; that is the
    entire reason the focal loss has a (1-gt)^4 term."""
    grid = 48
    hm, _, _, _, _ = D.encode(np.array([[40.0, 40.0, 68.0, 68.0]]), np.array([0]), grid)
    cy, cx = np.unravel_index(hm[0].argmax(), hm[0].shape)
    assert 0.1 < hm[0, cy, cx + 1] < 1.0     # r=1 for a 7-cell box, sigma=0.5
    assert hm[0, cy, cx + 5] == 0.0


def test_radius_grows_with_box():
    r = [D.gaussian_radius(h, h) for h in (2, 5, 10, 20)]
    assert all(a < b for a, b in zip(r, r[1:]))
    assert D.gaussian_radius(7, 7) < 7        # never larger than the object


def test_maxpool_nms():
    """Two adjacent peaks are one object seen twice; only the stronger survives."""
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 4, 4] = 0.9
    hm[0, 0, 4, 5] = 0.7                        # inside the 3x3 window
    hm[0, 0, 1, 1] = 0.6                        # outside it
    keep = (hm == torch.nn.functional.max_pool2d(hm, 3, 1, 1)).float()
    assert (hm * keep > 0).sum() == 2
    assert (hm * keep)[0, 0, 4, 5] == 0


def test_focal_loss_zero_at_truth():
    hm, _, _, _, _ = D.encode(np.array([[40.0, 40.0, 68.0, 68.0]]), np.array([0]), 48)
    gt = torch.from_numpy(hm)[None]
    assert D.focal_loss(gt.clamp(1e-4, 1 - 1e-4), gt).item() < 0.02
    assert D.focal_loss(torch.full_like(gt, 0.5), gt).item() > 1.0


def test_offset_is_needed():
    """A box whose centre is not on a cell boundary is off by up to stride/2
    without the offset head. Quantifies why the head exists."""
    grid = 48
    b = np.array([[12.0, 12.0, 40.0, 40.0]])   # centre 26px = 6.5 cells, worst case
    hm, ind, wh, off, _ = D.encode(b, np.array([0]), grid)
    zero = np.zeros_like(off)
    h = torch.from_numpy(hm)[None]
    with_off, _, _ = D.decode(h, _dense(ind[:1], wh[:1], grid), _dense(ind[:1], off[:1], grid), k=1)
    no_off, _, _ = D.decode(h, _dense(ind[:1], wh[:1], grid), _dense(ind[:1], zero[:1], grid), k=1)
    e1 = np.abs(with_off[0, 0].numpy() - b[0]).max()
    e2 = np.abs(no_off[0, 0].numpy() - b[0]).max()
    assert e1 < 1e-3 and e2 > 1.0
    print(f"    offset head worth {e2:.2f} px of centre error on this box")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok  {f.__name__}")
    print(f"{len(fns)} passed")
