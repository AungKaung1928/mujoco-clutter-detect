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



def test_roundtrip_off_diagonal():
    """Every box above has its centre on the grid diagonal (cx == cy), where a
    transposed index or swapped x/y decodes to the same place. Off the diagonal
    the index layout is visible."""
    grid = 48
    boxes = np.array([[100.0, 10.0, 130.0, 34.0],     # centre (115, 22) px -> cell (28, 5)
                      [8.0, 150.0, 30.0, 178.0]])     # centre (19, 164) px -> cell (4, 41)
    cls = np.array([1, 0])
    hm, ind, wh, off, mask = D.encode(boxes, cls, grid)
    assert ind[:2].tolist() == [5 * grid + 28, 41 * grid + 4]
    b, c, _ = D.decode(torch.from_numpy(hm)[None], _dense(ind[:2], wh[:2], grid),
                       _dense(ind[:2], off[:2], grid), k=2)
    got = b[0].numpy()[np.argsort(b[0].numpy()[:, 0])]
    assert np.allclose(got, boxes[np.argsort(boxes[:, 0])], atol=1e-3), got


def test_decode_applies_maxpool_nms():
    """test_maxpool_nms checks the expression; this checks that decode() uses it:
    the weaker of two adjacent peaks must come back with score 0."""
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 4, 4], hm[0, 0, 4, 5], hm[0, 0, 1, 1] = 0.9, 0.7, 0.6
    z = torch.zeros(1, 2, 8, 8)
    _, _, scores = D.decode(hm, z, z, k=3)
    assert np.allclose(scores[0].numpy(), [0.9, 0.6, 0.0])


def test_reg_l1_is_the_mean_over_real_objects():
    pred = torch.zeros(1, 2, 4, 4)
    pred[0, :, 1, 2] = torch.tensor([1.0, 2.0])         # flat index 6
    pred[0, :, 3, 0] = torch.tensor([5.0, 5.0])         # flat index 12, padding slot
    ind = torch.tensor([[6, 12]])
    target = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    # one real object, two channels: (|1| + |2|) / 2; the padding slot counts for nothing
    assert abs(D.reg_l1(pred, ind, target, mask).item() - 1.5) < 1e-6


def test_focal_loss_hand_computed():
    """One peak (gt 1, pred 0.8) and one near-peak cell (gt 0.5, pred 0.3)."""
    gt = torch.tensor([[[[1.0, 0.5]]]])
    pred = torch.tensor([[[[0.8, 0.3]]]])
    want = -(np.log(0.8) * 0.2 ** 2 + np.log(0.7) * 0.3 ** 2 * 0.5 ** 4)
    assert abs(D.focal_loss(pred, gt).item() - want) < 1e-6


def test_same_class_neighbours_do_not_sum_above_one():
    hm, _, _, _, _ = D.encode(np.array([[40.0, 40.0, 68.0, 68.0], [44.0, 40.0, 72.0, 68.0]]),   # cells 13, 14
                              np.array([0, 0]), 48)
    assert hm[0].max() == 1.0 and (hm[0] == 1.0).sum() == 2


def test_degenerate_and_outside_boxes_are_skipped():
    grid = 48
    boxes = np.array([[20.0, 20.0, 20.0, 40.0],          # zero width
                      [188.0, 30.0, 196.0, 40.0]])       # centre x 192 px = cell 48, outside
    _, _, _, _, mask = D.encode(boxes, np.array([0, 1]), grid)
    assert mask.sum() == 0


class _Shift:
    """rng stand-in for train_det.geometric: no flip, shift (dx, dy)."""
    def __init__(self, dx, dy):
        self.d = [dx, dy]

    def random(self):
        return 0.9

    def integers(self, lo, hi):
        return self.d.pop(0)


def test_geometric_clips_and_drops():
    """A box pushed mostly out of frame is dropped; one that stays mostly inside
    is kept clipped to the frame. (np.clip with out=b[:, [1, 3]] wrote into a
    fancy-index copy, so neither happened.)"""
    import train_det as T
    x = np.zeros((192, 192, 3), np.float32)
    b = np.array([[0, 2, 50, 22, 70],          # 20 px wide, shifted 14 left: 6 px stay
                  [1, 10, 100, 40, 130],       # 30 px wide: 26 px stay -> kept, clipped
                  [2, 100, 100, 130, 130]], np.float32)
    _, out = T.geometric(x, b, _Shift(-14, 0), 192)
    assert out[:, 0].tolist() == [1, 2], out
    assert out[0, 1] == 0 and out[0, 3] == 26, out
    assert (out[:, 1:5] >= 0).all() and (out[:, 1:5] <= 192).all()


class _FlipShift(_Shift):
    """rng stand-in: flip (random() < 0.5) or not, then shift (dx, dy)."""
    def __init__(self, flip, dx, dy):
        super().__init__(dx, dy)
        self.flip = flip

    def random(self):
        return 0.1 if self.flip else 0.9


def test_geometric_moves_image_and_boxes_together():
    """The augmentation is only correct if the pixels of an object end up inside
    its transformed box. A bright block drawn exactly on the box must match the
    output box after every flip / shift combination."""
    import train_det as T
    size = 64
    box = np.array([[0, 10, 20, 26, 30]], np.float32)       # x 10..26, y 20..30
    x = np.zeros((size, size, 3), np.float32)
    x[20:30, 10:26] = 1.0
    for flip in (False, True):
        for dx, dy in ((5, -3), (-4, 6), (0, 0)):
            img, out = T.geometric(x.copy(), box.copy(), _FlipShift(flip, dx, dy), size)
            ys, xs = np.nonzero(img[..., 0] > 0.5)
            got = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            assert len(out) == 1 and np.allclose(out[0, 1:5], got), (flip, dx, dy, out, got)


def test_geometric_fills_the_wrapped_strip_with_the_edge():
    """np.roll wraps; the strip it wraps in must repeat the edge pixel, not show
    the opposite side of the image."""
    import train_det as T
    size = 16
    x = np.tile(np.arange(size, dtype=np.float32)[None, :, None], (size, 1, 3))  # value = column
    img, _ = T.geometric(x, np.zeros((0, 5), np.float32), _Shift(3, 0), size)
    assert (img[:, :3, 0] == 0).all() and (img[:, 3, 0] == 0).all()           # column 0 repeated
    img, _ = T.geometric(x, np.zeros((0, 5), np.float32), _Shift(-3, 0), size)
    assert (img[:, -3:, 0] == size - 1).all()
    y = np.transpose(x, (1, 0, 2)).copy()                                       # value = row
    img, _ = T.geometric(y, np.zeros((0, 5), np.float32), _Shift(0, -3), size)
    assert (img[-3:, :, 0] == size - 1).all()
    img, _ = T.geometric(y, np.zeros((0, 5), np.float32), _Shift(0, 3), size)
    assert (img[:3, :, 0] == 0).all()


def test_geometric_drops_slivers_under_5px():
    """Over half the area survives (3 of 5 px) but 3 px is a sliver, not a cube."""
    import train_det as T
    b = np.array([[0, 0, 40, 5, 50]], np.float32)          # 5 px wide at the left edge
    _, out = T.geometric(np.zeros((64, 64, 3), np.float32), b, _Shift(-2, 0), 64)
    assert len(out) == 0, out


def test_geometric_flips_about_half_the_time():
    """With a real generator, about half the calls flip (the box's x0 moves to the
    other side of the frame); a flip probability of 0 or 1 would pass every test above."""
    import train_det as T
    rng = np.random.default_rng(0)
    b = np.array([[0, 20, 30, 30, 40]], np.float32)     # x0 lands in 6..34 unflipped, 148..176 flipped
    flips = sum(T.geometric(np.zeros((192, 192, 3), np.float32), b.copy(), rng, 192)[1][0, 1] > 96
                for _ in range(400))
    assert 160 < flips < 240, flips

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok  {f.__name__}")
    print(f"{len(fns)} passed")
