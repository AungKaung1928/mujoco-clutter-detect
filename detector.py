"""Anchor-free detector: heatmap + size + offset, at output stride 4.

Why this architecture and not a box regressor with anchors.

Block 1 ended with a measured result: a spatial soft-argmax head beat a flatten
+ linear head on the same trunk, with 5x fewer parameters, because the head *is*
a coordinate -- position was handed to the network instead of learned. That
finding does not generalise to detection by itself, because soft-argmax computes
one expected location over the whole image and there are now up to six objects.
The generalisation that does work is to keep the spatial map and stop reducing
it: predict a per-pixel probability that an object centre is here, and read the
answer off the peaks. That is CenterNet (Zhou et al. 2019), and it is the same
idea one step further -- the output *is* an image, so nothing has to be undone.

Three heads share one feature map at stride 4 (48x48 for a 192px image):

  heatmap  (3, 48, 48)  per-class centre probability, trained with focal loss
                        against a Gaussian splat, not a single hot pixel. A hard
                        one-hot target would make 2303 of 2304 cells negative and
                        punish a prediction one pixel off exactly as hard as one
                        on the far side of the table.
  size     (2, 48, 48)  box w, h in grid units, L1 at the centre cell only
  offset   (2, 48, 48)  sub-cell centre residual, L1 at the centre cell only.
                        Stride 4 quantises the centre to 4px; on a 28px object
                        that is a 14% IoU cost for free if it is not corrected.

No anchors, no NMS by IoU: peak extraction is a 3x3 max-pool, which is NMS in
the only place a centre-based detector needs it. No two objects here share a
centre cell -- MIN_SEP in the generator guarantees it -- so that is exact, not
an approximation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C

STRIDE = 4
N_CLS = len(C.CLASSES)


# ------------------------------------------------------------------- targets

def gaussian_radius(h, w, min_overlap=0.7):
    """CornerNet's radius: how far a box of this size may be displaced and still
    overlap the truth by min_overlap. Three cases -- the displaced box can be
    larger, smaller, or shifted -- and the smallest of the three wins.

    This uses the *larger* root of each quadratic, which is what the reference
    CornerNet/CenterNet implementations do. The smaller root is the one the
    algebra actually calls for, and it was measured here before being rejected:
    at stride 4 a 28px object is 7 cells across and the smaller root gives
    r = 0.57, which floors to 0 and collapses the Gaussian back to a single hot
    pixel -- destroying the only thing the soft target is for. The larger root
    gives r = 1.9. Reproducing the reference is worth more than winning the
    argument, so the reference form is kept and the discrepancy is recorded here.
    """
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(max(b1 * b1 - 4 * c1, 0))) / 2
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + np.sqrt(max(b2 * b2 - 16 * c2, 0))) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(max(b3 * b3 - 4 * a3 * c3, 0))) / 2
    return max(1.0, min(r1, r2, r3))


def draw_gaussian(hm, cx, cy, radius):
    """Splat exp(-d^2/2s^2) with s = (2r+1)/6, combined with max() so that two
    nearby objects of the same class do not sum to a value above 1."""
    r = int(max(radius, 0))
    d = 2 * r + 1
    sigma = d / 6.0
    ax = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sigma * sigma))
    g[g < np.finfo(np.float32).eps * g.max()] = 0

    H, W = hm.shape
    l, rr = min(cx, r), min(W - cx, r + 1)
    t, b = min(cy, r), min(H - cy, r + 1)
    if rr <= -l or b <= -t:
        return
    np.maximum(hm[cy - t:cy + b, cx - l:cx + rr],
               g[r - t:r + b, r - l:r + rr],
               out=hm[cy - t:cy + b, cx - l:cx + rr])


def encode(boxes, cls, grid, stride=STRIDE, max_obj=8):
    """boxes (N,4) pixels, cls (N,) -> heatmap, and a compact per-object list.

    Size and offset are supervised at the centre cell only, so they are stored
    as (index, value) pairs rather than dense maps. A dense map with a mask
    would be 2*48*48 floats per image to carry 4.5 useful numbers."""
    hm = np.zeros((N_CLS, grid, grid), np.float32)
    ind = np.zeros(max_obj, np.int64)
    wh = np.zeros((max_obj, 2), np.float32)
    off = np.zeros((max_obj, 2), np.float32)
    mask = np.zeros(max_obj, np.float32)

    for i, (b, k) in enumerate(zip(boxes[:max_obj], cls[:max_obj])):
        w, h = (b[2] - b[0]) / stride, (b[3] - b[1]) / stride
        if w <= 0 or h <= 0:
            continue
        fx, fy = (b[0] + b[2]) / 2 / stride, (b[1] + b[3]) / 2 / stride
        cx, cy = int(fx), int(fy)
        if not (0 <= cx < grid and 0 <= cy < grid):
            continue
        draw_gaussian(hm[int(k)], cx, cy, gaussian_radius(h, w))
        ind[i] = cy * grid + cx
        wh[i] = (w, h)
        off[i] = (fx - cx, fy - cy)
        mask[i] = 1.0
    return hm, ind, wh, off, mask


# ---------------------------------------------------------------------- loss

def focal_loss(pred, gt):
    """CenterNet's penalty-reduced focal loss. `pred` is a probability.

    Positives: only the exact peak cell (gt == 1). Everything else is a negative
    whose weight is (1-gt)^4 -- so a cell just outside the peak, where gt is
    0.9, is almost unpenalised, and a cell far away is fully penalised. Without
    that term the Gaussian would be pointless: the model would be punished for
    the soft evidence the Gaussian exists to provide."""
    pos = gt.eq(1).float()
    neg = 1 - pos
    p = pred.clamp(1e-4, 1 - 1e-4)
    l_pos = torch.log(p) * (1 - p) ** 2 * pos
    l_neg = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum()
    return -(l_pos.sum() + l_neg.sum()) / n.clamp(min=1)


def _gather(feat, ind):
    """feat (B,C,H,W) -> (B,M,C) at flat spatial indices ind (B,M)."""
    b, c, h, w = feat.shape
    feat = feat.view(b, c, h * w).permute(0, 2, 1)
    return feat.gather(1, ind.unsqueeze(2).expand(-1, -1, c))


def reg_l1(pred, ind, target, mask):
    p = _gather(pred, ind)
    m = mask.unsqueeze(2).expand_as(p)
    return (torch.abs(p - target) * m).sum() / m.sum().clamp(min=1)


# --------------------------------------------------------------------- model

def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Detector(nn.Module):
    """Encoder to stride 16, FPN-style decoder back to stride 4.

    The decoder exists because the heads need both. Stride 16 has the context to
    say 'object'; stride 4 has the resolution to say 'here'. Predicting at stride
    16 on a 28px object gives a 1.75-cell footprint and the offset head has to
    absorb an 8px quantisation -- larger than the localisation error the
    classical baseline already achieves.
    """

    def __init__(self, w=32, head=64):
        super().__init__()
        self.c1 = conv_bn(3, w // 2, s=2)          # 96
        self.c2 = nn.Sequential(conv_bn(w // 2, w, s=2), conv_bn(w, w))        # 48
        self.c3 = nn.Sequential(conv_bn(w, w * 2, s=2), conv_bn(w * 2, w * 2))  # 24
        self.c4 = nn.Sequential(conv_bn(w * 2, w * 4, s=2), conv_bn(w * 4, w * 4))  # 12

        self.l3 = nn.Conv2d(w * 2, w * 2, 1)
        self.l2 = nn.Conv2d(w, w * 2, 1)
        self.p4 = nn.Conv2d(w * 4, w * 2, 1)
        self.s3 = conv_bn(w * 2, w * 2)
        self.s2 = conv_bn(w * 2, head)

        self.hm = nn.Conv2d(head, N_CLS, 1)
        self.wh = nn.Conv2d(head, 2, 1)
        self.off = nn.Conv2d(head, 2, 1)
        # Start every heatmap logit at p ~= 0.01. With 2304 cells and ~4 objects
        # the target is 99.8% zeros; from a neutral init the focal loss spends
        # its first epochs just walking the bias down, and can diverge doing it.
        self.hm.bias.data.fill_(-4.6)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        y3 = self.s3(self.l3(x3) + F.interpolate(self.p4(x4), scale_factor=2, mode="nearest"))
        y2 = self.s2(self.l2(x2) + F.interpolate(y3, scale_factor=2, mode="nearest"))
        return torch.sigmoid(self.hm(y2)), self.wh(y2), self.off(y2)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------- decoding

@torch.no_grad()
def decode(hm, wh, off, k=10, stride=STRIDE):
    """(B,C,H,W) heads -> list of (boxes, cls, score) in pixels.

    3x3 max-pool keep is the whole of NMS. A cell survives only if it is the
    maximum of its neighbourhood, which removes the ridge of near-peak cells the
    Gaussian training target deliberately creates."""
    b, c, h, w = hm.shape
    keep = (hm == F.max_pool2d(hm, 3, 1, 1)).float()
    scores, idx = (hm * keep).view(b, -1).topk(k, dim=1)
    cls = idx // (h * w)
    pix = idx % (h * w)
    ys, xs = (pix // w).float(), (pix % w).float()

    o = _gather(off, pix)
    s = _gather(wh, pix)
    cx = (xs + o[..., 0]) * stride
    cy = (ys + o[..., 1]) * stride
    bw = s[..., 0].clamp(min=0) * stride
    bh = s[..., 1].clamp(min=0) * stride
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
    return boxes, cls, scores
