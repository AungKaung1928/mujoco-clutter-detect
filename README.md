# mujoco-clutter-detect — block 2: detection, augmentation, mAP

Multi-object detection on a simulated tabletop. Three classes (`box`, `cylinder`,
`sphere`), 3–6 objects per scene, a tilted camera, and labels read straight out of
the renderer's segmentation buffer.

Block 2 of the ML track. Block 1 (`../mujoco-cube-pose-cnn`) regressed one pose
from a top-down view and closed at 0.59 mm median error with a 27k-parameter
soft-argmax head.

## Why the camera moved

Block 1's camera looked straight down. That view has an exact affine pixel↔world
map, which is why it was the right choice for pose regression — but it makes
detection degenerate. Objects resting on a flat table cannot overlap in a
top-down image, so there is no occlusion, no perspective, and no depth-dependent
scale. A detector trained there would learn none of the three things that make
detection hard.

So the scene now carries two cameras:

| camera | pose | used for |
|---|---|---|
| `top` | 0.45 m directly overhead | label verification, and the world-frame coordinates blocks 4–6 need |
| `tilt` | 39° elevation, 0.54 m out | the detection task itself |

Keeping both costs nothing and sets up block 5: train under one camera pose,
evaluate under another, and report the gap.

## Why the labels are exact

The boxes are not annotated and not projected by hand. Each scene is rendered
twice — once for RGB, once with `enable_segmentation_rendering()` — and the second
pass returns, per pixel, which geom won it. A tight box around an instance mask is
therefore already correct under perspective, already correct for the silhouette of
a rotated box, and already correct about what is hidden behind what.

This is the concrete reason robot-learning work starts in simulation. The label is
not produced by a human, it is read out of the renderer.

### The occlusion label needs a second pass

The segmentation buffer says which object won each pixel. It cannot say how many
pixels an object *would* have had alone, which is what `visible_frac` needs. So
each object is re-rendered with the others parked outside the frame, and the two
counts are divided.

That is N extra renders per scene and it dominates the cost, so it is skipped for
any object whose visible box intersects no other visible box — such an object is
provably unoccluded. In practice this cuts the second pass to about 2.6 renders
per scene instead of 4.7.

## Task decisions

- **Only the box carries a yaw label.** A sphere has no observable orientation and
  a cylinder's rotation about z is unobservable too. Labelling them would train the
  network to fit noise. `yaw` is `NaN` for both, and the loss must mask it.
- **Box yaw keeps block 1's 90° fold**, `(sin 4θ, cos 4θ)`, for the same reason: a
  square-topped box is only observable modulo 90°, and regressing raw degrees puts
  a wrap discontinuity in the middle of the label range.
- **192 px, not 128.** Objects median ≈ 28 px on a side. At 128 px they would be
  ≈ 19 px, below the COCO "small object" threshold, and the task would be measuring
  the renderer rather than the detector.
- **Objects are placed, not simulated.** No `mj_step`, no dynamics. Bodies are
  static and their pose, shape, size and colour are written directly into `mjModel`
  before `mj_forward`. One XML covers every class.

## Scene contract (frozen — later blocks depend on it)

| | |
|---|---|
| table plane | z = 0, half-size 1.5 m |
| object centres | ±0.12 m square, ≥ 0.075 m apart |
| object size | 0.018–0.035 m half-extent |
| image | 192 × 192 RGB |
| regimes | `easy` (fixed appearance) / `hard` (randomised hue, table, light) |

The table plane is 1.5 m, not 0.5 m, because the tilted camera's top frame ray only
meets the ground 1.15 m out. A smaller plane leaves a constant black band across the
top of every image — dead pixels, and a trivial cue for the network to latch onto.

## Steps

1. **Scene + dataset + label verification** — done.
2. Classical CV baseline (contours → boxes) and a hand-written COCO-style
   AP@[.5:.95] harness. Baseline before model, per block 1's finding that half a
   CNN's apparent win can be calibration.
3. Anchor-free detector (heatmap + size + offset). The heatmap head is the direct
   generalisation of block 1's soft-argmax, which won there at 5× fewer parameters.
4. Augmentation ablation: none / photometric / geometric / both, against the `easy`
   and `hard` regimes. The open question is whether photometric augmentation buys
   anything once the simulator already randomises appearance.
5. ONNX export and latency, carrying block 1's finding that the runtime, not the
   architecture, dominated edge latency.

## Reproducing

```bash
source ~/personal/ml/env.sh
python gen_dataset.py --regime hard --n 200 --smoke   # smallest useful run
python gen_dataset.py --regime hard --n 12000
python gen_dataset.py --regime easy --n 12000
python view_dataset.py --regime hard --n 12           # -> out/labels.png
```

## Results

Filled in as each step lands.

### Step 1 — dataset

