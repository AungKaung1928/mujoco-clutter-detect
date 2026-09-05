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
2. **a** — hand-written COCO-style AP@[.5:.95] harness. **b** — classical CV
   baseline (contours → boxes). Metric before baseline, baseline before model,
   per block 1's finding that half a CNN's apparent win can be calibration.
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

Four splits, 192 px, camera `tilt`, generated 2026-09-05. 2.9 GB total.

| split | images | objects | obj/img | dropped | occluded <0.9 | <0.5 | img/s |
|---|---|---|---|---|---|---|---|
| `hard/train` | 12000 | 53961 | 4.50 | 5 | 10.0% | 0.6% | 111.1 |
| `hard/val` | 2000 | 8992 | 4.50 | 0 | 10.2% | 0.6% | 115.1 |
| `easy/train` | 12000 | 53835 | 4.49 | 6 | 10.1% | 0.5% | 106.4 |
| `easy/val` | 2000 | 9049 | 4.52 | 0 | 9.8% | 0.5% | 103.6 |

Classes are balanced to within 1% in every split. Box size (sqrt area) median
28.4 px, p5 18.5, p95 43.5 — small objects by COCO's convention, which is the
regime worth measuring.

**Cost.** 108 s for 12000 images. Throughput ran 111 → 104 img/s across four
consecutive runs, a 7% decline, under the 20% band that block 1's training loop
flags as a sustained-power limit. So this is not throttled and the numbers are
real. Rendering is single-threaded (llvmpipe under WSLg); the 8 available threads
are idle throughout, which is why dataset size is never the constraint on this box.

**The occlusion-pruning optimisation paid.** Naively, `visible_frac` costs one
solo re-render per object: 4.50 per image. Skipping objects whose visible box
intersects no other visible box brought that to 2.19 — 51% of the second-pass
renders removed, 35% off the total. The pruning is exact, not an approximation:
an object that overlaps nothing in the image cannot have been occluded by
anything.

**Known limitation, stated before any detector result.** Only 10% of objects are
occluded at all and 0.6% are occluded past half. That is what a 39° elevation and
a 0.075 m minimum separation produce. So mAP measured here is *not* stressed by
occlusion, and any claim that the detector "handles occlusion" would be
unsupported by this dataset. If step 3 shows occlusion matters, the honest fix is
a lower camera or a smaller separation, and a regenerated set — not a softer
claim.


### Step 2a — the AP metric

`pycocotools` is one import away and is deliberately not used. Every detection
interview question is about what `ap.py` does; calling a library teaches none of
it. Semantics follow pycocotools closely enough to be comparable, and the places
they differ are marked in the source.

**15 hand-computed unit cases** in `test_ap.py`. Every expected value is derived
on paper, not captured from a previous run — a test that records whatever the
code happened to print proves only that the code is deterministic. Three of them
are worth stating because they are counter-intuitive and each one is a real bug
this catches:

| case | AP@0.5 | why |
|---|---|---|
| 1 gt, 1 hit, then 1 false positive | **1.000** | the FP arrives *after* recall 1.0, so no recall level was ever achievable at lower precision. It costs nothing. |
| the same FP placed *before* the hit | **0.500** | identical detections, identical recall, half the AP |
| 1 of 2 objects found, no FPs | **0.5050** | not 0.5 — 51 of the 101 recall levels are reachable, so it is 51/101 |

Plus: matching is greedy and cannot cross images; a duplicate detection is a
false positive and is only paid for when it outranks a later true positive
(0.835 vs 1.000); the IoU threshold is inclusive; a class absent from the data
gives `NaN` and is excluded from the mean rather than scored 0.

**Ignore semantics get their own test** because stratified AP depends entirely on
them. Ground truth outside a stratum is marked *ignore*, not deleted. Deleting it
turns every correct detection of an out-of-stratum object into a false positive:
the test shows the same data scoring **1.000 with ignore and 0.500 with
deletion**.

**End-to-end on `hard/val`** (2000 images, 8992 objects). A metric with known
inputs has known outputs, which is what makes these checks worth anything:

| input | mAP | expected |
|---|---|---|
| ground truth fed back as detections | **1.0000** | exactly 1 |
| correct boxes, classes shuffled | 0.1130 | ≈ 1/9: precision 1/3 × recall 1/3 |
| random boxes, correct classes | 0.0000 | 0 |
| keep 75% / 50% / 25% of the hits | 0.7492 / 0.4983 / 0.2541 | ≈ the fraction kept |

Controlled degradation — shifting every box by *d* px in x gives IoU exactly
`(w-d)/(w+d)`:

| shift | predicted IoU (w=28) | mAP | AP50 |
|---|---|---|---|
| 0 px | 1.000 | 1.0000 | 1.0000 |
| 2 px | 0.867 | 0.7309 | 1.0000 |
| 4 px | 0.750 | 0.4639 | 1.0000 |
| 7 px | 0.600 | 0.1415 | 0.6873 |
| 12 px | 0.400 | 0.0057 | 0.0480 |

AP50 survives a 4 px shift untouched while mAP has already lost half its value.
That gap *is* the argument for reporting AP@[.5:.95] — AP50 cannot see
localisation quality at all.

**Score ranking is worth as much as box quality.** Ground truth plus an equal
number of random false positives, the same box set both times, only the ranking
swapped:

| ranking | mAP |
|---|---|
| hits scored above the misses | **1.0000** |
| hits scored below the misses | **0.5000** |

Same boxes, same recall, half the AP. A detector whose confidence does not rank
its own hits above its own misses is scored as though it missed them. This is why
step 2b's classical baseline needs a *real* score and not a constant 1.0.

**Two stratification decisions:**

- **Size bands are terciles of this dataset, not COCO's 32 px / 96 px.** Those
  absolute cut-offs were chosen for ~640 px images. At 192 px every object here
  is "small" and the stratification would carry no information. Terciles at 25.5
  and 31.9 px show what a single number hides: a uniform 3 px shift costs
  mAP 0.517 on the smallest third against 0.701 on the largest, because IoU is
  scale-relative.
- **Visibility bands report recall, not AP.** A visibility band can be applied to
  ground truth but not to a detection — the detector never says how occluded it
  thought an object was. So unmatched detections cannot be attributed to a band,
  precision is undefined, and an "AP for heavily occluded objects" computed this
  way would be an artefact of where the *other* bands' false positives landed.

**Cost:** 436 ms for 2000 images × 8992 objects × 3 classes × 10 IoU thresholds.
IoU is built once per image and reused across all ten thresholds, which is the
only optimisation the metric needs.

