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
2. **a** — hand-written COCO-style AP@[.5:.95] harness — done. **b** — classical
   CV baseline with a fitted score — done, mAP 0.532 on `hard`. Metric before
   baseline, baseline before model.
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


### Step 2b — classical baseline

Built before the network, not after it, because block 1 showed a carelessly built
baseline inflates the model's apparent win — there, half the CNN's advantage was
a single calibration scalar the baseline had never been given. So this one gets
everything a classical pipeline can fairly have, including a fitted classifier.

Pipeline: estimate the table per image → threshold the residual → split touching
regions → nine shape features per region → multinomial logistic regression for
class and confidence. The classifier is **linear on hand-designed features**,
which is what the classical convention is (a linear model on HOG, on SIFT, on
shape moments). Making it an MLP would quietly turn the baseline into a small
neural network and destroy the comparison it exists for. A fourth class,
background, lets it suppress its own false positives — a hand-built objectness.

**Nothing is thresholded before AP.** AP integrates over every score cut-off, so
discarding low-confidence detections in advance only removes recall the metric
would have credited.

**All tuning was done on `train` and reported on `val`.** Choosing the watershed's
peak-smoothing on the split you then report is exactly how a baseline gets
silently inflated.

#### Results, `val`, 2000 images

| method | regime | mAP | AP50 | AP75 | agnostic mAP | det rate | ms/img |
|---|---|---|---|---|---|---|---|
| otsu | easy | 0.4460 | 0.5178 | 0.4488 | 0.5138 | 0.5482 | 0.36 |
| otsu | **hard** | **0.1430** | 0.2182 | 0.1401 | 0.1699 | 0.2316 | 0.19 |
| bgsub | easy | 0.5002 | 0.5639 | 0.4962 | 0.5307 | 0.5524 | 1.04 |
| bgsub | hard | 0.4114 | 0.4979 | 0.4128 | 0.4761 | 0.5251 | 1.12 |
| bgsub+ws | easy | 0.5895 | 0.7672 | 0.5853 | 0.6658 | 0.7164 | 1.88 |
| **bgsub+ws** | **hard** | **0.5322** | 0.6979 | 0.5637 | 0.6108 | 0.6847 | 2.02 |

#### Four findings

**1. A repair applied to something that is not broken is damage.**

`bgsub` alone leaves a *bimodal* error distribution: 47.3% of objects recovered at
IoU ≥ 0.9, and 38.6% below 0.5 because touching objects merge into one region. The
obvious fix — distance-transform watershed — made things worse when applied to
every region:

| | reg/img | det rate | agnostic mAP | IoU ≥ 0.9 | IoU < 0.5 |
|---|---|---|---|---|---|
| `bgsub` | 3.40 | 0.5211 | 0.3746 | 47.3% | 38.6% |
| watershed on **every** region | 4.29 | 0.4933 | 0.3004 | **6.3%** | 25.0% |
| watershed on **multi-seed regions only** | 4.29 | **0.7067** | **0.5525** | **52.6%** | **14.1%** |

Running the watershed everywhere redraws the boundary of regions that were already
correct, and the near-perfect tier collapsed from 47.3% to 6.3% while the merged
tier only fell from 38.6% to 25.0%. Counting the distance-transform seeds inside
each connected region first, and leaving single-seed regions untouched, keeps the
good tier *and* fixes the merges: **det rate 0.52 → 0.71, agnostic mAP 0.37 →
0.55.** A parameter sweep would never have found this — the sweep converged
towards "split less", which was the wrong axis entirely.

**2. A wrong prior still returns an answer.** Global Otsu on greyscale is a
reasonable-looking method: objects and table differ in brightness, so split the
histogram. Under `easy` it scores mAP 0.446. Under `hard`, where table value is
uniform(0.25, 0.85) and object value uniform(0.45, 1.0), the two distributions
overlap and the split lands inside the objects — mAP **0.143**, a 3.1× collapse,
detection rate 0.232. It never raised an error. It returned a mask every time, and
its per-class numbers on `hard` (box 0.091) read like a weak detector rather than
a broken one. This is block 1's finding 1 reproduced in a different task: report
detection rate beside accuracy, always.

The robust segmenter degrades 18% (`bgsub`, 0.500 → 0.411) and the full pipeline
10% (`bgsub+ws`, 0.590 → 0.532) across the same appearance shift.

**3. Classification is the bottleneck, not localisation.** On `hard`, `bgsub+ws`
scores class-agnostic AP50 **0.823** against class-aware AP50 **0.698**. The boxes
are in the right place; the label on them is wrong 15% of the time. Per class:

| class | AP@[.5:.95] |
|---|---|
| sphere | 0.695 |
| cylinder | 0.510 |
| box | 0.391 |

A sphere's silhouette is a circle from every direction, so nine shape features
describe it completely. A box's silhouette under a tilted camera changes with
yaw, and at 28 px there is not enough of it left for a linear rule. That gap is
the specific thing a learned feature extractor should close in step 3, and it is
now measured rather than assumed.

**4. Occlusion is where the classical pipeline stops, not degrades.** Recall at
IoU 0.5, by how much of the object is visible:

| visible | n | recall |
|---|---|---|
| ≥ 0.9 | 8078 | 0.785 |
| 0.5 – 0.9 | 861 | 0.348 |
| < 0.5 | 53 | **0.000** |

Not a slope — a cliff. A partially hidden object has the wrong silhouette, so
every shape feature it produces is wrong at once. Step 1 flagged that this
dataset has only 10% occluded objects and 0.6% heavily occluded, so this costs
little mAP here; it is recorded because it is the failure a learned detector is
supposed to fix, and because the honest way to test that claim later is to
regenerate the dataset with a lower camera.

**Cost:** 2.02 ms/img for the full pipeline, single-threaded. Block 1's CNN ran at
0.65 ms under PyTorch eager and 0.23 ms under ONNX Runtime, so the classical
pipeline is not the cheap option either.

Qualitative output in `out/detections.png`: boxes land on objects cleanly, and
the visible errors are class flips and near-zero confidences on ambiguous boxes —
consistent with finding 3.

### Viewing the scene

`out/showcase_*.png` render the same `scene.xml` with shadows on and a textured
floor, and `view_live.py` opens it in MuJoCo's interactive viewer with both task
cameras in the dropdown. The dataset renders plain because shadow mapping is what
software rendering is slow at — roughly 130 img/s with shadows against 800+
without, a 6× difference on the only cost that matters during generation.
