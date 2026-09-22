# Issues to open on GitHub

One block per issue: title, then body. These are the known gaps at the point
step 6's code landed, written down so the tracker says what is not done
instead of the README implying it is.

---

**Step 6 numbers are not measured yet**

`quantize.py`, `prune.py` and `edge_curve.py` are written and smoke-tested on
a 100-image subset. The README table carries `TODO(measure)` until the full
`hard/val` run is done on an idle box with the load average recorded, per the
step 5 rule. Commands are in `verify.sh`.

---

**Percentile calibration lost 0.014 mAP on the smoke subset; MinMax and Entropy lost 0.002**

On 100 images with 32 calibration images, `percentile` (99.99) clipped the
box-class heatmap harder than the other two methods. Either the percentile is
too aggressive for a heatmap whose peaks are the whole signal, or 32 images
is too few for a histogram method. Re-check at 200 calibration images on the
full set before drawing a conclusion.

---

**Pruning and INT8 are only combined, not studied**

`prune.py --int8` quantizes the pruned graph with MinMax after fine-tuning.
Whether the pruned network is more or less sensitive to quantization than
the full one (fewer channels, larger per-channel ranges) is not measured
separately. One extra row: INT8 drop on pruned vs on full, same calibration.

---

**Quantization-aware training was not attempted**

PTQ only. If the INT8 mAP drop on the full set is above ~0.01, QAT with fake
quantization in the training loop is the next step, and the fine-tune entry
point in `prune.py` is where it would plug in.

---

**A one-thread laptop latency is not a Jetson latency**

Every ms/img here is an x86 core with AVX2/AVX-VNNI under ONNX Runtime's CPU
provider. The deployment target is an ARM Cortex-A78 plus TensorRT, where the
INT8 path is different hardware and a different runtime. The ordering of the
graphs is expected to hold; the ratios are not. No Jetson is available to
this project, so the claim stays "1 thread, x86".

---

**Pruning saliency is BatchNorm gamma only**

Network slimming picks channels by |gamma| with no sparsity penalty during
the original training, so gammas are not driven toward zero and the ranking
is weak (25% global pruning cost 0.70 mAP before fine-tuning on the smoke
subset). A short L1-on-gamma fine-tune before pruning, or a Taylor/first-order
saliency, would give a better ranking. Not started.
