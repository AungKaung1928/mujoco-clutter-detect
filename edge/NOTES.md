# What limits each backend

The CenterNet-style detector with 380,631 (onnx.json › params) parameters, batch 1, 192 px,
`hard/val`, one shared x86 laptop. Every number below is followed by the tracked file and
key it comes from (`runs/<file> › key/path`); `python edge/test_notes.py` checks each one.
Timings move between sessions on this laptop (last section): compare rows inside one file only.

| Backend | Measured | What limits it |
|---|---|---|
| PyTorch eager fp32 | 1 thread 5.14 ms (onnx.json › latency_ms/torch_eager_1t), 4 threads 2.54 ms (onnx.json › latency_ms/torch_eager_4t), 8 threads 2.75 ms (onnx.json › latency_ms/torch_eager_8t) | The runtime, not the arithmetic: the same graph in ONNX Runtime takes 3.69 ms (onnx.json › latency_ms/onnxrt_1t) on 1 thread at the same mAP 0.9127 (onnx.json › mAP_onnx). More than 4 threads is slower. |
| ONNX Runtime fp32 | 1 thread 3.57 ms (quant_minmax.json › ref/latency_ms/1t), 4 threads 1.52 ms (quant_minmax.json › ref/latency_ms/4t) | Thread scaling flattens: 4 to 8 threads goes 1.12 ms (onnx.json › latency_ms/onnxrt_4t) to 1.01 ms (onnx.json › latency_ms/onnxrt_8t). Why (not profiled): batch 1 at 192 px leaves little work per layer to split; ORT profiling would show it. |
| ONNX Runtime INT8 static (QDQ) | 1 thread 1.38 ms (quant_minmax.json › latency_ms/1t), 4 threads 0.61 ms (quant_minmax.json › latency_ms/4t); mAP 0.9097 (quant_minmax.json › score/mAP) vs 0.9107 (quant_minmax.json › ref/score/mAP) | Exact reproducibility, not speed: the heatmap sits on few levels, so 144 (cpp_ort.json › images_with_tie_at_k) of 200 images tie at the 10th score and top-k order is not unique; a second implementation needs a tie rule to match. Calibrated on 200 (quant_minmax.json › calib_images) sim images only; on the real HOPE photos it scores 0.0061 (real_hope_onnx.json › models/detector_int8_minmax.onnx/mAP) vs fp32 0.0063 (real_hope_onnx.json › models/detector.onnx/mAP), so calibration adds no real-photo loss (the photos fail anyway, see README). |
| ONNX Runtime INT8 dynamic | 1 thread 1.82 ms (quant_dynamic.json › latency_ms/1t), 4 threads 1.54 ms (quant_dynamic.json › latency_ms/4t) | Barely scales with threads, while static INT8 does. The graph has 15 (quant_dynamic.json › nodes/ConvInteger) ConvInteger and 115 (quant_dynamic.json › nodes/total) nodes against 92 (quant_minmax.json › nodes/total) for static: activations are quantised inside the graph on every call ([ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)). |
| Pruned 50 % channels, ORT fp32 | 1 thread 2.42 ms (prune_50.json › latency_ms/onnxrt_1t); mAP 0.8454 (prune_50.json › score/mAP) | Latency does not follow compute: MACs drop from 214,990,848 (prune_50.json › macs_base) to 101,795,328 (prune_50.json › macs), time only from 3.57 to 2.42 ms. Pruned + INT8 reaches 1.25 ms (prune_50.json › int8/latency_ms/onnxrt_1t), little under INT8 alone, for mAP 0.8411 (prune_50.json › int8/score/mAP). |

## Around the model: Python vs C++ (one process, 5 interleaved rounds)

INT8 static, 1 thread, median per image (`cpp/bench.py`):

- Python `onnx_predict`: model 1.222 ms (cpp_ort.json › python_ms_p50/model_ms), end to end 1.609 ms (cpp_ort.json › python_ms_p50/e2e_ms).
- C++ `cpp/detect_ort.cpp`: model 1.1583 ms (cpp_ort.json › cpp_ms_p50/model_ms), end to end 1.2601 ms (cpp_ort.json › cpp_ms_p50/e2e_ms), decode 0.0466 ms (cpp_ort.json › cpp_ms_p50/decode_ms).

So the model call costs the same from both languages; the Python limit is normalise + decode per
image. That wrapper decides the race with the classical pipeline: timed together on one thread,
classical 1.8029 ms (edge_vs_classical.json › ms_per_img/classical bgsub+ws/median) against INT8
end to end in Python 1.9444 ms (edge_vs_classical.json › ms_per_img/int8 end to end/median).
C++ end to end against classical in one process: not measured yet.

## The host

- Tail: p99 end to end 5.4061 ms (cpp_ort.json › cpp_ms_p99/e2e_ms) in C++ and 6.0375 ms (cpp_ort.json › python_ms_p99/e2e_ms) in Python, about 4x the median on both sides: set by the shared laptop, not by the code. Not a deployment number.
- Drift: the same INT8 graph on 1 thread measured 1.38 ms (quant_minmax.json › latency_ms/1t), 1.5649 ms (edge_vs_classical.json › ms_per_img/int8 model/median) and 1.222 ms (cpp_ort.json › python_ms_p50/model_ms) in three sessions.
