// The clutter detector end to end in C++: uint8 RGB image -> normalise -> ONNX Runtime
// (1 intra-op thread, batch 1) -> CenterNet decode (3x3 max keep, top-k, offset + size)
// -> score threshold. Same steps as export_onnx.onnx_predict, checked against it.
// Inputs from cpp/bench.py: images.u8 (N x 192 x 192 x 3, HWC RGB) and ref.f32
// (N x K x 7: flat index, class, score, x0, y0, x1, y1 from Python ORT + detector.decode).
// Usage: detect_ort <model.onnx> <images.u8> <ref.f32> [n_time]
// Prints one JSON line; exit 1 when C++ and Python disagree.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr int kSize = 192, kStride = 4, kGrid = kSize / kStride, kCls = 3, kTopK = 10;
constexpr float kMean = 0.45f, kStd = 0.25f, kThr = 0.02f;  // train_det.MEAN/STD, predict thr

struct Det { int idx, cls; float score, x0, y0, x1, y1; };

template <typename T>
std::vector<T> read_all(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
  std::vector<T> v(static_cast<size_t>(f.tellg()) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  return v;
}

// HWC uint8 -> CHW float, (x / 255 - mean) / std.
void normalise(const uint8_t* img, float* chw) {
  constexpr int hw = kSize * kSize;
  for (int p = 0; p < hw; ++p)
    for (int c = 0; c < 3; ++c) chw[c * hw + p] = (img[p * 3 + c] / 255.0f - kMean) / kStd;
}

// detector.decode for one image: a cell is a peak when it equals the max of its 3x3
// neighbourhood (max_pool2d(3, 1, 1) pads with -inf); top-k over class x cell, ties by
// lower flat index; wh clamped at 0. Returns all k, threshold applied by the caller.
std::vector<Det> decode(const float* hm, const float* wh, const float* off) {
  constexpr int hw = kGrid * kGrid, n = kCls * hw;
  std::vector<float> s(n);
  for (int c = 0; c < kCls; ++c)
    for (int y = 0; y < kGrid; ++y)
      for (int x = 0; x < kGrid; ++x) {
        const float v = hm[c * hw + y * kGrid + x];
        float m = -std::numeric_limits<float>::infinity();
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < kGrid && xx >= 0 && xx < kGrid) m = std::max(m, hm[c * hw + yy * kGrid + xx]);
          }
        s[c * hw + y * kGrid + x] = v == m ? v : 0.0f;
      }
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + kTopK, idx.end(),
                    [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
  std::vector<Det> out;
  for (int j = 0; j < kTopK; ++j) {
    const int i = idx[j], pix = i % hw, ys = pix / kGrid, xs = pix % kGrid;
    const float cx = (xs + off[pix]) * kStride, cy = (ys + off[hw + pix]) * kStride;
    const float bw = std::max(wh[pix], 0.0f) * kStride, bh = std::max(wh[hw + pix], 0.0f) * kStride;
    out.push_back({i, i / hw, s[i], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2});
  }
  return out;
}

double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(q * (v.size() - 1))];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: %s model.onnx images.u8 ref.f32 [n_time]\n", argv[0]); return 2; }
  const int n_time = argc > 4 ? std::atoi(argv[4]) : 1000;
  const auto imgs = read_all<uint8_t>(argv[2]);
  const auto ref = read_all<float>(argv[3]);
  constexpr size_t img_bytes = size_t(kSize) * kSize * 3;
  const size_t n_img = imgs.size() / img_bytes;
  if (n_img == 0 || imgs.size() != n_img * img_bytes || ref.size() != n_img * kTopK * 7) {
    std::fprintf(stderr, "images/ref size mismatch\n");
    return 2;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "detect");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(1);
  so.SetInterOpNumThreads(1);
  Ort::Session sess(env, argv[1], so);
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<float> x(3 * kSize * kSize), hm(kCls * kGrid * kGrid), wh(2 * kGrid * kGrid), off(2 * kGrid * kGrid);
  const int64_t xs[4] = {1, 3, kSize, kSize}, hs[4] = {1, kCls, kGrid, kGrid}, ws[4] = {1, 2, kGrid, kGrid};
  Ort::Value in = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), xs, 4);
  Ort::Value outs[3] = {Ort::Value::CreateTensor<float>(mem, hm.data(), hm.size(), hs, 4),
                        Ort::Value::CreateTensor<float>(mem, wh.data(), wh.size(), ws, 4),
                        Ort::Value::CreateTensor<float>(mem, off.data(), off.size(), ws, 4)};
  const char* iname[] = {"image"};
  const char* onames[] = {"hm", "wh", "off"};
  Ort::RunOptions ro;
  auto run = [&](size_t i) {
    normalise(imgs.data() + i * img_bytes, x.data());
    sess.Run(ro, iname, &in, 1, onames, outs, 3);
  };

  // Parity: every image, all k detections, against Python.
  double max_diff = 0;
  int cls_mismatch = 0, idx_mismatch = 0, kept = 0;
  for (size_t i = 0; i < n_img; ++i) {
    run(i);
    const auto d = decode(hm.data(), wh.data(), off.data());
    for (int j = 0; j < kTopK; ++j) {
      const float* r = &ref[(i * kTopK + j) * 7];
      idx_mismatch += d[j].idx != static_cast<int>(r[0]);
      cls_mismatch += d[j].cls != static_cast<int>(r[1]);
      const float got[5] = {d[j].score, d[j].x0, d[j].y0, d[j].x1, d[j].y1};
      for (int c = 0; c < 5; ++c) max_diff = std::max(max_diff, double(std::fabs(got[c] - r[2 + c])));
      kept += d[j].score > kThr;
    }
  }

  // Timing, batch 1: model only, and end to end (normalise, model, decode, threshold).
  for (int i = 0; i < 50; ++i) run(i % n_img);
  std::vector<double> t_model(n_time), t_e2e(n_time), t_decode(n_time);
  int sink = 0;
  for (int i = 0; i < n_time; ++i) {
    const size_t k = i % n_img;
    normalise(imgs.data() + k * img_bytes, x.data());
    auto t0 = std::chrono::steady_clock::now();
    sess.Run(ro, iname, &in, 1, onames, outs, 3);
    auto t1 = std::chrono::steady_clock::now();
    t_model[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    run(k);
    auto t2 = std::chrono::steady_clock::now();
    for (const Det& d : decode(hm.data(), wh.data(), off.data())) sink += d.score > kThr;
    auto t3 = std::chrono::steady_clock::now();
    t_decode[i] = std::chrono::duration<double, std::milli>(t3 - t2).count();
    t_e2e[i] = std::chrono::duration<double, std::milli>(t3 - t0).count();
  }
  const bool ok = max_diff < 1e-4 && cls_mismatch == 0 && idx_mismatch == 0;
  std::printf("{\"ort\": \"%s\", \"images\": %zu, \"detections_above_thr\": %d, \"max_abs_diff\": %.3e, "
              "\"class_mismatch\": %d, \"index_mismatch\": %d, \"n_time\": %d, "
              "\"model_ms\": [%.4f, %.4f], \"decode_ms\": [%.4f, %.4f], \"e2e_ms\": [%.4f, %.4f], \"sink\": %d}\n",
              OrtGetApiBase()->GetVersionString(), n_img, kept, max_diff, cls_mismatch, idx_mismatch, n_time,
              pct(t_model, 0.5), pct(t_model, 0.99), pct(t_decode, 0.5), pct(t_decode, 0.99),
              pct(t_e2e, 0.5), pct(t_e2e, 0.99), sink);
  return ok ? 0 : 1;
}
