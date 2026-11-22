#!/usr/bin/env bash
# CI: build detect_ort against ONNX Runtime 1.30.0 and require C++ == Python on synthetic scenes
# (data/ is not in the repo), for the INT8 and the fp32 graph. Headers: conda-forge
# onnxruntime-cpp 1.30.0 (sha256 pinned). Library: the onnxruntime 1.30.0 pip wheel.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="${PY:-python}"
work="${WORK:-$(mktemp -d)}"
pkg=onnxruntime-cpp-1.30.0-h987d1f6_1_cpu
sha=6f042fb15ce643fe954c1eefef3bc3c02a2f302a1b2c76d4bbea8eedaca75e3e

if [ ! -f "$work/$pkg.conda" ]; then
  curl -sSfL -o "$work/$pkg.conda" "https://conda.anaconda.org/conda-forge/linux-64/$pkg.conda"
fi
echo "$sha  $work/$pkg.conda" | sha256sum -c -
mkdir -p "$work/x" "$work/ort"
"$py" -m zipfile -e "$work/$pkg.conda" "$work/x/"
zstd -dc "$work/x/pkg-$pkg.tar.zst" | tar -x -C "$work/ort" include
lib="$("$py" -c 'import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))')"
mkdir -p "$work/lib"
ln -sf "$lib/libonnxruntime.so.1.30.0" "$work/lib/libonnxruntime.so.1"

ORT_INCLUDE="$work/ort/include/onnxruntime/core/session" ORT_LIB="$work/lib" "$here/build.sh"
for m in runs/detector_int8_minmax.onnx runs/detector.onnx; do
  "$py" "$here/bench.py" --model "$m" --synthetic --n 50 --n-time 100 --rounds 1
done
# A bad timing count must be refused (exit 2), not crash: n_time 0 used to segfault, -1 abort.
for n in 0 -1; do
  rc=0; "$here/detect_ort" runs/detector.onnx "$here/images.u8" "$here/ref.f32" "$n" 2>/dev/null || rc=$?
  [ "$rc" -eq 2 ] || { echo "detect_ort n_time $n: exit $rc, want 2"; exit 1; }
done
echo "detect_ort refuses n_time 0 and -1"
