#!/usr/bin/env bash
# Steps 3(easy) -> 4 -> 5, serially. Never two trainings at once: 8 threads
# shared between two jobs makes both slower and neither number trustworthy.
#
# The first version of this script waited with `pgrep -f "train_det.py --regime
# hard --aug none"`. That deadlocked: the parent shell that launched this file
# had the script's own text on its command line, so `pgrep -f` matched itself
# and the loop never exited. Serialisation is now handled by the caller.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "[$(date +%T)] step 3 easy"
nice -n 10 "$PY" train_det.py --regime easy --aug none --epochs 20 \
     --save runs/det_easy_none.json --ckpt runs/det_easy_none.pt > runs/log_easy_none.txt 2>&1
echo "[$(date +%T)] step 4 ablation"
nice -n 10 "$PY" run_ablation.py --epochs 12 --fit-n 6000 \
     --out runs/ablation.json > runs/log_ablation.txt 2>&1
echo "[$(date +%T)] step 5 onnx"
nice -n 10 "$PY" export_onnx.py --ckpt runs/det_hard_none.pt --regime hard \
     --save runs/onnx.json > runs/log_onnx.txt 2>&1
echo "[$(date +%T)] all done"
