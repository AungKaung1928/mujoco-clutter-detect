# CPU-only, reproducible. Runs the three no-data test files by default: the AP
# metric, the target encoding, and the step-6 quantization / pruning checks.
#   docker build -t mujoco-clutter-detect .
#   docker run --rm mujoco-clutter-detect
# The dataset (2.9 GB) and rendering need a GL context and are not part of
# the image; see README "Reproducing" for regenerating them.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MUJOCO_GL=disable

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple -r requirements.txt

COPY *.py *.sh scene.xml README.md ./
COPY runs/*.json runs/detector.onnx runs/det_hard_none.pt ./runs/

CMD ["sh", "-c", "python test_ap.py && python test_detector.py && python test_edge.py"]
