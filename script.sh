#!/bin/bash
set -e

echo "=== Step 0: Make sure venv is active ==="
which python
python -V

echo "=== Step 1: Uninstall conflicting packages ==="
pip uninstall -y numpy opencv-python onnxruntime torch torchvision \
  unstructured unstructured-inference

echo "=== Step 2: Purge pip cache ==="
pip cache purge

echo "=== Step 3: Install pinned compatible stack in one go ==="
pip install \
  "numpy==1.26.4" \
  "opencv-python==4.8.1.78" \
  "onnxruntime==1.18.1" \
  "torch==2.2.2" \
  "torchvision==0.17.2" \
  "unstructured==0.18.11" \
  "unstructured-inference==1.0.5" \
  "scipy<1.14"

echo "=== Step 4: Sanity check ==="
python - <<'PY'
import numpy, cv2, onnxruntime, torch
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("ONNXRuntime:", onnxruntime.__version__)
print("Torch:", torch.__version__)
PY
