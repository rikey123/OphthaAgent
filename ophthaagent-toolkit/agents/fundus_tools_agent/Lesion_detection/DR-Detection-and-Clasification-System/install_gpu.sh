#!/usr/bin/env bash
set -e

ENV_NAME="ddcs"

# 0) 先确认显卡驱动是否存在（TF 官方也要求先装驱动）:contentReference[oaicite:4]{index=4}
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "WARNING: nvidia-smi not found. GPU may not work."

# 1) 建环境
conda env create -f base.yml || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

# 2) TensorFlow GPU：官方推荐这样装（会按 and-cuda 拉 GPU 相关依赖）:contentReference[oaicite:5]{index=5}
python -m pip install "tensorflow[and-cuda]==2.16.1"

# 3) PyTorch GPU：用 PyTorch 官方 index-url（2.7.1 支持 cu118/cu126/cu128）:contentReference[oaicite:6]{index=6}
# 方案 A：CUDA 12.6
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

# 方案 B：CUDA 12.8（如果你更想对齐 12.8）
# python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
#   --index-url https://download.pytorch.org/whl/cu128

# 4) 装其它依赖（锁版本）
python -m pip install -r requirements.base.txt

# 5) 验证 GPU
python - << 'PY'
import tensorflow as tf
print("TF GPUs:", tf.config.list_physical_devices('GPU'))
PY

python - << 'PY'
import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
PY
