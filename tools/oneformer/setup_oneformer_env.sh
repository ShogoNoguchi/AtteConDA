#!/usr/bin/env bash
set -euxo pipefail

ENV_NAME="oneformer_env"
PYTHON_VER="3.10"

# 1) 新規 conda 環境
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VER}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 2) pip を最新化
python -m pip install --upgrade pip wheel setuptools

# 3) PyTorch (公式 cu128 wheel; 2.7.1 安定版)
#    公式ページの記載通り、--index-url を cu128 にする
#    参考: https://pytorch.org/get-started/previous-versions/ の v2.7.1 セクション
pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"

# 4) 推論に必要なライブラリ（NumPy 1.x 系で固定して衝突を避ける）
#    ・transformers は OneFormer を含む安定帯 (4.41+)
#    ・OpenCV は 4.10 系 (headless) に固定 → NumPy 1.26.4 で安定
pip install \
  "transformers>=4.41,<4.46" \
  "huggingface_hub>=0.24" \
  "safetensors>=0.4.3" \
  "timm==0.9.16" \
  "opencv-python-headless==4.10.0.84" \
  "numpy==1.26.4" \
  "pillow>=10.2" "tqdm>=4.66"

# 5) 動作確認
python - <<'PY'
import torch, platform
print("torch           =", torch.__version__)
print("torch.version.cuda (build) =", getattr(torch.version, "cuda", None))
print("cuda.is_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))
print("python         =", platform.python_version())
PY

echo "=== READY: conda activate ${ENV_NAME} ==="
