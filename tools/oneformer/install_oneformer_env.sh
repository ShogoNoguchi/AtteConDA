#!/usr/bin/env bash
# =========================================================
# RTX5090 + CUDA 12.8 向け OneFormer(Transformers) 推論用 環境構築スクリプト（Conda）
# - 新規 conda 環境 oneformer_cu128 を作成
# - PyTorch 2.7.x (CUDA 12.8) を公式インデックスからインストール
# - 推論に必要な Python パッケージを最小限かつ安定構成で導入
# - 既存環境は一切変更しない（非破壊）
# =========================================================
set -euo pipefail

ENV_NAME="oneformer_cu128"
PYVER="3.10"

echo "[INFO] Create conda env: ${ENV_NAME} (Python ${PYVER})"
conda create -y -n "${ENV_NAME}" python="${PYVER}"

echo "[INFO] Activate env"
# conda init 済みを仮定。未設定なら: source ~/anaconda3/etc/profile.d/conda.sh など
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[INFO] Upgrade pip"
python -m pip install --upgrade pip

echo "[INFO] Install PyTorch (CUDA 12.8) from official cu128 index"
# 公式ブログ: "pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128"
# ただし torchvision/torchaudio のバージョンは OS/ABI により解決されるので index-url 指定で一括導入
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "[INFO] Install runtime deps (Transformers版 OneFormer で推論に必要なもの)"
# - transformers: OneFormer実装
# - opencv-python-headless: 画像I/O（GUI不要）
# - pillow,numpy,tqdm,huggingface-hub: 基本ユーティリティ
# 破壊的アップデートを避けるため、大幅メジャー更新を跨がない緩めの上限を掛ける
pip install --no-cache-dir \
  "transformers>=4.41,<5" \
  "opencv-python-headless>=4.11" \
  "pillow>=10.0" \
  "numpy>=1.24" \
  "tqdm>=4.66" \
  "huggingface-hub>=0.23"

echo "[INFO] Quick sanity check"
python - <<'PY'
import torch, cv2, transformers
print("Torch Version:", torch.__version__)
print("Torch CUDA Build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("cv2 version:", cv2.__version__)
print("Transformers:", transformers.__version__)
PY

echo "[INFO] Done. Use:  conda activate ${ENV_NAME}"
