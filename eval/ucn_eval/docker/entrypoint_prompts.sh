#!/usr/bin/env bash
set -euo pipefail

# ===== ユーザ調整可能な環境変数 =====
: "${MAIN_PY:=/app/ucn_build_prompts.py}"                  # 実行する Python スクリプト
: "${PIP_OVERLAY_DIR:=/data/ucn_prep_cache/pip-overlay}"   # pip overlay 設置先（torch を壊さない）
: "${HF_HOME:=/root/.cache/huggingface}"                   # HF キャッシュ
: "${OPENCLIP_CACHE_DIR:=/root/.cache/huggingface/hub}"    # open_clip のHFキャッシュ
: "${VLM_LOCAL_DIR:=/data/hf_models/Qwen/Qwen3-VL-32B-Instruct}"  # Qwen3-VL-32B の固定保存先

# 依存解決あり（torch系は絶対入れない）
: "${PIP_INSTALL:=transformers==4.57.1 huggingface_hub==0.36.0 accelerate safetensors pandas tqdm pillow opencv-python-headless}"

# 依存解決なし（torch を巻き込ませない；nodeps）
: "${PIP_NODEPS:=open_clip_torch==3.2.0 bitsandbytes>=0.44.1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
mkdir -p "${PIP_OVERLAY_DIR}" "${HF_HOME}"

echo "[prompts-entry] torch summary (python3):"
python3 - <<'PY'
try:
    import torch
    print(f"  torch={torch.__version__}, torch.version.cuda={getattr(torch.version,'cuda',None)}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device0={torch.cuda.get_device_name(0)}")
except Exception as e:
    print("  torch import failed:", e)
PY

# ===== pip overlay 安全インストール（torchは絶対入れない） =====
pip_base=(python3 -m pip install --upgrade --no-cache-dir --progress-bar off --target "${PIP_OVERLAY_DIR}")

if [[ -n "${PIP_NODEPS}" ]]; then
  echo "[prompts-entry] nodeps install: ${PIP_NODEPS}"
  "${pip_base[@]}" --no-deps ${PIP_NODEPS}
fi
if [[ -n "${PIP_INSTALL}" ]]; then
  echo "[prompts-entry] normal install: ${PIP_INSTALL}"
  "${pip_base[@]}" ${PIP_INSTALL}
fi

export PYTHONPATH="${PIP_OVERLAY_DIR}:${PYTHONPATH:-}"
export HF_HOME OPENCLIP_CACHE_DIR

# ===== 事前ダウンロード（python3；huggingface-cli 不要） =====
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

VLM_LOCAL_DIR = os.environ.get("VLM_LOCAL_DIR","/data/hf_models/Qwen/Qwen3-VL-32B-Instruct")
os.makedirs(VLM_LOCAL_DIR, exist_ok=True)

def ok_dir(p):
    try:
        return any(n.endswith(".safetensors") for n in os.listdir(p))
    except Exception:
        return False

# Qwen3-VL-32B-Instruct → /data/hf_models/Qwen/Qwen3-VL-32B-Instruct
if not ok_dir(VLM_LOCAL_DIR):
    print(f"[prompts-entry] downloading Qwen to: {VLM_LOCAL_DIR}")
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-32B-Instruct",
        local_dir=VLM_LOCAL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.safetensors","*.json","*.model","*.bin","*.txt",
            "tokenizer*","image*","*.md","*.py"
        ]
    )
else:
    print(f"[prompts-entry] Qwen exists: {VLM_LOCAL_DIR}")

# SigLIP は HF キャッシュへ（open_clip が参照）
print("[prompts-entry] caching SigLIP (timm/ViT-SO400M-14-SigLIP-384)…")
snapshot_download(repo_id="timm/ViT-SO400M-14-SigLIP-384", repo_type="model")
print("[prompts-entry] prefetch done")
PY

# 以降は（基本）オフライン
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

# 実行（python3 固定）
exec python3 -u "${MAIN_PY}" "$@"
