#!/usr/bin/env bash
set -euo pipefail

# ===== ユーザ調整可能な環境変数 =====
: "${MAIN_PY:=/app/ucn_build_prompts.py}"                  # 実行する Python スクリプト
: "${PIP_OVERLAY_DIR:=/data/ucn_prep_cache/pip-overlay}"   # pip overlay 設置先（torch を壊さない）
: "${HF_HOME:=/root/.cache/huggingface}"                   # HF キャッシュ
: "${OPENCLIP_CACHE_DIR:=/root/.cache/huggingface/hub}"    # open_clip のHFキャッシュ
: "${VLM_LOCAL_DIR:=/data/hf_models/Qwen/Qwen3-VL-32B-Instruct}"  # Qwen3-VL-32B の固定保存先
: "${QWEN_QUANT:=4bit}"  # 4bit|8bit|none

# 依存解決なし（torch を巻き込ませない；nodeps）
: "${PIP_NODEPS:=open_clip_torch==3.2.0 bitsandbytes>=0.48.2 accelerate==1.11.0}"

# 依存解決あり（torch系は絶対入れない）
: "${PIP_INSTALL:=transformers==4.57.1 huggingface_hub==0.36.0 safetensors pandas tqdm pillow opencv-python-headless psutil matplotlib}"


export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
mkdir -p "${PIP_OVERLAY_DIR}" "${HF_HOME}"

echo "[prompts-entry-v2] torch summary (python3):"
python3 - <<'PY'
try:
    import torch
    print(f"  torch={torch.__version__}, torch.version.cuda={getattr(torch.version,'cuda',None)}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device0={torch.cuda.get_device_name(0)}")
    print(f"  torch.__file__={getattr(torch,'__file__',None)}")
except Exception as e:
    print("  torch import failed:", e)
PY

# ===== pip overlay 安全インストール（torchは絶対入れない） =====
pip_base=(python3 -m pip install --upgrade --no-cache-dir --progress-bar off --target "${PIP_OVERLAY_DIR}")

if [[ -n "${PIP_NODEPS}" ]]; then
  echo "[prompts-entry-v2] nodeps install: ${PIP_NODEPS}"
  "${pip_base[@]}" --no-deps ${PIP_NODEPS}
fi
if [[ -n "${PIP_INSTALL}" ]]; then
  echo "[prompts-entry-v2] normal install: ${PIP_INSTALL}"
  "${pip_base[@]}" ${PIP_INSTALL}
fi

export PYTHONPATH="${PIP_OVERLAY_DIR}:${PYTHONPATH:-}"
export HF_HOME OPENCLIP_CACHE_DIR QWEN_QUANT

# ===== 事前ダウンロード（オンライン時のみ） =====
if [[ "${TRANSFORMERS_OFFLINE:-0}" != "1" ]]; then
  echo "[prompts-entry-v2] snapshot_download for Qwen (HF) & cache open_clip SigLIP (webli)"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

VLM_LOCAL_DIR = os.environ.get("VLM_LOCAL_DIR","/data/hf_models/Qwen/Qwen3-VL-32B-Instruct")
os.makedirs(VLM_LOCAL_DIR, exist_ok=True)

def ok_dir(p):
    try:
        return any(n.endswith(".safetensors") for n in os.listdir(p))
    except Exception:
        return False

# Qwen 本体
if not ok_dir(VLM_LOCAL_DIR):
    print(f"[snap] downloading Qwen -> {VLM_LOCAL_DIR}")
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-32B-Instruct",
        local_dir=VLM_LOCAL_DIR,
        local_dir_use_symlinks=False,
        max_workers=16
    )
else:
    print(f"[snap] Qwen exists: {VLM_LOCAL_DIR}")
PY

  # open_clip 側の SigLIP (ViT-SO400M-14, 384, webli) をキャッシュ
  python3 - <<'PY'
import os
import open_clip
cache_dir = os.environ.get("OPENCLIP_CACHE_DIR","/root/.cache/huggingface/hub")
print(f"[snap] caching open_clip SigLIP weights (webli) into {cache_dir} …")
_ = open_clip.create_model_and_transforms(
        'ViT-SO400M-14-SigLIP-384',
        pretrained='webli',
        cache_dir=cache_dir,
    )
print("[snap] open_clip SigLIP cached")
PY
fi

# 以降は（基本）オフラインでもOK
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# 主要モジュールの読み取り確認
python3 - <<'PY'
import importlib, os
mods = ["transformers","huggingface_hub","safetensors","open_clip","bitsandbytes","tokenizers","cv2"]
for m in mods:
    try:
        mod = importlib.import_module(m if m!="cv2" else "cv2")
        v = getattr(mod, "__version__", "OK")
        p = getattr(mod, "__file__", "builtin")
        print(f"[prompts-entry-v2] import {m}: OK ({v}) path={p}")
    except Exception as e:
        print(f"[prompts-entry-v2] import {m}: MISSING ({e})")
PY

# 実行（python3 固定）
exec python3 -u "${MAIN_PY}" "$@"
