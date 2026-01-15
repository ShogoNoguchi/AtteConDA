#!/usr/bin/env bash
set -Eeuo pipefail

# ====== 設定（環境変数） ======
: "${MAIN_PY:?MAIN_PY must be set (e.g., /app/gen_prompts_synad.py)}"
PIP_OVERLAY_DIR="${PIP_OVERLAY_DIR:-/data/ucn_prompt_cache/pip-overlay}"
REQS_OVERLAY_PATH="${REQS_OVERLAY_PATH:-}"
PIP_INSTALL="${PIP_INSTALL:-}"
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

# Python 実行バイナリの自動検出（python3 優先）
PY_BIN="${PY_BIN:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN=python
fi

# ====== 便利表示 ======
_now() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(_now)] [entrypoint-nodeps] $*"; }

# ====== パス準備 ======
mkdir -p "$PIP_OVERLAY_DIR"
export PYTHONPATH="${PIP_OVERLAY_DIR}:${PYTHONPATH:-}"

# ====== 既存 Torch/CUDA の可視化（ここで torch を上書きしないことが重要） ======
"$PY_BIN" - <<'PY'
import sys
try:
  import torch
  print(f"[entrypoint-nodeps] torch={getattr(torch,'__version__','NA')}, cuda_build={getattr(getattr(torch,'version',None),'cuda',None)}, cuda_available={torch.cuda.is_available()}")
  if torch.cuda.is_available():
      try:
          print(f"[entrypoint-nodeps] device0={torch.cuda.get_device_name(0)}")
      except Exception as e:
          print(f"[entrypoint-nodeps] device0=<unknown> ({e})")
except Exception as e:
  print(f"[entrypoint-nodeps] torch=none (import error: {e})")
PY

log "PYTHONPATH head: ${PIP_OVERLAY_DIR}"

# ====== nodeps インストール関数 ======
py_have () {
  local mod="$1"
  "$PY_BIN" - <<PY
import importlib, sys
mod = "${mod}"
try:
    importlib.import_module(mod)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

# pip パッケージ名 → import モジュール名 変換
to_import_mod () {
  case "$1" in
    open_clip_torch* ) echo "open_clip" ;;
    opencv-python*  )  echo "cv2" ;;
    pillow*         )  echo "PIL" ;;
    huggingface-hub*)  echo "huggingface_hub" ;;
    qwen-vl-utils*  )  echo "qwen_vl_utils" ;;
    transformers*   )  echo "transformers" ;;
    tokenizers*     )  echo "tokenizers" ;;
    tqdm*           )  echo "tqdm" ;;
    pandas*         )  echo "pandas" ;;
    einops*         )  echo "einops" ;;
    requests*       )  echo "requests" ;;
    *)                echo "$1" ;;
  esac
}

# ====== NO-DEPS インストール（必要なものだけ） ======
if [[ -n "${PIP_INSTALL}" ]]; then
  log "Installing (nodeps) into overlay: ${PIP_INSTALL}"
  IFS=' ' read -r -a PKGS <<< "${PIP_INSTALL}"
  for pkg in "${PKGS[@]}"; do
    mod="$(to_import_mod "${pkg}")"
    if py_have "${mod}"; then
      log "skip (already present): ${pkg}  [import ${mod}]"
      continue
    fi
    log "pip install --no-deps --upgrade --target ${PIP_OVERLAY_DIR} ${pkg}"
    "$PY_BIN" -m pip install --no-deps --upgrade --target "${PIP_OVERLAY_DIR}" "${pkg}"
  done
else
  log "PIP_INSTALL is empty; no overlay install."
fi

# ====== （誤って overlay に入ってしまった重量級を掃除：予防） ======
log "pruning heavy packages accidentally installed into overlay (if any)"
for name in torch torchvision torchaudio triton nvidia cuda cudnn cublas cufft cusparse cufile; do
  find "${PIP_OVERLAY_DIR}" -maxdepth 1 -type d \( -name "${name}" -o -name "${name}-*" -o -name "${name}*.dist-info" -o -name "${name}*.libs" \) -print -exec rm -rf {} +
done

# ====== バージョン検査 ======
"$PY_BIN" - <<'PY'
def show(mod):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "n/a")
        print(f"[entrypoint-nodeps] import {mod}: OK (v={v})")
        return True
    except Exception as e:
        print(f"[entrypoint-nodeps] import {mod}: MISSING ({e})")
        return False
for m in ["transformers","huggingface_hub","tokenizers","open_clip","cv2","numpy","PIL","tqdm","einops","requests","pandas"]:
    show(m)
PY

# ====== 実行 ======
log "EXEC: ${MAIN_PY} $*"
exec "$PY_BIN" -u "${MAIN_PY}" "$@"
