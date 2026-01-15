#!/usr/bin/env bash
set -euo pipefail

# ============= GPU / Torch 情報 =============
nvidia-smi || true
python3 - <<'PY'
import sys
try:
    import torch
    print(f"[entrypoint] torch={torch.__version__}, torch.version.cuda(build)={getattr(torch.version,'cuda',None)}")
    print(f"[entrypoint] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[entrypoint] device0={torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[entrypoint] FATAL: import torch failed: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ============= 永続 pip オーバーレイ（再ビルド不要/ハッシュ差分適用） =============
PIP_OVERLAY_DIR="${PIP_OVERLAY_DIR:-/data/ucn_eval_cache/pip-overlay}"
REQS_OVERLAY_PATH="${REQS_OVERLAY_PATH:-/data/ucn_eval_cache/requirements.overlay.txt}"

export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "$PIP_OVERLAY_DIR"
export PYTHONPATH="${PIP_OVERLAY_DIR}:${PYTHONPATH:-}"

_overlay_inputs=""
if [ -n "${PIP_INSTALL:-}" ]; then _overlay_inputs="${PIP_INSTALL}"; fi
if [ -f "$REQS_OVERLAY_PATH" ]; then
  _overlay_inputs="${_overlay_inputs}"$'\n'"$(cat "$REQS_OVERLAY_PATH")"
fi

if [ -n "${_overlay_inputs}" ]; then
  need_hash="$(printf "%s" "${_overlay_inputs}" | sha1sum | awk '{print $1}')"
  prev_hash="$(cat "${PIP_OVERLAY_DIR}/.overlay_hash" 2>/dev/null || echo "")"
  if [ "${need_hash}" != "${prev_hash}" ]; then
    echo "[entrypoint] Installing/updating overlay packages into: ${PIP_OVERLAY_DIR}"

    SAFE_REQ="${PIP_OVERLAY_DIR}/.reqs.safe.txt"
    NODEPS_REQ="${PIP_OVERLAY_DIR}/.reqs.nodeps.txt"
    rm -f "$SAFE_REQ" "$NODEPS_REQ"

    # --- REQS ファイルの取り込み（先頭末尾のクォートを除去してから分類） ---
    if [ -f "$REQS_OVERLAY_PATH" ]; then
      while IFS= read -r line; do
        trimmed="$(printf '%s' "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [ -z "$trimmed" ] && continue
        # 先頭末尾の " と ' を除去（中間にあるクォートは保持）
        case "$trimmed" in
          \"*\" ) trimmed="${trimmed%\"}"; trimmed="${trimmed#\"}";;
        esac
        case "$trimmed" in
          \'*\' ) trimmed="${trimmed%\'}"; trimmed="${trimmed#\'}";;
        esac

        printf '%s\n' "$trimmed" | grep -Eqi '^(torch|torchvision|torchaudio)($|[<=>])' && continue
        printf '%s\n' "$trimmed" | grep -Eqi '^(thop|timm)($|[<=>])' && { echo "$trimmed" >> "$NODEPS_REQ"; continue; }
        echo "$trimmed" >> "$SAFE_REQ"
      done < "$REQS_OVERLAY_PATH"
    fi

    # 1) 依存ありで安全群をインストール
    if [ -s "$SAFE_REQ" ]; then
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" -r "$SAFE_REQ"
    fi

    # 2) 依存なしで NoDeps 群（thop, timm）
    if [ -s "$NODEPS_REQ" ]; then
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" --no-deps -r "$NODEPS_REQ"
    fi

    # 3) 環境変数 PIP_INSTALL の取り込み（トークンからクォートを除去）
    if [ -n "${PIP_INSTALL:-}" ]; then
      _pi_sanitized="$(
        printf '%s\n' "$PIP_INSTALL" \
          | tr ' ' '\n' \
          | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
          | sed 's/^"//;s/"$//' \
          | sed "s/^'//;s/'$//" \
          | grep -Evi '^(torch|torchvision|torchaudio)($|[<=>])' \
          | tr '\n' ' '
      )"
      if [ -n "$_pi_sanitized" ]; then
        # shellcheck disable=SC2086
        python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" $_pi_sanitized
      fi
    fi

    # 4) 念のため overlay から torch 系を物理削除（誤混入の保険）
    rm -rf "${PIP_OVERLAY_DIR}/torch" "${PIP_OVERLAY_DIR}/torchvision" "${PIP_OVERLAY_DIR}/torchaudio" || true
    rm -rf "${PIP_OVERLAY_DIR}"/nvidia_* "${PIP_OVERLAY_DIR}"/nvidia* || true

    # 5) transformers との互換確保：huggingface_hub が 1.x なら 0.44.1 にダウングレード
    python3 - <<'PY'
import sys
try:
    import transformers, huggingface_hub
    from packaging.version import Version
    tv = Version(transformers.__version__)
    hv = Version(huggingface_hub.__version__)
    print(f"[entrypoint] versions: transformers={tv}, huggingface_hub={hv}")
    if hv.major >= 1:
        sys.exit(42)
except Exception as e:
    print(f"[entrypoint] version check note: {e}")
    sys.exit(0)
PY
    ret=$?
    if [ "$ret" -eq 42 ]; then
      echo "[entrypoint] Downgrading huggingface_hub to 0.44.1 for transformers compatibility (<1.0)"
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" "huggingface_hub==0.44.1"
    fi

    echo "${need_hash}" > "${PIP_OVERLAY_DIR}/.overlay_hash"
  else
    echo "[entrypoint] Overlay unchanged; skip pip."
  fi
else
  echo "[entrypoint] No overlay inputs (PIP_INSTALL/requirements.overlay.txt)."
fi

# 可視化: 重要依存が読めるか即確認（バージョンも表示）
python3 - <<'PY'
import importlib.util, os
def status(m): return "OK" if importlib.util.find_spec(m) else "MISSING"
mods = [
  "prefetch_generator","easydict","thop","skimage",
  "timm","einops","transformers","huggingface_hub","safetensors",
  "onnxruntime","cv2","numpy","scipy","PIL","yaml","torchvision"
]
print("[entrypoint] PYTHONPATH head:", os.environ.get("PYTHONPATH","").split(":")[0])
for m in mods:
    print(f"[entrypoint] import {m}:", status(m))
try:
    import transformers, huggingface_hub
    print(f"[entrypoint] versions summary: transformers={transformers.__version__}, huggingface_hub={huggingface_hub.__version__}")
except Exception:
    pass
PY

# ============= 評価スクリプト起動 =============
# -------------- ここから挿入してください --------------
# MAIN_PY が指定されていればそれを実行、未指定なら従来通り eval_unicontrol_waymo.py を実行
MAIN_PY="${MAIN_PY:-/app/eval_unicontrol_waymo.py}"
exec python3 -u "${MAIN_PY}" "$@"
# -------------- ここまで挿入してください --------------