#!/usr/bin/env bash
set -euo pipefail

log() { echo "[entrypoint] $*"; }

# ===== pip オーバーレイの準備（ベース依存を壊さない）=====
if [ -n "${PIP_OVERLAY_DIR:-}" ]; then
  mkdir -p "${PIP_OVERLAY_DIR}"
  export PYTHONPATH="${PIP_OVERLAY_DIR}:${PYTHONPATH-}"
  log "PYTHONPATH (overlay first): ${PYTHONPATH}"
fi

# 追加 requirements（ファイル）をオーバーレイへ
if [ -n "${REQS_OVERLAY_PATH:-}" ] && [ -f "${REQS_OVERLAY_PATH}" ]; then
  log "Installing overlay requirements from: ${REQS_OVERLAY_PATH}"
  python3 -m pip install --target "${PIP_OVERLAY_DIR}" -r "${REQS_OVERLAY_PATH}"
fi

# 追加パッケージ（スペース区切り）をオーバーレイへ
if [ -n "${PIP_INSTALL:-}" ]; then
  log "Installing overlay packages: ${PIP_INSTALL}"
  python3 -m pip install --target "${PIP_OVERLAY_DIR}" ${PIP_INSTALL}
fi

# 将来の torch/cu128 追加（必要時のみ、ベースを汚さない）
# 例: -e PIP_TORCH_PACKAGES="torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0"
#     -e PIP_TORCH_INDEX="https://download.pytorch.org/whl/cu128"
if [ -n "${PIP_TORCH_PACKAGES:-}" ]; then
  log "Installing torch (cu128) overlay from ${PIP_TORCH_INDEX}: ${PIP_TORCH_PACKAGES}"
  python3 -m pip install --target "${PIP_OVERLAY_DIR}" --index-url "${PIP_TORCH_INDEX}" ${PIP_TORCH_PACKAGES}
fi

# ===== Ollama URL 既定 =====
DEFAULT_OLLAMA_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"

# 対象スクリプトの絶対パス（毎回 bind mount で読み込む前提）
TARGET_SCRIPT="/home/shogo/coding/tools/promptgen/generate_waymo_prompts_gptoss_v3.1.py"

if [ ! -f "${TARGET_SCRIPT}" ]; then
  log "ERROR: script not found at ${TARGET_SCRIPT}"
  log "Mount it with: -v /home/shogo/coding/tools/promptgen/generate_waymo_prompts_gptoss_v3.1.py:${TARGET_SCRIPT}:ro"
  exit 1
fi

# --ollama-base-url がユーザ引数に含まれているかチェック
pass_args=("$@")
has_ollama_arg="no"
for a in "${pass_args[@]:-}"; do
  if [[ "$a" == "--ollama-base-url" ]] || [[ "$a" == --ollama-base-url=* ]]; then
    has_ollama_arg="yes"; break
  fi
done

if [ "${has_ollama_arg}" = "no" ]; then
  log "No --ollama-base-url found in args; injecting default: ${DEFAULT_OLLAMA_URL}"
  exec python3 "${TARGET_SCRIPT}" --ollama-base-url "${DEFAULT_OLLAMA_URL}" "$@"
else
  exec python3 "${TARGET_SCRIPT}" "$@"
fi
