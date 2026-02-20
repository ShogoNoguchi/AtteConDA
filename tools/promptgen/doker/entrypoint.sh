#!/usr/bin/env bash
# /home/shogo/coding/tools/promptgen/docker/entrypoint.sh
set -Eeuo pipefail

# ===== 変数（環境変数で上書き可能）=====
: "${PYTHON:=python}"
: "${SCRIPT_ABS:=/home/shogo/coding/tools/promptgen/generate_waymo_prompts_gptoss_v3.1.py}"
: "${OLLAMA_BASE_URL:=}"        # 例: http://host.docker.internal:11434 あるいは http://127.0.0.1:11434 (hostネット)
: "${OLLAMA_WAIT:=1}"           # 1なら起動前に疎通確認
: "${PIP_INSTALL:=}"            # 任意の追加pipパッケージ（空でスキップ）
: "${REQS_OVERLAY_PATH:=}"      # 追加requirements.txtへのパス（空でスキップ）
: "${TORCH_CU128:=0}"           # 1なら torch/cu128 を起動時に導入（安全）
: "${TORCH_VERSION:=2.7.0}"
: "${TORCHVISION_VERSION:=0.22.0}"
: "${TORCHAUDIO_VERSION:=2.7.0}"

# venv は Dockerfile で /opt/venv に作ってある
export PATH="/opt/venv/bin:${PATH}"

echo "[INFO] Python: $(${PYTHON} -V)"
echo "[INFO] Using script: ${SCRIPT_ABS}"

# ===== 追加パッケージ（オーバーレイ）=====
# ここでは「既存の site-packages を壊さずに」venv に安全導入
if [[ -n "${REQS_OVERLAY_PATH}" && -f "${REQS_OVERLAY_PATH}" ]]; then
  echo "[INFO] Installing overlay requirements: ${REQS_OVERLAY_PATH}"
  ${PYTHON} -m pip install -r "${REQS_OVERLAY_PATH}"
fi

if [[ -n "${PIP_INSTALL}" ]]; then
  echo "[INFO] Installing extra packages: ${PIP_INSTALL}"
  ${PYTHON} -m pip install ${PIP_INSTALL}
fi

# ===== 任意: Torch/cu128 の安全導入（必要時のみ）=====
if [[ "${TORCH_CU128}" == "1" ]]; then
  echo "[INFO] Installing torch/cu128 pinned versions into venv..."
  ${PYTHON} -m pip install \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      torchaudio==${TORCHAUDIO_VERSION} \
      --index-url https://download.pytorch.org/whl/cu128
fi

# ===== 任意: Ollama の疎通待機 =====
if [[ -n "${OLLAMA_BASE_URL}" && "${OLLAMA_WAIT}" == "1" ]]; then
  echo "[INFO] Waiting for Ollama at: ${OLLAMA_BASE_URL}"
  for i in $(seq 1 60); do
    if curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
      echo "[INFO] Ollama reachable."
      break
    fi
    echo "[INFO] Ollama not ready yet... (${i}/60)"
    sleep 2
  done
fi

# ===== 絶対パスのスクリプト起動 =====
# 使い方:
#   docker run ... <この後の全引数> はスクリプトの引数として渡される
# 例:
#   docker run ... -- --splits training validation testing --camera front ...
if [[ ! -f "${SCRIPT_ABS}" ]]; then
  echo "[ERROR] Script not found at ${SCRIPT_ABS}"
  exit 1
fi

echo "[INFO] Launching: ${PYTHON} ${SCRIPT_ABS} $*"
exec ${PYTHON} "${SCRIPT_ABS}" "$@"
