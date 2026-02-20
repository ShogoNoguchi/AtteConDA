# CUDA 12.8 ランタイム（NVIDIA公式）
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 基本ツール（最小構成）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git wget curl ca-certificates \
      libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# pip 更新
RUN python3 -m pip install -U pip

WORKDIR /app

# ベース依存（torchなしの最小セット）
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# エントリポイント（pipオーバーレイ＋実行ラッパ）
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 将来の bind mount で使う絶対パスをコンテナ内に確保（空ディレクトリ作成）
# ※ スクリプト本体は毎回ホストから :ro でマウントする前提
RUN mkdir -p /home/shogo/coding/tools/promptgen

# 既定の環境変数（オーバーレイのデフォルト）
ENV PIP_OVERLAY_DIR=/opt/pip-overlay \
    REQS_OVERLAY_PATH=/opt/requirements.overlay.txt \
    PIP_INSTALL="" \
    PIP_TORCH_PACKAGES="" \
    PIP_TORCH_INDEX=https://download.pytorch.org/whl/cu128

# 既定のOllama URL（entrypointで参照）。Linuxでは --add-host or --network=host が必要
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

ENTRYPOINT ["/app/entrypoint.sh"]
