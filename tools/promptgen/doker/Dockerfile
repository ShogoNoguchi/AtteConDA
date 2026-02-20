# /home/shogo/coding/tools/promptgen/docker/Dockerfile
# CUDA 12.8 ランタイム（NVIDIA公式）
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# ===== 基本環境変数 =====
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# ===== 必須パッケージ（最小限） =====
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      git wget curl ca-certificates \
      libgl1 libglib2.0-0 \
      tini && \
    rm -rf /var/lib/apt/lists/*

# ===== 非rootユーザ作成（ホストUID/GIDと合わせると出力権限が楽）=====
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -g ${HOST_GID} appuser && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash appuser

# ===== 作業ディレクトリ =====
WORKDIR /app

# ===== Python venv を用意（ベース依存はこの中に固定導入）=====
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install -U pip

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ===== ベース依存（Torchは入れない：必要時だけ起動時に安全導入）=====
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# ===== エントリポイント配置 =====
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ===== ヘルスチェック（Ollama 接続確認：任意）=====
#   OLLAMA_BASE_URL が未設定なら healthcheck は no-op
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD /bin/bash -lc '[ -z "$OLLAMA_BASE_URL" ] || curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null'

# ===== デフォルトは非rootで実行 =====
USER appuser

# ===== init（tini）でPID1のゾンビ回収 & ENTRYPOINT =====
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
# CMD は空：引数は docker run の -- オプション以降で渡す
