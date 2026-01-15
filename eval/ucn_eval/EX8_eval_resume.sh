#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX8_eval_resume.sh
# EX8_train_infer_eval_wait.sh の run_evaluation() を
# 「評価のみ・オーバーライトなし・キャッシュ完全再利用」で単独実行するスクリプト
# nohup 前提・ログ永続化対応

set -euo pipefail
umask 022

# ============================================================
# 0) ログ設定（nohup対応）
# ============================================================
LOG_DIR="/data/ucn_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/EX8_eval_resume.$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[INFO] LOG_FILE=${LOG_FILE}"

# ============================================================
# 1) 基本設定（EX8 と完全一致）
# ============================================================
EXID="EX8"

EVAL_CACHE_ROOT="/data/ucn_eval_cache_ex8"
EVAL_TB_DIR="${EVAL_CACHE_ROOT}/tensorboard_ex8"

EVAL_ENTRYPOINT="/home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh"
EVAL_SCRIPT="/home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py"

EVAL_DOCKER_IMAGE="ucn-eval"

EVAL_ORIG_ROOT="/home/shogo/coding/datasets/WaymoV2/extracted"
EVAL_GEN_ROOT="/data/coding/datasets/WaymoV2/Pam2_60k_FromUniCKPT"
EVAL_PROMPT_ROOT="/data/syndiff_prompts/prompts_eval_waymo"

# ===== EX8 固定条件 =====
EVAL_TASKS="all"
EVAL_REALITY_METRIC="clip-cmmd"
EVAL_GDINO_MODEL="IDEA-Research/grounding-dino-base"
EVAL_OCR_ENGINE="tesseract"
EVAL_IOU_THR="0.5"
EVAL_ANNOT_MODE="all"
EVAL_ANNOT_LIMIT="24"
EVAL_DRIVABLE_METHODS=(onefroad)
EVAL_AUTO_BATCH_FLAG="--no-auto-batch"

EVAL_PIP_INSTALL="timm yacs prefetch_generator easydict thop scikit-image pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib lpips pytorch-msssim"

EVAL_DET_PROMPTS=(
  car truck bus motorcycle bicycle person pedestrian
  "traffic light" "traffic sign" "stop sign"
  "speed limit sign" "crosswalk sign"
  "construction sign" "traffic cone"
)

HOST_HF_CACHE="/home/shogo/.cache/huggingface"
HOST_TORCH_HUB_DIR="${EVAL_CACHE_ROOT}/torch_hub"

# ★ experiment-note は EX8 と完全一致させる（絶対変更禁止）
EXPERIMENT_NOTE="CustomPamModeling_60KStep from only UniCKPT + Qwen3-VL prompts"

# ============================================================
# 2) 事前チェック（最低限）
# ============================================================
echo "[INFO] date:"
date
echo "[INFO] nvidia-smi:"
nvidia-smi || true
echo

[[ -d "${EVAL_CACHE_ROOT}" ]] || { echo "[FATAL] cache-root missing"; exit 1; }
[[ -d "${EVAL_GEN_ROOT}" ]] || { echo "[FATAL] gen-root missing"; exit 1; }
[[ -f "${EVAL_ENTRYPOINT}" ]] || { echo "[FATAL] entrypoint missing"; exit 1; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "[FATAL] eval script missing"; exit 1; }

# ============================================================
# 3) 評価実行（EX8 評価Bと完全同一）
# ============================================================
echo "[INFO] === START EX8 EVAL RESUME (CACHE REUSE, NO OVERWRITE) ==="

docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e "PIP_OVERLAY_DIR=${EVAL_CACHE_ROOT}/pip-overlay" \
  -e "REQS_OVERLAY_PATH=${EVAL_CACHE_ROOT}/requirements.overlay.txt" \
  -e "PIP_INSTALL=${EVAL_PIP_INSTALL}" \
  -v "${EVAL_ENTRYPOINT}:/app/entrypoint.sh:ro" \
  -v "${EVAL_SCRIPT}:/app/eval_unicontrol_waymo.py:ro" \
  -v "/home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro" \
  -v "/home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro" \
  -v "/data:/data" \
  -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
  -v "${HOST_TORCH_HUB_DIR}:/root/.cache/torch/hub" \
  "${EVAL_DOCKER_IMAGE}" \
  --cache-root "${EVAL_CACHE_ROOT}" \
  --splits training validation testing \
  --camera front \
  --tasks "${EVAL_TASKS}" \
  --reality-metric "${EVAL_REALITY_METRIC}" \
  --gdinomodel "${EVAL_GDINO_MODEL}" \
  --det-prompts "${EVAL_DET_PROMPTS[@]}" \
  --ocr-engine "${EVAL_OCR_ENGINE}" \
  --iou-thr "${EVAL_IOU_THR}" \
  --orig-root "${EVAL_ORIG_ROOT}" \
  --gen-root "${EVAL_GEN_ROOT}" \
  --prompt-root "${EVAL_PROMPT_ROOT}" \
  --annotation-mode "${EVAL_ANNOT_MODE}" \
  --annotate-limit "${EVAL_ANNOT_LIMIT}" \
  --tb --tb-dir "${EVAL_TB_DIR}" \
  --experiment-id "${EXID}" \
  --experiment-note "${EXPERIMENT_NOTE}" \
  --drivable-methods "${EVAL_DRIVABLE_METHODS[@]}" \
  "${EVAL_AUTO_BATCH_FLAG}" \
  --verbose

echo "[INFO] === EX8 EVAL RESUME FINISHED ==="
echo "[INFO] TensorBoard dir: ${EVAL_TB_DIR}"
echo "[INFO] Log file: ${LOG_FILE}"
