#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX9_infer_eval_wait.sh
set -euo pipefail

umask 022

# ============================================================
# 0) 二重起動防止（EX9 専用 lock）
# ============================================================
LOCK_DIR="/data/ucn_locks"
LOCK_FILE="${LOCK_DIR}/EX9_infer_eval.lock"
mkdir -p "${LOCK_DIR}"

exec 201>"${LOCK_FILE}"
if ! flock -n 201; then
  echo "[FATAL] Another EX9 script instance is already running (lock: ${LOCK_FILE})"
  exit 1
fi

# ============================================================
# 1) 基本設定（EX9: DGInStyle）
# ============================================================
EXID="EX9"

# --- DGInStyle 推論 ---
DG_REPO="/data/coding/DGInStyle"
INFER_SCRIPT="${DG_REPO}/src/tools/infer_waymo_unicontrol_offline.py"

INFER_SPLITS=(training validation testing)
INFER_CAMERA="front"
INFER_DG_CONFIG="/data/coding/DGInStyle/configs/dginstyle_sd15_semseg.yaml"
INFER_DG_CKPT="yurujaja/DGInStyle"

INFER_IMAGE_RES="512"
INFER_NUM_SAMPLES="1"
INFER_DDIM_STEPS="50"
INFER_SCALE="7.5"
INFER_STRENGTH="1.0"
INFER_ETA="0.0"

INFER_PROMPT_ROOT="/data/syndiff_prompts/prompts_eval_waymo"
INFER_OUT_ROOT="/data/coding/datasets/WaymoV2/DGInstyle_Pure"
INFER_EXPERIMENTS_ROOT="/data/ucn_infer_cache"
INFER_NOTE="DGInStyle + Qwen3-VL prompts"

# --- Evaluation ---
EVAL_CACHE_ROOT="/data/ucn_eval_cache_ex9"
EVAL_TB_DIR="${EVAL_CACHE_ROOT}/tensorboard_ex9"
EVAL_ANNOT_OUT="${EVAL_CACHE_ROOT}/viz_ex9"

EVAL_ENTRYPOINT="/home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh"
EVAL_SCRIPT="/home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py"

EVAL_DOCKER_IMAGE="ucn-eval"

EVAL_ORIG_ROOT="/home/shogo/coding/datasets/WaymoV2/extracted"
EVAL_GEN_ROOT="${INFER_OUT_ROOT}"
EVAL_PROMPT_ROOT="${INFER_PROMPT_ROOT}"

EVAL_REALITY_METRIC="clip-cmmd"
EVAL_GDINO_MODEL="IDEA-Research/grounding-dino-base"
EVAL_OCR_ENGINE="tesseract"
EVAL_IOU_THR="0.5"
EVAL_TASKS="all"
EVAL_ANNOT_MODE="all"
EVAL_ANNOT_LIMIT="24"

# drivable は onefroad のみ（EX8 と同一思想）
EVAL_DRIVABLE_METHODS=(onefroad)
EVAL_AUTO_BATCH_FLAG="--no-auto-batch"

EVAL_PIP_INSTALL="timm yacs prefetch_generator easydict thop scikit-image pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib lpips pytorch-msssim"

EVAL_DET_PROMPTS=(
  car truck bus motorcycle bicycle person pedestrian
  "traffic light" "traffic sign" "stop sign" "speed limit sign" "crosswalk sign" "construction sign" "traffic cone"
)

HOST_HF_CACHE="/home/shogo/.cache/huggingface"
HOST_TORCH_HUB_DIR="${EVAL_CACHE_ROOT}/torch_hub"

# ============================================================
# 2) ログユーティリティ
# ============================================================
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  log "[FATAL] $*"
  exit 1
}

# ============================================================
# 3) BUSY 判定ロジック（EX8 完全互換 + EX9 拡張）
# ============================================================
gpu_compute_lines() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

is_gpu_compute_busy() {
  [[ -n "$(gpu_compute_lines | head -n 1 || true)" ]]
}

is_ucn_eval_container_running() {
  docker ps --format '{{.Image}}' 2>/dev/null | grep -Fxq "${EVAL_DOCKER_IMAGE}"
}

is_ucn_related_process_running() {
  if pgrep -af "infer_waymo_unicontrol_offline.py" >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -af "eval_unicontrol_waymo.py" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

is_ex8_running() {
  if [[ -f "/data/ucn_locks/EX8_train_infer_eval.lock" ]]; then
    if lsof "/data/ucn_locks/EX8_train_infer_eval.lock" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

is_system_busy() {
  local reasons=()

  is_gpu_compute_busy && reasons+=("GPU compute exists")
  is_ucn_eval_container_running && reasons+=("ucn-eval docker running")
  is_ucn_related_process_running && reasons+=("infer/eval python running")
  is_ex8_running && reasons+=("EX8 lock still held")

  if ((${#reasons[@]} > 0)); then
    log "[WAIT] BUSY: ${reasons[*]}"
    local gl
    gl="$(gpu_compute_lines || true)"
    if [[ -n "${gl}" ]]; then
      log "[WAIT] nvidia-smi compute apps:"
      while IFS= read -r line; do
        log "  ${line}"
      done <<< "${gl}"
    fi
    return 0
  fi

  return 1
}

wait_until_idle_stable() {
  local idle_required_sec="${1}"
  local interval_sec="${2}"

  log "=== Waiting for safe idle state (EX9) ==="
  log "Require idle for ${idle_required_sec}s (interval=${interval_sec}s)"

  local idle_sec=0
  while true; do
    if is_system_busy; then
      idle_sec=0
    else
      idle_sec=$((idle_sec + interval_sec))
      log "[WAIT] IDLE: stable ${idle_sec}/${idle_required_sec}s"
      if (( idle_sec >= idle_required_sec )); then
        log "=== Idle stable achieved. Start EX9. ==="
        break
      fi
    fi
    sleep "${interval_sec}"
  done
}

# ============================================================
# 4) 推論（DGInStyle）
# ============================================================
run_inference() {
  log "=== (1/2) INFER: DGInStyle Waymo offline (EX9) ==="
  mkdir -p "${INFER_OUT_ROOT}"

  cd "${DG_REPO}"

  local cmd=(
    python -u "${INFER_SCRIPT}"
    --splits "${INFER_SPLITS[@]}"
    --camera "${INFER_CAMERA}"
    --DG-config "${INFER_DG_CONFIG}"
    --DG-ckpt "${INFER_DG_CKPT}"
    --image-resolution "${INFER_IMAGE_RES}"
    --num-samples "${INFER_NUM_SAMPLES}"
    --ddim-steps "${INFER_DDIM_STEPS}"
    --scale "${INFER_SCALE}"
    --strength "${INFER_STRENGTH}"
    --eta "${INFER_ETA}"
    --prompt-root "${INFER_PROMPT_ROOT}"
    --out-root "${INFER_OUT_ROOT}"
    --experiments-root "${INFER_EXPERIMENTS_ROOT}"
    --experiment-id "${EXID}"
    --experiment-note "${INFER_NOTE}"
    --verbose
  )

  log "[CMD][INFER] ${cmd[*]}"
  "${cmd[@]}"

  log "=== INFER finished ==="
}

# ============================================================
# 5) 評価
# ============================================================
run_evaluation() {
  log "=== (2/2) EVAL: Docker ucn-eval (EX9) ==="
  mkdir -p "${EVAL_CACHE_ROOT}" "${HOST_TORCH_HUB_DIR}"
  mkdir -p "${EVAL_CACHE_ROOT}/pip-overlay"
  : > "${EVAL_CACHE_ROOT}/requirements.overlay.txt" || true

  date
  nvidia-smi || true
  echo

  local cmd=(
    docker run --rm --gpus all
    -e MPLBACKEND=Agg
    -e "PIP_OVERLAY_DIR=${EVAL_CACHE_ROOT}/pip-overlay"
    -e "REQS_OVERLAY_PATH=${EVAL_CACHE_ROOT}/requirements.overlay.txt"
    -e "PIP_INSTALL=${EVAL_PIP_INSTALL}"
    -v "${EVAL_ENTRYPOINT}:/app/entrypoint.sh:ro"
    -v "${EVAL_SCRIPT}:/app/eval_unicontrol_waymo.py:ro"
    -v "/home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro"
    -v "/home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro"
    -v "/data:/data"
    -v "${HOST_HF_CACHE}:/root/.cache/huggingface"
    -v "${HOST_TORCH_HUB_DIR}:/root/.cache/torch/hub"
    "${EVAL_DOCKER_IMAGE}"
    --cache-root "${EVAL_CACHE_ROOT}"
    --splits training validation testing
    --camera front
    --tasks "${EVAL_TASKS}"
    --reality-metric "${EVAL_REALITY_METRIC}"
    --gdinomodel "${EVAL_GDINO_MODEL}"
    --det-prompts "${EVAL_DET_PROMPTS[@]}"
    --ocr-engine "${EVAL_OCR_ENGINE}"
    --iou-thr "${EVAL_IOU_THR}"
    --orig-root "${EVAL_ORIG_ROOT}"
    --gen-root "${EVAL_GEN_ROOT}"
    --prompt-root "${EVAL_PROMPT_ROOT}"
    --annotation-mode "${EVAL_ANNOT_MODE}"
    --annotate-limit "${EVAL_ANNOT_LIMIT}"
    --annotate-out "${EVAL_ANNOT_OUT}"
    --tb --tb-dir "${EVAL_TB_DIR}"
    --experiment-id "${EXID}"
    --experiment-note "${INFER_NOTE}"
    --drivable-methods "${EVAL_DRIVABLE_METHODS[@]}"
    "${EVAL_AUTO_BATCH_FLAG}"
    --verbose
  )

  log "[CMD][EVAL] ${cmd[*]}"
  "${cmd[@]}"

  log "=== EVAL finished ==="
}

# ============================================================
# 6) main
# ============================================================
main() {
  log "=== EX9 DGInStyle super-wait pipeline start ==="
  wait_until_idle_stable 300 20   # ★5分安定待機（EX8 完全終了保証）
  run_inference
  run_evaluation
  log "[DONE] EX9 pipeline finished successfully."
}

main "$@"
