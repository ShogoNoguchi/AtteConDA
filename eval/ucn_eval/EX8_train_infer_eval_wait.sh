#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX8_train_infer_eval_wait.sh
set -euo pipefail

umask 022

# ============================================================
# 0) 二重起動防止（同じEX8スクリプトを複数走らせる事故防止）
# ============================================================
LOCK_DIR="/data/ucn_locks"
LOCK_FILE="${LOCK_DIR}/EX8_train_infer_eval.lock"
mkdir -p "${LOCK_DIR}"

exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
  echo "[FATAL] Another EX8 script instance is already running (lock: ${LOCK_FILE})"
  exit 1
fi

# ============================================================
# 1) 基本設定（あなたの資産パスに固定）
# ============================================================
EXID="EX8"

# --- Uni-ControlNet repo ---
WORKDIR="/data/coding/Uni-ControlNet"

# --- Training (PAM2 from uni.ckpt) ---
TRAIN_SCRIPT="${WORKDIR}/src/train/train.py"
TRAIN_CONFIG="${WORKDIR}/configs/local_v15_syndiff_pam2.yaml"
TRAIN_LR="1e-5"
TRAIN_BS="4"
TRAIN_STEPS="60168" #使用するckptは60Kのものだが一応余分に学習しておく。
TRAIN_RESUME_CKPT="${WORKDIR}/ckpt/uni.ckpt"
TRAIN_LOGDIR="${WORKDIR}/logs/Pam2_FromUniCKPT"
TRAIN_LOGGER_VERSION="0"
TRAIN_LOG_FREQ="1000"
TRAIN_CKPT_EVERY="30000"
TRAIN_SD_LOCKED="True"
TRAIN_GPUS="1"

# どの periodic ckpt を推論に使うか（EX8は 60K を使う要件）
INFER_TARGET_STEP="000060000"

# --- Inference (Waymo offline) ---
INFER_SCRIPT="${WORKDIR}/src/tools/infer_waymo_unicontrol_offline.py"
INFER_SPLITS=(training validation testing)
INFER_CAMERA="front"
INFER_UNI_CONFIG="${TRAIN_CONFIG}"
INFER_IMAGE_RES="512"
INFER_NUM_SAMPLES="1"
INFER_DDIM_STEPS="50"
INFER_SCALE="7.5"
INFER_STRENGTH="1.0"
INFER_GLOBAL_STRENGTH="0.0"
INFER_PROMPT_ROOT="/data/syndiff_prompts/prompts_eval_waymo"
INFER_OUT_ROOT="/data/coding/datasets/WaymoV2/Pam2_60k_FromUniCKPT"
INFER_EXPERIMENTS_ROOT="/data/ucn_infer_cache"
INFER_NOTE="CustomPamModeling_60KStep from only UniCKPT + Qwen3-VL prompts"

# --- Evaluation (Docker ucn-eval) ---
EVAL_CACHE_ROOT="/data/ucn_eval_cache_ex8"
EVAL_TB_DIR="${EVAL_CACHE_ROOT}/tensorboard_ex8"
EVAL_ANNOT_OUT="${EVAL_CACHE_ROOT}/viz_ex8"

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

# あなたのEX8要件：drivable は onefroad のみ、auto-batch 無効
EVAL_DRIVABLE_METHODS=(onefroad)
EVAL_AUTO_BATCH_FLAG="--no-auto-batch"

# overlay pip install（あなたのEX5/EX8テンプレに合わせる）
EVAL_PIP_INSTALL="timm yacs prefetch_generator easydict thop scikit-image pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib lpips pytorch-msssim"

# det-prompts（スペース入りを含むので配列で厳密に渡す）
EVAL_DET_PROMPTS=(
  car truck bus motorcycle bicycle person pedestrian
  "traffic light" "traffic sign" "stop sign" "speed limit sign" "crosswalk sign" "construction sign" "traffic cone"
)

# HuggingFace / Torch hub cache（あなたの運用に合わせる）
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
# 3) 事前チェック（不足があると深夜に落ちて最悪なので最初に止める）
# ============================================================
preflight_check() {
  log "=== Preflight check ==="

  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
  command -v python >/dev/null 2>&1 || die "python not found"
  command -v docker >/dev/null 2>&1 || die "docker not found"
  command -v pgrep >/dev/null 2>&1 || die "pgrep not found"
  command -v flock >/dev/null 2>&1 || die "flock not found"

  [[ -d "${WORKDIR}" ]] || die "WORKDIR not found: ${WORKDIR}"
  [[ -f "${TRAIN_SCRIPT}" ]] || die "TRAIN_SCRIPT not found: ${TRAIN_SCRIPT}"
  [[ -f "${TRAIN_CONFIG}" ]] || die "TRAIN_CONFIG not found: ${TRAIN_CONFIG}"
  [[ -f "${TRAIN_RESUME_CKPT}" ]] || die "TRAIN_RESUME_CKPT not found: ${TRAIN_RESUME_CKPT}"
  [[ -f "${INFER_SCRIPT}" ]] || die "INFER_SCRIPT not found: ${INFER_SCRIPT}"

  [[ -f "${EVAL_ENTRYPOINT}" ]] || die "EVAL_ENTRYPOINT not found: ${EVAL_ENTRYPOINT}"
  [[ -f "${EVAL_SCRIPT}" ]] || die "EVAL_SCRIPT not found: ${EVAL_SCRIPT}"
  [[ -d "/home/shogo/coding/datasets/WaymoV2" ]] || die "WaymoV2 dataset root missing: /home/shogo/coding/datasets/WaymoV2"
  [[ -d "/data" ]] || die "/data mount missing"

  # Docker image existence check (optional but safer)
  if ! docker image inspect "${EVAL_DOCKER_IMAGE}" >/dev/null 2>&1; then
    die "Docker image '${EVAL_DOCKER_IMAGE}' not found. Build it or confirm its name."
  fi

  log "OK: preflight check passed"
}

# ============================================================
# 4) “衝突しない待機”ロジック
#    - GPU compute process
#    - EX5系のシェル/推論/評価プロセス
#    - ucn-eval コンテナ稼働
#    - さらに「完全に空の状態が N 秒継続」したら開始
# ============================================================
gpu_compute_lines() {
  # compute-apps が空なら何も出ない
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

is_gpu_compute_busy() {
  local ln
  ln="$(gpu_compute_lines | head -n 1 || true)"
  [[ -n "${ln}" ]]
}

is_ucn_eval_container_running() {
  # 走っているコンテナの image 名を見る（ucn-eval が動いてたらbusy）
  docker ps --format '{{.Image}}' 2>/dev/null | grep -Fxq "${EVAL_DOCKER_IMAGE}"
}

is_ex5_related_process_running() {
  # EX5_infer_then_eval.sh / EX5_all.sh のシェル
  if pgrep -af "/home/shogo/coding/eval/ucn_eval/EX5_infer_then_eval.sh" >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -af "/home/shogo/coding/eval/ucn_eval/EX5_all.sh" >/dev/null 2>&1; then
    return 0
  fi

  # 評価本体（ホスト上 or docker内はホストのpythonとして見える可能性あり）
  if pgrep -af "eval_unicontrol_waymo.py" >/dev/null 2>&1; then
    return 0
  fi

  # 推論本体（EX5が走っているときに確実に引っかける）
  if pgrep -af "infer_waymo_unicontrol_offline.py" >/dev/null 2>&1; then
    return 0
  fi

  return 1
}

is_system_busy() {
  # busy なら 0 を返す（bashの慣習）
  local reasons=()

  if is_gpu_compute_busy; then
    reasons+=("GPU compute process exists")
  fi
  if is_ucn_eval_container_running; then
    reasons+=("Docker container '${EVAL_DOCKER_IMAGE}' is running")
  fi
  if is_ex5_related_process_running; then
    reasons+=("EX5-related process (shell/infer/eval) is running")
  fi

  if ((${#reasons[@]} > 0)); then
    log "[WAIT] BUSY: ${reasons[*]}"
    # 参考としてGPU compute一覧も出す（長すぎない範囲で）
    local gl
    gl="$(gpu_compute_lines || true)"
    if [[ -n "${gl}" ]]; then
      log "[WAIT] nvidia-smi compute apps:"
      # 1行ずつ整形して出す
      while IFS= read -r line; do
        log "  ${line}"
      done <<< "${gl}"
    fi
    return 0
  fi

  return 1
}

wait_until_idle_stable() {
  # “完全に空” が idle_required_sec 継続したら開始
  local idle_required_sec="${1}"
  local interval_sec="${2}"

  log "=== Waiting for safe idle state ==="
  log "Condition: no GPU compute + no EX5-related procs + no '${EVAL_DOCKER_IMAGE}' container"
  log "Stable idle required: ${idle_required_sec}s (check interval: ${interval_sec}s)"

  local idle_sec=0

  while true; do
    if is_system_busy; then
      idle_sec=0
    else
      idle_sec=$((idle_sec + interval_sec))
      log "[WAIT] IDLE: stable ${idle_sec}/${idle_required_sec} sec"
      if (( idle_sec >= idle_required_sec )); then
        log "=== Idle stable achieved. Start EX8 pipeline. ==="
        break
      fi
    fi
    sleep "${interval_sec}"
  done
}

# ============================================================
# 5) 訓練 → 推論 → 評価
# ============================================================
run_training() {
  log "=== (1/3) TRAIN: PAM2 from uni.ckpt (EX8) ==="
  mkdir -p "${TRAIN_LOGDIR}"

  cd "${WORKDIR}"

  local cmd=(
    python -u "${TRAIN_SCRIPT}"
    --config-path "${TRAIN_CONFIG}"
    --learning-rate "${TRAIN_LR}"
    --batch-size "${TRAIN_BS}"
    --training-steps "${TRAIN_STEPS}"
    --resume-path "${TRAIN_RESUME_CKPT}"
    --logdir "${TRAIN_LOGDIR}"
    --logger-version "${TRAIN_LOGGER_VERSION}"
    --log-freq "${TRAIN_LOG_FREQ}"
    --ckpt-every-n-steps "${TRAIN_CKPT_EVERY}"
    --sd-locked "${TRAIN_SD_LOCKED}"
    --gpus "${TRAIN_GPUS}"
  )

  log "[CMD][TRAIN] ${cmd[*]}"
  "${cmd[@]}"

  log "=== TRAIN finished ==="
}

find_infer_ckpt() {
  local ckpt_dir="${TRAIN_LOGDIR}/lightning_logs/version_${TRAIN_LOGGER_VERSION}/checkpoints"
  [[ -d "${ckpt_dir}" ]] || die "Checkpoint dir not found: ${ckpt_dir}"

  # あなたの観測した命名（periodic-stepstep=000060000.ckpt）に合わせ、globで拾う
  local pattern1="${ckpt_dir}/periodic-stepstep=${INFER_TARGET_STEP}.ckpt"
  local pattern2="${ckpt_dir}/periodic-step${INFER_TARGET_STEP}.ckpt"
  local found=""

  if [[ -f "${pattern1}" ]]; then
    found="${pattern1}"
  elif [[ -f "${pattern2}" ]]; then
    found="${pattern2}"
  else
    # 最終手段：部分一致で探索
    local g
    g="$(ls -1 "${ckpt_dir}"/periodic-step*${INFER_TARGET_STEP}*.ckpt 2>/dev/null | head -n 1 || true)"
    if [[ -n "${g}" ]]; then
      found="${g}"
    fi
  fi

  [[ -n "${found}" ]] || die "Target ckpt for step ${INFER_TARGET_STEP} not found in ${ckpt_dir}"

  echo "${found}"
}

run_inference() {
  log "=== (2/3) INFER: Waymo offline inference (EX8) ==="
  mkdir -p "${INFER_OUT_ROOT}"

  local infer_ckpt
  infer_ckpt="$(find_infer_ckpt)"
  log "[INFO] Using inference ckpt: ${infer_ckpt}"

  cd "${WORKDIR}"

  local cmd=(
    python -u "${INFER_SCRIPT}"
    --splits "${INFER_SPLITS[@]}"
    --camera "${INFER_CAMERA}"
    --uni-config "${INFER_UNI_CONFIG}"
    --uni-ckpt "${infer_ckpt}"
    --image-resolution "${INFER_IMAGE_RES}"
    --num-samples "${INFER_NUM_SAMPLES}"
    --ddim-steps "${INFER_DDIM_STEPS}"
    --scale "${INFER_SCALE}"
    --strength "${INFER_STRENGTH}"
    --global-strength "${INFER_GLOBAL_STRENGTH}"
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

run_evaluation() {
  log "=== (3/3) EVAL: Docker ucn-eval (EX8) ==="
  mkdir -p "${EVAL_CACHE_ROOT}" "${HOST_TORCH_HUB_DIR}"
  mkdir -p "${EVAL_CACHE_ROOT}/pip-overlay"
  : > "${EVAL_CACHE_ROOT}/requirements.overlay.txt" || true

  log "[INFO] date:"
  date
  log "[INFO] nvidia-smi:"
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
  log "[DONE] EX8 pipeline done."
  log "[DONE] Train logdir: ${TRAIN_LOGDIR}"
  log "[DONE] Inference out-root: ${INFER_OUT_ROOT}"
  log "[DONE] Eval cache-root: ${EVAL_CACHE_ROOT}"
  log "[DONE] TensorBoard dir: ${EVAL_TB_DIR}"
}

# ============================================================
# 6) main
# ============================================================
main() {
  preflight_check

  # いまEX5が走っているので、衝突ゼロを優先して待つ
  # “空の状態が 180 秒継続” を要求（評価中の瞬間的GPU解放や、推論→評価のつなぎ時間を吸収）
  wait_until_idle_stable 180 20

  run_training
  run_inference
  run_evaluation
}

main "$@"
