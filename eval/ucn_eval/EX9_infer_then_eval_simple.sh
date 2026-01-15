#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX9_infer_then_eval_simple.sh
# EX9: DGInStyle 推論 → 評価（WAITなし・人間監視前提）

set -euo pipefail
umask 022

# ============================================================
# ログ（1ファイル・nohup対応）
# ============================================================
LOG_DIR="/data/ucn_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/EX9_infer_then_eval.$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[INFO] LOG_FILE=${LOG_FILE}"
date
nvidia-smi || true
echo

# ============================================================
# (1) 推論（EX9: DGInStyle）
# ============================================================
echo "[INFO] === START EX9 INFERENCE (DGInStyle) ==="

python -u /data/coding/DGInStyle/src/tools/infer_waymo_unicontrol_offline.py \
  --splits training validation testing \
  --camera front \
  --DG-config /data/coding/DGInStyle/configs/dginstyle_sd15_semseg.yaml \
  --DG-ckpt yurujaja/DGInStyle \
  --image-resolution 512 \
  --num-samples 1 \
  --ddim-steps 50 \
  --scale 7.5 \
  --strength 1.0 \
  --eta 0.0 \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --out-root /data/coding/datasets/WaymoV2/DGInstyle_Pure \
  --experiments-root /data/ucn_infer_cache \
  --experiment-id EX9 \
  --experiment-note "DGInStyle (semseg-only) + Qwen3-VL prompts" \
  --verbose

echo "[INFO] === EX9 INFERENCE FINISHED ==="
echo

# ============================================================
# (2) 評価（EX9）
# ============================================================
echo "[INFO] === START EX9 EVALUATION ==="

bash /home/shogo/coding/eval/ucn_eval/EX9_all.sh

echo "[INFO] === EX9 EVALUATION FINISHED ==="
echo "[DONE] EX9 infer → eval pipeline completed."
