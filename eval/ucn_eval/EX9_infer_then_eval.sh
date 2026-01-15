#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX9_infer_then_eval.sh
set -euo pipefail

# ========== 1) 推論（EX9: DGInStyle） ==========
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

# ========== 2) 評価（EX9） ==========
bash /home/shogo/coding/eval/ucn_eval/EX9_all.sh
