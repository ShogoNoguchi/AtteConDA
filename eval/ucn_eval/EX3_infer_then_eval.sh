#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX3_infer_then_eval.sh
set -euo pipefail

# ========== 1) 推論（EX3: 公開base ckpt + Qwen CSV） ==========
python -u /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py \
  --splits training validation testing \
  --camera front \
  --uni-config /data/coding/Uni-ControlNet/configs/uni_v15.yaml \
  --uni-ckpt /data/coding/Uni-ControlNet/ckpt/uni.ckpt \
  --image-resolution 512 \
  --num-samples 1 \
  --ddim-steps 50 \
  --scale 7.5 \
  --strength 1.0 \
  --global-strength 0.0 \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --out-root /data/coding/datasets/WaymoV2/Ucn_fromBasePublicCkpt \
  --experiments-root /data/ucn_infer_cache \
  --experiment-id EX3 \
  --experiment-note "Uni public base ckpt + Qwen3-VL prompts (ablation of finetune)" \
  --overwrite --verbose

# ========== 2) 評価（EX3_all.sh をそのまま呼ぶ） ==========
bash /home/shogo/coding/eval/ucn_eval/EX3_all.sh
