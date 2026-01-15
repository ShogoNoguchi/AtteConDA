#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX5_infer_then_eval.sh
set -euo pipefail

# ========== 1) 推論（EX5） ==========
python -u /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py \
  --splits training validation testing \
  --camera front \
  --uni-config /data/coding/Uni-ControlNet/configs/uni_v15.yaml \
  --uni-ckpt /data/coding/ckpts/version_4/checkpoints/epoch=2-step=90000.ckpt \
  --image-resolution 512 \
  --num-samples 1 \
  --ddim-steps 50 \
  --scale 7.5 \
  --strength 1.0 \
  --global-strength 0.0 \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --out-root /data/coding/datasets/WaymoV2/Ucn_byPure_Finetune_90Kstep \
  --experiments-root /data/ucn_infer_cache \
  --experiment-id EX5 \
  --experiment-note "Pure 90Kstep ckpt + Qwen3-VL prompts (ablation of Training time)" \
  --verbose

# ========== 2) 評価（EX5_all.sh をそのまま呼ぶ） ==========
bash /home/shogo/coding/eval/ucn_eval/EX5_all.sh
