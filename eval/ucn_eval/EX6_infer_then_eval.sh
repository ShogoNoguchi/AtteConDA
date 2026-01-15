#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX6_infer_then_eval.sh
set -euo pipefail

# ========== 1) 推論（EX6） ==========
python -u /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py \
  --splits training validation testing \
  --camera front \
  --uni-config /data/coding/Uni-ControlNet/configs/local_v15_syndiff_pam2.yaml \
  --uni-ckpt /data/coding/ckpts/version_4/Pam/periodic-stepstep=000060000.ckpt \
  --image-resolution 512 \
  --num-samples 1 \
  --ddim-steps 50 \
  --scale 7.5 \
  --strength 1.0 \
  --global-strength 0.0 \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --out-root /data/coding/datasets/WaymoV2/Ucn_byCustomPamModeling_Finetune_60KStep_PretrainIsPureFine60K \
  --experiments-root /data/ucn_infer_cache \
  --experiment-id EX6 \
  --experiment-note "CustomPamModeling_Finetune_60KStep_PretrainIsPureFine60K + Qwen3-VL prompts" \
  --verbose

# ========== 2) 評価（EX6_all.sh をそのまま呼ぶ） ==========
bash /home/shogo/coding/eval/ucn_eval/EX6_all.sh
