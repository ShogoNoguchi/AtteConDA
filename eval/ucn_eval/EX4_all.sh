#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX4_all.sh
set -euo pipefail

# ==========  ==========
# 生成画像: 60Kckpt + Qwen プロンプト → /data/coding/datasets/WaymoV2/Ucn_byPure_Finetune_60Kstep
# 評価キャッシュ: /data/ucn_eval_cache_ex4

# 0) キャッシュ
mkdir -p /data/ucn_eval_cache_ex4 /data/ucn_eval_cache_ex4/torch_hub

# 1) 参考ログ
date
nvidia-smi || true
echo

# 2) Docker 実行（YOLOPは使わない / DrivableはOneFormer-roadのみ）
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e PIP_OVERLAY_DIR=/data/ucn_eval_cache_ex4/pip-overlay \
  -e REQS_OVERLAY_PATH=/data/ucn_eval_cache_ex4/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs prefetch_generator easydict thop scikit-image pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib lpips pytorch-msssim' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py:/app/eval_unicontrol_waymo.py:ro \
  -v /home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /data:/data \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data/ucn_eval_cache_ex4/torch_hub:/root/.cache/torch/hub \
  ucn-eval \
  --cache-root /data/ucn_eval_cache_ex4 \
  --splits training validation testing \
  --camera front \
  --tasks all \
  --reality-metric clip-cmmd \
  --gdinomodel IDEA-Research/grounding-dino-base \
  --det-prompts car truck bus motorcycle bicycle person pedestrian "traffic light" "traffic sign" "stop sign" "speed limit sign" "crosswalk sign" "construction sign" "traffic cone" \
  --ocr-engine tesseract \
  --iou-thr 0.5 \
  --orig-root /home/shogo/coding/datasets/WaymoV2/extracted \
  --gen-root  /data/coding/datasets/WaymoV2/Ucn_byPure_Finetune_60Kstep \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --annotation-mode all \
  --annotate-limit 24 \
  --annotate-out /data/ucn_eval_cache_ex4/viz_ex4 \
  --tb --tb-dir /data/ucn_eval_cache_ex4/tensorboard_ex4 \
  --experiment-id EX4 \
  --experiment-note "Pure 60Kstep ckpt + Qwen3-VL prompts (ablation of Training time)" \
  --drivable-methods onefroad \
  --no-auto-batch \
  --verbose "$@"
