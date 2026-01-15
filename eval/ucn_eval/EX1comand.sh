#!/usr/bin/env bash

set -e

# ★EX1 用キャッシュディレクトリを SSD(/data) 側に作成
mkdir -p /data/ucn_eval_cache_ex1 /data/ucn_eval_cache_ex1/torch_hub

#EX1は既にキャッシュ済なのでno-auto-batchを使用。
#yolop-roi-filterはDINOの評価が壊れるので使用しない。
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e PIP_OVERLAY_DIR=/data/ucn_eval_cache_ex1/pip-overlay \
  -e REQS_OVERLAY_PATH=/data/ucn_eval_cache_ex1/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs prefetch_generator pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib lpips pytorch-msssim' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py:/app/eval_unicontrol_waymo.py:ro \
  -v /home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /data:/data \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data/ucn_eval_cache_ex1/torch_hub:/root/.cache/torch/hub \
  ucn-eval \
  --cache-root /data/ucn_eval_cache_ex1 \
  --splits training validation testing \
  --camera front \
  --tasks all \
  --reality-metric clip-cmmd \
  --gdinomodel IDEA-Research/grounding-dino-base \
  --det-prompts car truck bus motorcycle bicycle person pedestrian "traffic light" "traffic sign" "stop sign" "speed limit sign" "crosswalk sign" "construction sign" "traffic cone" \
  --use-yolop \
  --ocr-engine tesseract \
  --iou-thr 0.5 \
  --orig-root /home/shogo/coding/datasets/WaymoV2/extracted \
  --gen-root /home/shogo/coding/datasets/WaymoV2/UniControlNet_offline \
  --prompt-root /home/shogo/coding/datasets/WaymoV2/Prompts_gptoss \
  --annotation-mode all \
  --annotate-limit 24 \
  --annotate-out /data/ucn_eval_cache_ex1/viz_ex1 \
  --tb --tb-dir /data/ucn_eval_cache_ex1/tensorboard_ex1 \
  --experiment-id EX1 \
  --experiment-note "UniControlNet_offline + GPT-OSS prompts (legacy, before finetune)" \
  --no-auto-batch \
  --verbose
