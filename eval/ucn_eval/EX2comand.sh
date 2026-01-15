#!/usr/bin/env bash
set -euo pipefail

# EX2: finetune 後 Uni-ControlNet + Qwen3-VL プロンプトによる F(X)' を評価
#  - キャッシュ : /data/ucn_eval_cache_ex2
#  - 元画像 X   : /home/shogo/coding/datasets/WaymoV2/extracted
#  - 生成画像 F(X)': /data/coding/datasets/WaymoV2/Ucn_byPure_Finetune
#  - プロンプト : /data/syndiff_prompts/prompts_eval_waymo/waymo_{split}.csv

# ===== 0. キャッシュディレクトリ =====
mkdir -p /data/ucn_eval_cache_ex2 \
         /data/ucn_eval_cache_ex2/torch_hub

# ===== 1. GPU 状態ログ =====
date
nvidia-smi || true
echo
#yolop-roi-filterはDINOの評価が壊れるので使用しない。
# ===== 2. Docker 実行 =====
docker run --rm --gpus all \
    -e MPLBACKEND=Agg \
    -e PIP_OVERLAY_DIR=/data/ucn_eval_cache_ex2/pip-overlay \
    -e REQS_OVERLAY_PATH=/data/ucn_eval_cache_ex2/requirements.overlay.txt \
    -e 'PIP_INSTALL=timm yacs prefetch_generator easydict thop scikit-image pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib lpips pytorch-msssim' \
    -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
    -v /home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py:/app/eval_unicontrol_waymo.py:ro \
    -v /home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro \
    -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
    -v /data:/data \
    -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
    -v /data/ucn_eval_cache_ex2/torch_hub:/root/.cache/torch/hub \
    ucn-eval \
    --cache-root /data/ucn_eval_cache_ex2 \
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
    --gen-root  /data/coding/datasets/WaymoV2/Ucn_byPure_Finetune \
    --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
    --annotation-mode all \
    --annotate-limit 24 \
    --annotate-out /data/ucn_eval_cache_ex2/viz_ex2 \
    --tb --tb-dir /data/ucn_eval_cache_ex2/tensorboard_ex2 \
    --experiment-id EX2 \
    --experiment-note "Uni-ControlNet finetuned (version_4) + Qwen3-VL prompts (waymo_{split}.csv)" \
    --no-auto-batch \
    --verbose "$@"
