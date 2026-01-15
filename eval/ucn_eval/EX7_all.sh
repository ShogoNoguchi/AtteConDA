#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX7_all.sh
set -euo pipefail

EXID="EX7"
EXNOTE="Perform finetune 30Ksteps without using uni.ckpt, which is officially distributed by UniControlNet. Abration is used to measure the power of prior learning."

GEN_ROOT="/data/coding/datasets/WaymoV2/Ucn_byPure_Finetune30K_FromNOPretrain"
CACHE_ROOT="/data/ucn_eval_cache_ex7"

mkdir -p "${CACHE_ROOT}" "${CACHE_ROOT}/torch_hub"

date
nvidia-smi || true
echo

docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e PIP_OVERLAY_DIR="${CACHE_ROOT}/pip-overlay" \
  -e REQS_OVERLAY_PATH="${CACHE_ROOT}/requirements.overlay.txt" \
  -e 'PIP_INSTALL=timm yacs prefetch_generator easydict thop scikit-image pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib lpips pytorch-msssim' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py:/app/eval_unicontrol_waymo.py:ro \
  -v /home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /data:/data \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v "${CACHE_ROOT}/torch_hub:/root/.cache/torch/hub" \
  ucn-eval \
  --cache-root "${CACHE_ROOT}" \
  --splits training validation testing \
  --camera front \
  --tasks all \
  --reality-metric clip-cmmd \
  --gdinomodel IDEA-Research/grounding-dino-base \
  --det-prompts car truck bus motorcycle bicycle person pedestrian "traffic light" "traffic sign" "stop sign" "speed limit sign" "crosswalk sign" "construction sign" "traffic cone" \
  --ocr-engine tesseract \
  --iou-thr 0.5 \
  --orig-root /home/shogo/coding/datasets/WaymoV2/extracted \
  --gen-root "${GEN_ROOT}" \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --annotation-mode all \
  --annotate-limit 24 \
  --annotate-out "${CACHE_ROOT}/viz_ex7" \
  --tb --tb-dir "${CACHE_ROOT}/tensorboard_ex7" \
  --experiment-id "${EXID}" \
  --experiment-note "${EXNOTE}" \
  --drivable-methods onefroad \
  --no-auto-batch \
  --verbose "$@"
