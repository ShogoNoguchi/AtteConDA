#!/usr/bin/env bash
set -e

OUT_DIR="/data/ucn_eval_cache_compare/EX1_vs_EX2"
mkdir -p "${OUT_DIR}"

docker run --rm --gpus all \
  --entrypoint python \
  -e MPLBACKEND=Agg \
  -v /home/shogo/coding/eval/ucn_eval/compare_ucn_experiments.py:/app/compare_ucn_experiments.py:ro \
  -v /data:/data \
  ucn-eval \
  /app/compare_ucn_experiments.py \
    --Comparison_between_EX_and_report_for_researchpresentation_Mode \
    --exp-a-json /data/ucn_eval_cache_ex1/experiments/EX1.eval.json \
    --exp-b-json /data/ucn_eval_cache_ex2/experiments/EX2.eval.json \
    --exp-a-id EX1 \
    --exp-b-id EX2 \
    --out-dir "${OUT_DIR}"
