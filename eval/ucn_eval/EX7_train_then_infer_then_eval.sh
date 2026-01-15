#!/usr/bin/env bash
# /home/shogo/coding/eval/ucn_eval/EX7_train_then_infer_then_eval.sh
set -euo pipefail

date
nvidia-smi || true
echo

EXID="EX7"
EXNOTE="Perform finetune 30Ksteps without using uni.ckpt, which is officially distributed by UniControlNet. Abration is used to measure the power of prior learning."

UCN_ROOT="/data/coding/Uni-ControlNet"
SD_CKPT="${UCN_ROOT}/ckpt/v1-5-pruned.ckpt"

CFG_INIT_LOCAL="${UCN_ROOT}/configs/local_v15.yaml"
CFG_INIT_GLOBAL="${UCN_ROOT}/configs/global_v15.yaml"
CFG_INIT_UNI="${UCN_ROOT}/configs/uni_v15.yaml"
CFG_TRAIN="${UCN_ROOT}/configs/local_v15_syndiff.yaml"

INIT_LOCAL="${UCN_ROOT}/ckpt/init_local_${EXID}_fromSD15.ckpt"
INIT_GLOBAL="${UCN_ROOT}/ckpt/init_global_${EXID}_fromSD15.ckpt"
INIT_UNI="${UCN_ROOT}/ckpt/init_uni_${EXID}_fromSD15.ckpt"

LOGDIR="${UCN_ROOT}/logs/finetune_uni_syndiff_${EXID}_FromNOPretrain_30Ksteps"
LOGGER_VERSION="7"
CKPT_DIR="${LOGDIR}/lightning_logs/version_${LOGGER_VERSION}/checkpoints"
CKPT_30K="${CKPT_DIR}/periodic-stepstep=000030000.ckpt"

GEN_ROOT="/data/coding/datasets/WaymoV2/Ucn_byPure_Finetune30K_FromNOPretrain"
INFER_EXPROOT="/data/ucn_infer_cache_ex7"
EVAL_CACHE="/data/ucn_eval_cache_ex7"

echo "========== [EX7] Sanity checks =========="
if [ ! -d "${UCN_ROOT}" ]; then
  echo "ERROR: Uni-ControlNet repo not found: ${UCN_ROOT}"
  exit 1
fi
if [ ! -f "${SD_CKPT}" ]; then
  echo "ERROR: SD1.5 pruned ckpt not found: ${SD_CKPT}"
  exit 1
fi
if [ ! -f "${CFG_INIT_LOCAL}" ]; then
  echo "ERROR: local_v15.yaml not found: ${CFG_INIT_LOCAL}"
  exit 1
fi
if [ ! -f "${CFG_INIT_GLOBAL}" ]; then
  echo "ERROR: global_v15.yaml not found: ${CFG_INIT_GLOBAL}"
  exit 1
fi
if [ ! -f "${CFG_INIT_UNI}" ]; then
  echo "ERROR: uni_v15.yaml not found: ${CFG_INIT_UNI}"
  exit 1
fi
if [ ! -f "${CFG_TRAIN}" ]; then
  echo "ERROR: train config not found: ${CFG_TRAIN}"
  exit 1
fi

echo
echo "========== [EX7] Step B: Create init weights from SD1.5 =========="
cd "${UCN_ROOT}"

if [ ! -f "${INIT_LOCAL}" ]; then
  echo "[init_local] create: ${INIT_LOCAL}"
  python -u utils/prepare_weights.py init_local "${SD_CKPT}" "${CFG_INIT_LOCAL}" "${INIT_LOCAL}"
else
  echo "[init_local] already exists, skip: ${INIT_LOCAL}"
fi

if [ ! -f "${INIT_GLOBAL}" ]; then
  echo "[init_global] create: ${INIT_GLOBAL}"
  python -u utils/prepare_weights.py init_global "${SD_CKPT}" "${CFG_INIT_GLOBAL}" "${INIT_GLOBAL}"
else
  echo "[init_global] already exists, skip: ${INIT_GLOBAL}"
fi

if [ ! -f "${INIT_UNI}" ]; then
  echo "[integrate] create: ${INIT_UNI}"
  python -u utils/prepare_weights.py integrate "${INIT_LOCAL}" "${INIT_GLOBAL}" "${CFG_INIT_UNI}" "${INIT_UNI}"
else
  echo "[integrate] already exists, skip: ${INIT_UNI}"
fi

echo
echo "========== [EX7] Step C: Train Uni-ControlNet (NO-pretrain) to 30K steps =========="
mkdir -p "${LOGDIR}"

if [ -f "${CKPT_30K}" ]; then
  echo "[train] 30K checkpoint already exists, skip training:"
  echo "  ${CKPT_30K}"
else
  echo "[train] start training..."
  python -u src/train/train.py \
    --config-path "${CFG_TRAIN}" \
    --learning-rate 1e-5 \
    --batch-size 4 \
    --training-steps 30084 \
    --resume-path "${INIT_UNI}" \
    --logdir "${LOGDIR}" \
    --logger-version "${LOGGER_VERSION}" \
    --log-freq 1000 \
    --ckpt-every-n-steps 30000 \
    --sd-locked True \
    --gpus 1
fi

if [ ! -f "${CKPT_30K}" ]; then
  echo "ERROR: 30K checkpoint not found after training:"
  echo "  expected: ${CKPT_30K}"
  exit 1
fi

echo
echo "========== [EX7] Step D: Inference (Waymo) =========="
mkdir -p "${GEN_ROOT}"
mkdir -p "${INFER_EXPROOT}"

python -u /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py \
  --splits training validation testing \
  --camera front \
  --uni-config /data/coding/Uni-ControlNet/configs/uni_v15.yaml \
  --uni-ckpt "${CKPT_30K}" \
  --image-resolution 512 \
  --num-samples 1 \
  --ddim-steps 50 \
  --scale 7.5 \
  --strength 1.0 \
  --global-strength 0.0 \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --out-root "${GEN_ROOT}" \
  --experiments-root "${INFER_EXPROOT}" \
  --experiment-id "${EXID}" \
  --experiment-note "${EXNOTE}" \
  --overwrite \
  --verbose

echo
echo "========== [EX7] Step E: Evaluation (docker) =========="
mkdir -p "${EVAL_CACHE}" "${EVAL_CACHE}/torch_hub"
bash /home/shogo/coding/eval/ucn_eval/EX7_all.sh

echo
echo "✅ [EX7] DONE"
echo "  Train logdir : ${LOGDIR}"
echo "  30K ckpt     : ${CKPT_30K}"
echo "  Gen root     : ${GEN_ROOT}"
echo "  Eval cache   : ${EVAL_CACHE}"
