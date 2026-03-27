# Researcher README for AtteConDA

This document is the **lab-facing hand-off guide** for AtteConDA.

It is intentionally more detailed than the public root README because the hidden goal of this repository is not only public presentation but also **successful continuation by future lab members**.

---

## 0. What this document is for

Use this file when you need to understand:

- what is original AtteConDA code and what is vendored upstream code
- how the full pipeline is connected
- which absolute paths the current lab machine uses
- how to go from raw RGB datasets to
  - condition maps
  - prompts
  - annotation CSV files
  - training
  - Waymo inference
  - evaluation
- how to publish this repository cleanly without losing provenance

---

## 1. Big-picture flow

```text
Raw RGB datasets
├── Training RGB
│   ├── BDD10K semantic subset
│   ├── Cityscapes
│   ├── GTA5
│   ├── nuImages (front)
│   └── BDD100K (excluding overlap with BDD10K subset)
└── Evaluation RGB
    └── Waymo front camera (first / mid10s / last)

Pipeline
├── prep/ucn_build_conditions.py
│   ├── OneFormer      -> semantic segmentation condition maps
│   ├── Metric3D       -> depth condition maps
│   └── Canny          -> edge condition maps
├── prep/ucn_build_prompts.py
│   ├── open_clip      -> source weather / time estimation
│   └── Qwen3-VL       -> caption generation + target-style prompt generation
├── Uni-ControlNet/src/tools/build_anno_syndiff_multi.py
│   └── generates anno_syndiff_*.csv
├── Uni-ControlNet/src/train/train.py
│   └── trains AtteConDA checkpoints
├── Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py
│   └── generates Waymo synthetic RGB
└── eval/ucn_eval/eval_unicontrol_waymo.py
    └── evaluates structure / reality / diversity / text alignment
```

---

## 2. Repository role map

```text
AtteConDA repository
├── Public-release files
│   ├── README.md
│   ├── LICENSE
│   ├── THIRD_PARTY_NOTICES.md
│   ├── CITATION.cff
│   ├── docs/index.html
│   └── docs/RESEARCHER_README.md
├── AtteConDA-original research scripts
│   ├── prep/ucn_build_conditions.py
│   ├── prep/ucn_build_prompts.py
│   ├── eval/ucn_eval/eval_unicontrol_waymo.py
│   ├── eval/ucn_eval/compare_experiments_ucn.py
│   └── custom glue code / configs / wrappers
├── Vendored upstream subtree: Uni-ControlNet/
│   ├── modified for AtteConDA training/inference
│   └── retain upstream MIT provenance
├── Vendored upstream subtree: DGInStyle/
│   ├── used as prior-work baseline
│   └── retain upstream Apache-2.0 provenance
└── Assets
    └── figs/
```

---

## 3. Dependency tree with provenance labels

```text
AtteConDA runtime dependency tree
├── [Original repo code]
│   ├── prep/ucn_build_conditions.py
│   ├── prep/ucn_build_prompts.py
│   ├── eval/ucn_eval/eval_unicontrol_waymo.py
│   ├── docs/*
│   └── release metadata
├── [Vendored upstream code]
│   ├── Uni-ControlNet/        -> upstream MIT
│   └── DGInStyle/             -> upstream Apache-2.0
├── [External Python libraries]
│   ├── torch
│   ├── torchvision
│   ├── torchaudio
│   ├── pytorch-lightning
│   ├── diffusers
│   ├── transformers
│   ├── open_clip
│   ├── onnxruntime
│   ├── lpips
│   ├── pytorch-msssim
│   ├── scikit-image
│   └── tensorboard
└── [External models / checkpoints]
    ├── Stable Diffusion v1.5 family
    ├── Uni-ControlNet initialization checkpoint
    ├── DGInStyle checkpoint
    ├── OneFormer Cityscapes
    ├── Metric3D / Metric3Dv2
    ├── Grounding DINO
    ├── CLIP / open_clip
    ├── Qwen3-VL-32B-Instruct
    └── AlexNet weights used through LPIPS / torchvision
```

---

## 4. Absolute dataset paths on the current lab machine

These are the current working assumptions used by the scripts.

### 4.1 Training datasets

```text
BDD10K semantic subset
└── /home/shogo/coding/datasets/BDD10K_Seg_Matched10K_ImagesLabels/images

Cityscapes
└── /home/shogo/coding/datasets/cityscapes/leftImg8bit

GTA5
└── /home/shogo/coding/datasets/GTA5/images/images

nuImages front
└── /home/shogo/coding/datasets/nuimages/samples/CAM_FRONT

BDD100K (excluding overlap)
└── /home/shogo/coding/datasets/BDD_100K_pure100k
```

### 4.2 Waymo evaluation RGB

```text
/home/shogo/coding/datasets/WaymoV2/extracted/{training|validation|testing}/front/{segment_id}/{frame}.jpg
```

Each segment contributes:

- `*_first.jpg`
- `*_mid10s.jpg`
- `*_last.jpg`

---

## 5. Waymo structure used by evaluation

The evaluation pipeline assumes the following aligned assets for Waymo:

```text
WaymoV2/
├── extracted/
│   └── {training|validation|testing}/front/{segment_id}/*.jpg
├── OneFormer_cityscapes/
│   └── {training|validation|testing}/front/{segment_id}/*_predTrainId.npy
├── Prompts_gptoss/ or /data/syndiff_prompts/prompts_eval_waymo
│   └── text prompts
├── Metricv2DepthNPY/
│   └── {training|validation|testing}/front/{segment_id}/*_depth.npy
└── CannyEdge/
    └── {training|validation|testing}/front/{segment_id}/*_edge.png
```

Important evaluation convention:

- **Waymo RGB is the source image X** in the evaluation pipeline.
- Generation is always run at **512 x 512** for metric consistency.
- The evaluation compares original vs generated images in shared prediction spaces.

---

## 6. Model family map

```text
Released checkpoints
├── FullScratch30K
│   └── no Uni-ControlNet initialization
├── Tune30K
│   └── initialized from Uni-ControlNet-compatible checkpoint
├── Tune60K
│   └── initialized from Uni-ControlNet-compatible checkpoint
├── Tune90K
│   └── initialized from Uni-ControlNet-compatible checkpoint
├── PAM60K
│   └── initialized from Uni-ControlNet-compatible checkpoint + PAM
└── AtteConDA-SDE-UniCon-Init
    └── initialization checkpoint used as epoch-0 style starting point
```

---

## 7. External models and where they come from

This is the most important “what do I need to download?” table for future lab members.

| Component | Used by | Default name / path in scripts | How it is obtained in practice |
|---|---|---|---|
| Stable Diffusion v1.5 family weights | Uni-ControlNet training / inference backbone | embedded in custom code / checkpoint lineage | obtained through the upstream model lineage used by Uni-ControlNet / local checkpoints |
| Uni-ControlNet initialization checkpoint | training starting point for Tune* / PAM60K | local checkpoint path you pass to `--resume-path` | downloaded from the released AtteConDA HF repo or from local lab storage |
| DGInStyle baseline weights | prior-work inference baseline | `yurujaja/DGInStyle` | pulled from Hugging Face by the DGInStyle inference code |
| OneFormer Cityscapes | condition-map semantic segmentation | `shi-labs/oneformer_cityscapes_swin_large` | downloaded automatically via `transformers` / Hugging Face cache |
| Metric3D ONNX | condition-map depth generation | `/home/shogo/coding/Metric3D/onnx/onnx/model.onnx` | local clone / exported ONNX file expected at the fixed path |
| Grounding DINO | object-preservation evaluation | `IDEA-Research/grounding-dino-base` | downloaded automatically during evaluation |
| open_clip / SigLIP | prompt-generation CLIP classification | open_clip cache under Hugging Face hub cache | downloaded automatically if missing |
| Qwen3-VL-32B-Instruct | prompt-generation caption model | `/data/hf_models/Qwen/Qwen3-VL-32B-Instruct` | local Hugging Face snapshot or remote download |
| LPIPS / AlexNet | diversity metric | Python package + model weights | installed via pip / downloaded by the relevant library |
| Tesseract OCR | text / sign-related evaluation support | system package inside Docker overlay | provided through the evaluation environment |
| CLIP embeddings for CMMD / R-Precision | reality / text-alignment metrics | library-managed | resolved through the evaluation environment |

---

## 8. Environment policy

### 8.1 Main rule

Keep the repository aligned with:

- **Ubuntu**
- **CUDA 12.8**
- **cu128 PyTorch**
- **RTX 5090-compatible stack**

### 8.2 Why the current `environment.yaml` is preserved

The current file already reflects a working lab environment.
For public release, reproducibility matters more than pretending the environment has been simplified.

### 8.3 Verification

Always run:

```bash
bash scripts/verify_env.sh atteconda_env
```

---

## 9. End-to-end command flow

The pipeline is easiest to remember as:

```text
RGB
├── build conditions
├── build prompts
├── build anno csv
├── train
├── infer on Waymo
└── evaluate
```

The rest of this document expands each stage.

---

## 10. Stage A: Build condition maps

The condition-map script is:

```text
prep/ucn_build_conditions.py
```

### 10.1 What it produces

```text
/data/ucn_condmaps/{DATASET_KEY}/
├── depth/
├── edge/
└── semseg/
```

### 10.2 Internal dependency tree

```text
prep/ucn_build_conditions.py
├── OneFormer (semantic segmentation)
├── Metric3D ONNX (depth)
├── OpenCV Canny (edge)
├── TensorBoard logging
└── strict CUDA requirement
```

### 10.3 Canonical command

```bash
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e MAIN_PY=/app/ucn_build_conditions.py \
  -e PIP_OVERLAY_DIR=/data/ucn_prep_cache/pip-overlay \
  -e REQS_OVERLAY_PATH=/data/ucn_prep_cache/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs huggingface_hub>=0.34,<1.0 einops' \
  -e PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/prep/ucn_build_conditions.py:/app/ucn_build_conditions.py:ro \
  -v /home/shogo/coding/datasets:/home/shogo/coding/datasets:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  ucn-eval \
  --datasets all --tasks all --semseg-batch-size 1
```

### 10.4 Subset check command

```bash
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e MAIN_PY=/app/ucn_build_conditions.py \
  -e PIP_OVERLAY_DIR=/data/ucn_prep_cache/pip-overlay \
  -e REQS_OVERLAY_PATH=/data/ucn_prep_cache/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs huggingface_hub>=0.34,<1.0 einops' \
  -e PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/prep/ucn_build_conditions.py:/app/ucn_build_conditions.py:ro \
  -v /home/shogo/coding/datasets:/home/shogo/coding/datasets:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  ucn-eval \
  --datasets all --tasks all --subset-check --verbose
```

### 10.5 What to watch carefully

- CUDA is required. The script is intentionally strict.
- Metric3D output direction is inverted in the pipeline so that the saved visualization matches the project convention.
- Cityscapes / BDD10K / BDD100K include split directories in their output tree.
- GTA5 / nuImages do not naturally have split directories in the same way.

---

## 11. Stage B: Build prompts

The prompt-generation script is:

```text
prep/ucn_build_prompts.py
```

### 11.1 What it does

```text
prep/ucn_build_prompts.py
├── open_clip SigLIP
│   ├── predicts source weather
│   └── predicts source time-of-day
├── Qwen3-VL-32B-Instruct
│   └── creates short, weather/time-suppressed scene captions
└── prompt formatter
    ├── training prompts
    └── Waymo evaluation prompts with target weather/time changes
```

### 11.2 Output location

```text
/data/syndiff_prompts/
├── prompts_train/
└── prompts_eval_waymo/
```

### 11.3 Canonical command

```bash
docker run --rm --gpus all \
  --entrypoint /app/entrypoint_prompts_v2.sh \
  -e MPLBACKEND=Agg \
  -e MAIN_PY=/app/ucn_build_prompts.py \
  -e PIP_OVERLAY_DIR=/data/ucn_prep_cache/pip-overlay \
  -e HF_HOME=/root/.cache/huggingface \
  -e OPENCLIP_CACHE_DIR=/root/.cache/huggingface/hub \
  -e VLM_LOCAL_DIR=/data/hf_models/Qwen/Qwen3-VL-32B-Instruct \
  -e QWEN_QUANT=4bit \
  -e TRANSFORMERS_OFFLINE=0 -e HF_HUB_OFFLINE=0 \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint_prompts_v2.sh:/app/entrypoint_prompts_v2.sh:ro \
  -v /home/shogo/coding/prep/ucn_build_prompts.py:/app/ucn_build_prompts.py:ro \
  -v /home/shogo/coding/datasets:/home/shogo/coding/datasets:ro \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data:/data \
  ucn-eval \
  --jobs train eval_waymo \
  --datasets all \
  --cityscapes-use-splits train \
  --bdd10k10k-use-splits train \
  --bdd100k-use-splits train \
  --vlm-local-dir /data/hf_models/Qwen/Qwen3-VL-32B-Instruct \
  --openclip-cache-dir /root/.cache/huggingface/hub \
  --quant 4bit \
  --clip-batch 8 \
  --cap-max-new 96 \
  --cap-words 45 \
  --hidethinking \
  --verbose
```

### 11.4 Important implementation idea

The prompt generator separates:

```text
Prompt construction
├── scene/layout caption
└── weather/time style injection
```

That separation is important because the project wants to:

- preserve structure,
- change appearance,
- and avoid trivial copying of original weather/time words into the caption.

---

## 12. Stage C: Build annotation CSV files

The CSV builder is:

```text
Uni-ControlNet/src/tools/build_anno_syndiff_multi.py
```

### 12.1 What it produces

```text
Uni-ControlNet/data/
├── anno_syndiff_train.csv
└── anno_syndiff_waymo_val.csv
```

### 12.2 Command

```bash
cd /data/coding/B_thesis_Repo/Uni-ControlNet

python -u src/tools/build_anno_syndiff_multi.py \
  --make-train \
  --make-waymo-val \
  --limit -1 \
  --verbose
```

### 12.3 Internal dependency tree

```text
build_anno_syndiff_multi.py
├── /data/syndiff_prompts/prompts_train/*.csv
├── /data/syndiff_prompts/prompts_eval_waymo/*.csv
├── /data/ucn_condmaps/*
├── Waymo condition-map directories
└── RGB dataset roots
```

---

## 13. Stage D: Training

Training entry point:

```text
Uni-ControlNet/src/train/train.py
```

### 13.1 Training concept tree

```text
Training
├── non-PAM configuration
│   └── local_v15_syndiff.yaml
└── PAM configuration
    └── local_v15_syndiff_pam2.yaml
```

### 13.2 Non-PAM training command

```bash
cd /data/coding/B_thesis_Repo/Uni-ControlNet

python -u src/train/train.py \
  --config-path ./configs/local_v15_syndiff.yaml \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --training-steps 30084 \
  --resume-path /data/coding/Uni-ControlNet/ckpt/init_uni_EX7_fromSD15.ckpt \
  --logdir ./logs/finetune_uni_syndiff_EX7_FromNOPretrain_30Ksteps \
  --logger-version 7 \
  --log-freq 1000 \
  --ckpt-every-n-steps 30000 \
  --sd-locked True \
  --gpus 1
```

### 13.3 PAM training command

```bash
cd /data/coding/B_thesis_Repo/Uni-ControlNet

python -u src/train/train.py \
  --config-path ./configs/local_v15_syndiff_pam2.yaml \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --training-steps 90168 \
  --resume-path /data/coding/ckpts/version_4/checkpoints/epoch=1-step=60000.ckpt \
  --logdir ./logs/finetune_uni_syndiff_pam2_From60KpurePretrain \
  --logger-version 0 \
  --log-freq 1000 \
  --ckpt-every-n-steps 30000 \
  --sd-locked True \
  --gpus 1
```

### 13.4 Resume-path rule

```text
resume-path behavior
├── continue training from your own ckpt  -> pass that ckpt
├── initialize from Uni-ControlNet-init   -> pass AtteConDA-SDE-UniCon-Init.ckpt
└── train fully from scratch              -> omit --resume-path
```

### 13.5 Reproducibility note

The project preference is:

- reproducibility across runs
- but diversity inside a run

So when you later refine seed handling, preserve that philosophy rather than forcing everything to become globally deterministic.

---

## 14. Stage E: Waymo inference for our model

Inference entry point:

```text
Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py
```

### 14.1 Canonical command

```bash
python -u /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py \
  --splits training validation testing \
  --camera front \
  --uni-config /data/coding/Uni-ControlNet/configs/uni_v15.yaml \
  --uni-ckpt /data/coding/Uni-ControlNet/logs/finetune_uni_syndiff_EX7_FromNOPretrain_30Ksteps/lightning_logs/version_7/checkpoints/periodic-stepstep=000030000.ckpt \
  --image-resolution 512 \
  --num-samples 1 \
  --ddim-steps 50 \
  --scale 7.5 \
  --strength 1.0 \
  --global-strength 0.0 \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --out-root /data/coding/datasets/WaymoV2/Ucn_byPure_Finetune30K_FromNOPretrain \
  --experiments-root /data/ucn_infer_cache_ex7 \
  --experiment-id EX7 \
  --experiment-note "Perform finetune 30Ksteps without using uni.ckpt, which is officially distributed by UniControlNet. Abration is used to measure the power of prior learning." \
  --overwrite \
  --verbose
```

### 14.2 Config rule

```text
Inference config rule
├── non-PAM checkpoint -> use the non-PAM config
└── PAM checkpoint     -> use the PAM config
```

---

## 15. Stage F: DGInStyle baseline inference

Baseline entry point:

```text
DGInStyle/src/tools/infer_waymo_unicontrol_offline.py
```

### 15.1 Canonical command

```bash
cd /data/coding/DGInStyle

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
  --experiment-note "DGInStyle + Qwen3-VL prompts" \
  --verbose
```

---

## 16. Stage G: Evaluation

Evaluation entry point:

```text
eval/ucn_eval/eval_unicontrol_waymo.py
```

### 16.1 Evaluation concept tree

```text
Evaluation metrics
├── Structure preservation
│   ├── semantic segmentation mIoU
│   ├── depth RMSE
│   ├── edge L1
│   └── object-preservation F1
├── Reality
│   └── CLIP-CMMD
├── Diversity
│   ├── LPIPS
│   └── 1 - MS-SSIM
└── Text alignment
    └── CLIP R-Precision
```

### 16.2 Build the Docker image

```bash
cd /home/shogo/coding/eval/ucn_eval

docker build \
  -t ucn-eval \
  -f /home/shogo/coding/eval/ucn_eval/docker/Dockerfile \
  /home/shogo/coding/eval/ucn_eval
```

### 16.3 Evaluation run

```bash
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e PIP_OVERLAY_DIR=/data/ucn_eval_cache_ex9/pip-overlay \
  -e REQS_OVERLAY_PATH=/data/ucn_eval_cache_ex9/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs prefetch_generator easydict thop scikit-image pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib lpips pytorch-msssim' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py:/app/eval_unicontrol_waymo.py:ro \
  -v /home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /data:/data \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /data/ucn_eval_cache_ex9/torch_hub:/root/.cache/torch/hub \
  ucn-eval \
  --cache-root /data/ucn_eval_cache_ex9 \
  --splits training validation testing \
  --camera front \
  --tasks all \
  --reality-metric clip-cmmd \
  --gdinomodel IDEA-Research/grounding-dino-base \
  --det-prompts car truck bus motorcycle bicycle person pedestrian "traffic light" "traffic sign" "stop sign" "speed limit sign" "crosswalk sign" "construction sign" "traffic cone" \
  --ocr-engine tesseract \
  --iou-thr 0.5 \
  --orig-root /home/shogo/coding/datasets/WaymoV2/extracted \
  --gen-root /data/coding/datasets/WaymoV2/DGInstyle_Pure \
  --prompt-root /data/syndiff_prompts/prompts_eval_waymo \
  --annotation-mode all \
  --annotate-limit 24 \
  --annotate-out /data/ucn_eval_cache_ex9/viz_ex9 \
  --tb --tb-dir /data/ucn_eval_cache_ex9/tensorboard_ex9 \
  --experiment-id EX9 \
  --experiment-note "DGInStyle + Qwen3-VL prompts" \
  --drivable-methods onefroad \
  --no-auto-batch \
  --verbose
```

### 16.4 Output tree to expect

```text
/data/ucn_eval_cache_<experiment>/
├── pip-overlay/
├── tensorboard_<experiment>/
├── viz_<experiment>/
├── metrics*.json / csv / logs
└── cached intermediate predictions
```

---

## 17. Canonical figures already present in `figs/`

These are already strong enough for the first public release and should be referenced directly in the README and Pages.

```text
Core presentation figures
├── 従来水増しとの比較.png
├── pipeline_multicondition.png
├── model_detail.png
├── pam_architecture.png
├── impact_pretrain.png
├── qualitative_tune.png
├── qualitative_tune_拡大PAM動機.png
├── qualitative_pam.png
├── qualitative_pam_拡大PAMによる改善示し図.png
├── スケーリング1.png
├── スケーリング2.png
├── evaluation_構造保持.png
├── リアリティ_例.png
├── 多様性_例.png
└── テキスト追従_例.png
```

---

## 18. Public-release cleanup plan

The repository should distinguish between:

```text
Public-root content
├── canonical pipeline scripts
├── documentation
├── licensing / attribution
└── figures

Legacy / experiment scratch
├── old shell wrappers
├── legacy eval variants
├── notebooks used once
└── backup files
```

### 18.1 Good cleanup candidates

```text
Safe delete
└── prep/gen_prompts_synad.py.bak

Archive
├── eval/ucn_eval/Compare_EX1_EX2.sh
├── eval/ucn_eval/EX*.sh
├── eval/ucn_eval/eval_unicontrol_waymo_old1.py
├── eval/ucn_eval/poin.py
└── eval/ucn_eval/YOLOP.ipynb
```

### 18.2 Helper script

This release bundle provides:

```text
REPO_CLEANUP_COMMANDS.sh
```

Use it only after reviewing the move plan.

---

## 19. GitHub Pages policy

The Pages source in this release is intentionally simple.

```text
Pages deployment logic
├── docs/index.html         -> source page in the repo
├── figs/*                  -> copied into the site artifact
└── .github/workflows/pages.yml
    ├── checkout
    ├── configure-pages
    ├── build _site
    ├── copy figs
    ├── patch figure paths
    ├── upload artifact
    └── deploy
```

The page is static on purpose.
It is easier to maintain and less likely to break than a heavier framework.

---

## 20. Hugging Face publication plan

A clean public release should **not** leave the model pages blank.

### 20.1 Provided files

```text
huggingface_model_cards/
├── AtteConDA-SDE-Scratch-30K.md
├── AtteConDA-SDE-UniCon-30K.md
├── AtteConDA-SDE-UniCon-60K.md
├── AtteConDA-SDE-UniCon-90K.md
├── AtteConDA-SDE-UniCon-60K-PAM.md
└── AtteConDA-SDE-UniCon-Init.md
```

### 20.2 Publication rule

For each Hugging Face model repo:

1. open the repo page
2. edit `README.md`
3. paste the matching file
4. save

That is enough to avoid an empty model page.

---

## 21. Public-release checklist

```text
Repository release checklist
├── README.md polished
├── LICENSE added
├── THIRD_PARTY_NOTICES.md added
├── CITATION.cff added
├── docs/index.html added
├── docs/RESEARCHER_README.md added
├── scripts/verify_env.sh added
├── .github/workflows/pages.yml added
├── model cards added to HF
├── obvious scratch files archived
└── GitHub Pages source switched to GitHub Actions
```

---

## 22. Common failure modes

### 22.1 Torch installed without cu128

Symptom:
- `torch.cuda.is_available()` is false
- RTX 5090 is not used

Check:
```bash
bash scripts/verify_env.sh atteconda_env
```

### 22.2 Prompt generation too slow

Reason:
- Qwen3-VL 32B is heavy
- this stage is intentionally cache-aware and resume-aware

Mitigation:
- keep `--resume`
- keep the Hugging Face cache on `/data` or persistent storage
- do not wipe prompt caches casually

### 22.3 Evaluation package drift inside Docker

Reason:
- the evaluation stack uses overlay installs

Mitigation:
- keep the overlay cache under `/data`
- keep exact command lines in version control
- avoid ad-hoc package changes without recording them

### 22.4 Confusing public readers with old experiment wrappers

Reason:
- too many `EX*.sh` files in the public root tree

Mitigation:
- archive them
- keep the canonical commands in the root README and this researcher README

---

## 23. Final recommendation for future contributors

When you modify this repository, think in this order:

```text
Contributor priority order
├── 1. Do not break train -> infer -> eval reproducibility
├── 2. Do not break CUDA 12.8 / cu128 compatibility
├── 3. Do not lose upstream provenance
├── 4. Keep the public README and Pages understandable
└── 5. Archive scratch files instead of leaving them in the main tree
```
