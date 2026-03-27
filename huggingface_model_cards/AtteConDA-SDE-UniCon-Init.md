---
license: creativeml-openrail-m
pipeline_tag: image-to-image
library_name: custom
base_model:
  - stable-diffusion-v1-5/stable-diffusion-v1-5
  - shihaozhao/uni-controlnet
tags:
  - diffusion
  - controllable-generation
  - image-to-image
  - autonomous-driving
  - synthetic-data
  - data-augmentation
  - stable-diffusion
  - unicontrolnet
  - atteconda
---

# AtteConDA-SDE-UniCon-Init

Initialization checkpoint used as the EPOCH-0 style starting point for Uni-ControlNet-initialized AtteConDA training.

> **Important**
>
> This release is a **project checkpoint** for the custom AtteConDA / Uni-ControlNet codebase.
> It is **not** presented here as a drop-in diffusers pipeline.

---

## Project overview

AtteConDA is a multi-condition diffusion framework for autonomous-driving synthetic data augmentation.
It uses:

- semantic segmentation
- depth
- edge
- text prompts

and introduces **PAM (Patch-wise Adaptation Module)** to reduce inter-condition conflicts in multi-condition generation.

---

## Variant summary

- **Model name:** `AtteConDA-SDE-UniCon-Init`
- **Paper-side label:** `Uni-ControlNet initialization checkpoint`
- **Variant type:** initialization checkpoint intended for training start rather than final evaluation
- **Primary domain:** autonomous-driving street scenes
- **Primary use case:** structure-preserving synthetic appearance augmentation
- **Reference repository:** `https://github.com/ShogoNoguchi/AtteConDA`
- **Model collection:** `https://huggingface.co/collections/Shogo-Noguchi/atteconda`

---

## Training / evaluation domain

Training datasets used in the project:

- BDD10K semantic subset
- Cityscapes
- GTA5
- nuImages (front)
- BDD100K (excluding BDD10K overlap)

Evaluation dataset used in the project:

- Waymo front camera images

Resolution for benchmarked Waymo inference:

- **512 x 512**

---

## Intended use

This checkpoint is intended for:

1. reproducible synthetic data generation experiments
2. autonomous-driving appearance augmentation experiments
3. comparison against prior work under the shared AtteConDA pipeline
4. follow-up research on structure-preserving multi-condition control

---

## Not intended use

This release is **not** intended as:

- a general-purpose unrestricted image-generation model
- a safety-critical deployment model
- a benchmark-free “one-click” productized model
- a replacement for checking the repository-specific inference commands

---

## How to use this checkpoint

This checkpoint is intended primarily as a **training initialization checkpoint**.

Example training usage:

```bash
cd /data/coding/B_thesis_Repo/Uni-ControlNet

python -u src/train/train.py   --config-path ./configs/local_v15_syndiff.yaml   --learning-rate 1e-5   --batch-size 4   --training-steps 30084   --resume-path /ABSOLUTE/PATH/TO/AtteConDA-SDE-UniCon-Init.ckpt   --logdir ./logs/finetune_uni_syndiff_release   --logger-version 0   --log-freq 1000   --ckpt-every-n-steps 30000   --sd-locked True   --gpus 1
```

Use the PAM config instead if you are training the PAM variant.

---

## Limitations

- The released benchmark setting focuses on **autonomous-driving** imagery.
- The checkpoint is evaluated with the AtteConDA project pipeline rather than presented as a generic diffusers package.
- Structure preservation is prioritized; some variants intentionally trade off semantic-only or style-only behavior.
- High-level semantic changes are controlled through prompts, but the main research focus is structure preservation under appearance change.

---

## License and attribution

This model-card template uses **CreativeML OpenRAIL-M** metadata because the released checkpoints belong to a Stable Diffusion v1.5-derived project lineage.
Users must also respect upstream terms and attribution requirements from the project dependencies.

Please also acknowledge:

- Uni-ControlNet
- DGInStyle
- Stable Diffusion v1.5 family
- OneFormer
- Metric3D / Metric3Dv2
- Grounding DINO
- CLIP / open_clip
- Qwen3-VL

Full practical attribution guidance for this repository is documented in:

- `THIRD_PARTY_NOTICES.md`

---

## Citation

```bibtex
@misc{noguchi2026atteconda,
  title        = {AtteConDA: Attention-Based Conflict Suppression in Multi-Condition Diffusion Models and Synthetic Data Augmentation},
  author       = {Shogo Noguchi},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Gunma University}
}
```
