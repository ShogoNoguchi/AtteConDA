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

# AtteConDA-SDE-UniCon-60K-PAM

Checkpoint for the 60K-step AtteConDA PAM model with explicit patch-wise condition-conflict suppression.

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

- **Model name:** `AtteConDA-SDE-UniCon-60K-PAM`
- **Paper-side label:** `PAM60K`
- **Variant type:** 60K-step PAM variant with Uni-ControlNet-compatible initialization
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

1. Clone the repository.
2. Place the checkpoint on the local machine.
3. Run the standard AtteConDA inference script with your checkpoint path.

Example:

```bash
python -u /data/coding/Uni-ControlNet/src/tools/infer_waymo_unicontrol_offline.py   --splits training validation testing   --camera front   --uni-config /data/coding/Uni-ControlNet/configs/local_v15_syndiff_pam2.yaml   --uni-ckpt /ABSOLUTE/PATH/TO/YOUR/CHECKPOINT.ckpt   --image-resolution 512   --num-samples 1   --ddim-steps 50   --scale 7.5   --strength 1.0   --global-strength 0.0   --prompt-root /data/syndiff_prompts/prompts_eval_waymo   --out-root /data/coding/datasets/WaymoV2/AtteConDA_Inference   --experiments-root /data/ucn_infer_cache_release   --experiment-id RELEASE   --experiment-note "AtteConDA released checkpoint"   --overwrite   --verbose
```

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
