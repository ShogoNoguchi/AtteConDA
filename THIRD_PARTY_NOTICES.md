# THIRD_PARTY_NOTICES.md

This document records the major third-party codebases, checkpoints, and runtime-downloaded models used by **AtteConDA**.

> **Important**
>
> This file is a practical release note for repository maintenance.
> It is **not** legal advice.
> When redistributing code or checkpoints, always read the upstream license text yourself.

---

## 1. Scope

This repository contains three different kinds of material:

```text
AtteConDA release content
├── A) AtteConDA-original files
│   ├── README.md
│   ├── docs/*
│   ├── release metadata
│   ├── prep/* (AtteConDA-added scripts)
│   ├── evaluation wrappers / tooling
│   └── repository organization files
├── B) Vendored / locally modified upstream repositories
│   ├── Uni-ControlNet/
│   └── DGInStyle/
└── C) External checkpoints and runtime-downloaded models
    ├── Stable Diffusion v1.5 family weights
    ├── Uni-ControlNet initialization checkpoint
    ├── DGInStyle baseline weights
    ├── OneFormer
    ├── Metric3D
    ├── Grounding DINO
    ├── CLIP / open_clip
    ├── Qwen3-VL
    └── LPIPS / AlexNet
```

The **root LICENSE** in this release bundle is intended for **AtteConDA-original files only**, unless a file or subtree states otherwise.

---

## 2. AtteConDA-original release files

Recommended license for AtteConDA-original top-level files in this public release:

- **Apache License 2.0**

Rationale for this recommendation:

1. This repository contains original AtteConDA-specific code, documentation, and release assets.
2. The project also vendors or modifies third-party subtrees with their own upstream licenses.
3. Apache-2.0 is a practical and explicit top-level license for original release files while still allowing vendored subtrees to preserve their own upstream notices.

Rule of thumb:

- If a file is **AtteConDA-original** and does not belong to a separately licensed upstream subtree, treat it as covered by the root `LICENSE`.
- If a file belongs to `Uni-ControlNet/` or `DGInStyle/`, keep the **upstream license of that subtree**.

---

## 3. Vendored / modified upstream repositories included in this repo

## 3.1 Uni-ControlNet

- Upstream repository: `https://github.com/ShihaoZhaoZSH/Uni-ControlNet`
- Upstream authors: Shihao Zhao et al.
- Upstream license: **MIT**
- Role inside AtteConDA:
  - architectural starting point
  - pretrained local-control initialization
  - training / inference backbone

Recommended handling inside this repository:

```text
Uni-ControlNet/
├── keep upstream copyright/license notice
├── mark locally modified files clearly in commit history
└── do not imply that Uni-ControlNet-original files are relicensed by the root LICENSE
```

If you substantially modify files under this subtree, keep the original notice and add clear modification markers in the file header or commit history.

---

## 3.2 DGInStyle

- Upstream repository: `https://github.com/prs-eth/DGInStyle`
- Upstream authors: Yuru Jia et al.
- Upstream code license: **Apache License 2.0**
- Role inside AtteConDA:
  - prior-work baseline
  - comparison inference path
  - reference for autonomous-driving diffusion augmentation

Recommended handling inside this repository:

```text
DGInStyle/
├── keep upstream Apache-2.0 notice
├── preserve attribution
└── separate AtteConDA-specific local edits from upstream-origin files when possible
```

---

## 4. External checkpoints and weight releases

These items are **not** covered by the root code license in the same way as ordinary source files.

## 4.1 Stable Diffusion v1.5 family base weights

- Public model source used in practice: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Upstream / base-model license: **CreativeML OpenRAIL-M**
- Why it matters:
  - AtteConDA checkpoints are derived from the Stable Diffusion v1.5 family.
  - Downstream checkpoint distribution should therefore respect the base-model terms.

**Release recommendation for Hugging Face model cards in this repository:**  
Use model-card metadata consistent with the Stable Diffusion v1.5 lineage and make the dependency explicit in the card body.

---

## 4.2 Uni-ControlNet initialization checkpoint

- Public checkpoint source: `shihaozhao/uni-controlnet`
- Upstream checkpoint page should be cited in model cards
- Practical role:
  - initialization for Tune30K / Tune60K / Tune90K / PAM60K
  - initialization checkpoint released here as `AtteConDA-SDE-UniCon-Init`

When you publish AtteConDA checkpoints, explicitly state whether the released checkpoint:

- starts from Stable Diffusion only, or
- starts from Uni-ControlNet initialization.

---

## 4.3 DGInStyle baseline checkpoint

- Public checkpoint source: `yurujaja/DGInStyle`
- Practical role:
  - prior-work comparison baseline only
  - not an AtteConDA weight release

If you document comparison results, clearly separate:

- **AtteConDA models**
- **external baseline models**

so that users do not confuse the licenses or authorship.

---

## 5. Runtime-downloaded models and tools

The following components are typically downloaded automatically or expected to exist locally during preparation or evaluation.

| Component | Used by | Expected source |
|---|---|---|
| OneFormer (Cityscapes) | `prep/ucn_build_conditions.py` | `shi-labs/oneformer_cityscapes_swin_large` |
| Metric3Dv2 / Metric3D ONNX | `prep/ucn_build_conditions.py`, `eval/ucn_eval/eval_unicontrol_waymo.py` | local Metric3D clone / exported ONNX |
| Grounding DINO | `eval/ucn_eval/eval_unicontrol_waymo.py` | `IDEA-Research/grounding-dino-base` |
| open_clip / SigLIP | `prep/ucn_build_prompts.py` | downloaded to Hugging Face or model cache |
| Qwen3-VL-32B-Instruct | `prep/ucn_build_prompts.py` | local HF snapshot or Hugging Face |
| LPIPS / AlexNet | `eval/ucn_eval/eval_unicontrol_waymo.py` | Python package / torch model zoo |
| Tesseract OCR | `eval/ucn_eval/eval_unicontrol_waymo.py` | system package inside Docker overlay |

For these components, the repository should document:

1. **what they are used for**
2. **where they are expected to come from**
3. **which path or cache the current scripts assume**
4. **that their upstream terms remain in force**

---

## 6. PixelPonder note

AtteConDA acknowledges the **PixelPonder** paper as an idea-level reference for conflict-aware multi-condition handling.

Release rule used in this repository:

- cite the **paper** as prior art,
- **do not** claim code provenance from an unlicensed code tree,
- and **do not** copy third-party source into the repository unless the license allows it.

This repository therefore treats PixelPonder as:

```text
Paper-level inspiration
└── not a declared code donor in THIRD_PARTY_NOTICES
```

---

## 7. Practical repository rules

## 7.1 Keep upstream code provenance visible

Do **not** flatten all upstream-origin files into a single anonymous repository history.
Keep the provenance visible through:

- preserved directory boundaries
- preserved headers where possible
- `THIRD_PARTY_NOTICES.md`
- commit messages that clearly distinguish original work from upstream edits

## 7.2 Keep model provenance visible

For every released AtteConDA model card, clearly state:

- base model lineage
- whether Uni-ControlNet initialization was used
- intended use
- limitations
- evaluation domain
- upstream acknowledgements

## 7.3 Keep public documentation honest

Do **not** say:

- “from scratch” if the checkpoint starts from Uni-ControlNet initialization
- “diffusers-native pipeline” if the checkpoint requires the custom training/inference code here
- “paper released” before the paper is actually public

---

## 8. Recommended attribution block for README / Pages

A concise public-facing attribution block can look like this:

```text
Acknowledgements:
This project builds on Uni-ControlNet and compares against DGInStyle.
Stable Diffusion v1.5 provides the base diffusion prior.
The preparation and evaluation pipeline uses OneFormer, Metric3D, Grounding DINO,
CLIP / open_clip, Qwen3-VL, LPIPS, and Tesseract.
PixelPonder is cited as paper-level inspiration for dynamic conflict-aware control,
but this release does not claim code provenance from an unlicensed source tree.
```

---

## 9. Recommended distribution policy for this repository

```text
Root repository release policy
├── AtteConDA-original top-level files -> Apache-2.0
├── Uni-ControlNet subtree            -> MIT (upstream)
├── DGInStyle subtree                 -> Apache-2.0 (upstream)
└── released model cards              -> explicitly state upstream base-model terms
```

This is the cleanest practical arrangement for a public release that contains both original work and vendored/modified upstream subtrees.

---

## 10. Files in this release bundle related to licensing

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- `huggingface_model_cards/*`
- `README.md`
- `docs/RESEARCHER_README.md`
