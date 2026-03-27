# RELEASE_REPORT.md

This report explains what this release bundle is intended to fix in the public AtteConDA repository.

---

## 1. Main public-release problems this bundle addresses

### 1.1 Repository-front-page problems

- The root `README.md` needs to become a real public landing page with:
  - badges
  - a one-line project summary
  - method figures
  - result snapshot
  - model zoo
  - end-to-end commands
  - explicit acknowledgements
  - explicit licensing notes

### 1.2 Missing release-critical legal files

The public repository should have, at minimum:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- `CITATION.cff`

### 1.3 Missing GitHub Pages source

The repository should have:

- `docs/index.html`
- `.github/workflows/pages.yml`

so that the project can be presented as a polished research page instead of only a code dump.

### 1.4 Missing researcher hand-off document

The repository should have:

- `docs/RESEARCHER_README.md`

because the hidden purpose of this repository is not only public visibility but also reproducibility for future lab members.

### 1.5 Missing environment verification entry point

The root README already instructs users to verify the environment, but the repository needs a real script:

- `scripts/verify_env.sh`

### 1.6 Repository hygiene problems

The current public tree mixes public-facing files and local experiment clutter.
A public release should keep canonical entry points visible and archive one-off experiment files where appropriate.

---

## 2. What this bundle adds or replaces

## 2.1 Root-level replacements

- `README.md`
- `.gitignore`

## 2.2 New root-level files

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- `CITATION.cff`
- `APPLY_ORDER.md`
- `RELEASE_REPORT.md`
- `REPO_CLEANUP_COMMANDS.sh`

## 2.3 New documentation files

- `docs/index.html`
- `docs/RESEARCHER_README.md`
- `docs/.nojekyll`

## 2.4 Environment support

- `environment/README.md`
- `scripts/verify_env.sh`

## 2.5 GitHub Pages deployment

- `.github/workflows/pages.yml`

## 2.6 Hugging Face publication support

- `huggingface_model_cards/AtteConDA-SDE-Scratch-30K.md`
- `huggingface_model_cards/AtteConDA-SDE-UniCon-30K.md`
- `huggingface_model_cards/AtteConDA-SDE-UniCon-60K.md`
- `huggingface_model_cards/AtteConDA-SDE-UniCon-90K.md`
- `huggingface_model_cards/AtteConDA-SDE-UniCon-60K-PAM.md`
- `huggingface_model_cards/AtteConDA-SDE-UniCon-Init.md`

---

## 3. Public cleanup recommendations

The following items are good candidates for deletion or archival because they make the public repository look like a lab scratch directory rather than a polished release.

### 3.1 Safe delete candidate

```text
prep/gen_prompts_synad.py.bak
```

Reason:
- backup artifact
- not a canonical entry point
- should not live in the public repository root tree

### 3.2 Archive instead of delete

```text
eval/ucn_eval/Compare_EX1_EX2.sh
eval/ucn_eval/EX*.sh
eval/ucn_eval/eval_unicontrol_waymo_old1.py
eval/ucn_eval/poin.py
eval/ucn_eval/YOLOP.ipynb
```

Reason:
- these are experiment-specific or legacy convenience files
- they are useful for lab history
- but they should not dominate the public-facing tree

### 3.3 Typo / duplicate asset candidate

```text
figs/ualitative_pam.png
```

Reason:
- appears to be a typo-duplicated figure next to the canonical `figs/qualitative_pam.png`

---

## 4. What should remain visible in the public root

These should stay visible and documented:

```text
Public-facing core
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── docs/
├── environment/
├── scripts/verify_env.sh
├── prep/
├── Uni-ControlNet/
├── DGInStyle/
├── eval/ucn_eval/
└── figs/
```

The goal is **not** to hide the real workflow.
The goal is to make the workflow understandable immediately.

---

## 5. Figure assets already strong enough for the first public release

The current `figs/` directory already contains the main materials needed for a strong v1 Pages release:

- `従来水増しとの比較.png`
- `pipeline_multicondition.png`
- `model_detail.png`
- `pam_architecture.png`
- `impact_pretrain.png`
- `qualitative_tune.png`
- `qualitative_tune_拡大PAM動機.png`
- `qualitative_pam.png`
- `qualitative_pam_拡大PAMによる改善示し図.png`
- `スケーリング1.png`
- `スケーリング2.png`
- `evaluation_構造保持.png`
- `リアリティ_例.png`
- `多様性_例.png`
- `テキスト追従_例.png`

So the first public GitHub + Pages release can already look strong without waiting for new figures.

---

## 6. Optional extra images to add later for even stronger Pages impact

These are **optional**, not required for the first release.

If you later want dedicated result-table images on the project page, prepare them with these exact filenames under `figs/`:

```text
figs/result_pretrain_ablation_table.png
figs/result_pam_ablation_table.png
figs/result_prior_comparison_table.png
```

These are purely for presentation.
The current Pages file already renders the tables in HTML, so the release does not depend on them.

---

## 7. Release order

Use `APPLY_ORDER.md` for the exact copy-paste application flow.
