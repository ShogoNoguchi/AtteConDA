#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo "[AtteConDA] Public-tree cleanup helper"
echo "============================================================"
echo "[Info] Repository root: ${REPO_ROOT}"

mkdir -p archive/legacy_eval/ucn_eval
mkdir -p archive/legacy_figs

if [[ -e "prep/gen_prompts_synad.py.bak" ]]; then
  echo "[Action] Removing backup artifact: prep/gen_prompts_synad.py.bak"
  git rm -f prep/gen_prompts_synad.py.bak
else
  echo "[Skip] prep/gen_prompts_synad.py.bak not found"
fi

shopt -s nullglob

LEGACY_FILES=(
  eval/ucn_eval/Compare_EX1_EX2.sh
  eval/ucn_eval/EX*.sh
  eval/ucn_eval/eval_unicontrol_waymo_old1.py
  eval/ucn_eval/poin.py
  eval/ucn_eval/YOLOP.ipynb
)

for src in "${LEGACY_FILES[@]}"; do
  for f in ${src}; do
    if [[ -e "${f}" ]]; then
      echo "[Action] Archiving ${f}"
      git mv "${f}" archive/legacy_eval/ucn_eval/
    fi
  done
done

if [[ -e "figs/ualitative_pam.png" ]]; then
  echo "[Action] Archiving likely typo duplicate: figs/ualitative_pam.png"
  git mv "figs/ualitative_pam.png" "archive/legacy_figs/ualitative_pam.png"
else
  echo "[Skip] figs/ualitative_pam.png not found"
fi

echo "[Done] Cleanup suggestions have been applied to the Git working tree."
echo "[Next] Review with: git status"
