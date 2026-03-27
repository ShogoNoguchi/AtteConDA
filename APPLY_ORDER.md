# APPLY_ORDER.md

This file explains exactly how to apply the release bundle to your local repository.

---

## 0. Assumed local repository path

The commands below assume your repository is here:

```bash
/data/coding/B_thesis_Repo
```

If your path is different, replace it once and keep the rest unchanged.

---

## 1. Create a safety branch

```bash
cd /data/coding/B_thesis_Repo
git checkout -b release/public-repo-hardening
```

---

## 2. Unzip the release bundle

Assume the downloaded bundle is:

```text
/path/to/AtteConDA_release_bundle.zip
```

Then run:

```bash
rm -rf /tmp/AtteConDA_release_bundle
mkdir -p /tmp/AtteConDA_release_bundle
unzip /path/to/AtteConDA_release_bundle.zip -d /tmp/AtteConDA_release_bundle
```

---

## 3. Copy the bundle into the repository root

```bash
rsync -av /tmp/AtteConDA_release_bundle/AtteConDA_release_bundle/ /data/coding/B_thesis_Repo/
```

---

## 4. Make scripts executable

```bash
cd /data/coding/B_thesis_Repo
chmod +x scripts/verify_env.sh
chmod +x REPO_CLEANUP_COMMANDS.sh
```

---

## 5. Verify the environment entry point

```bash
cd /data/coding/B_thesis_Repo
conda activate atteconda_env
bash scripts/verify_env.sh atteconda_env
```

---

## 6. Optional public-tree cleanup

This archives old experiment scripts and removes the obvious backup artifact.

```bash
cd /data/coding/B_thesis_Repo
bash REPO_CLEANUP_COMMANDS.sh
```

If you want to inspect the script first:

```bash
sed -n '1,240p' REPO_CLEANUP_COMMANDS.sh
```

---

## 7. Review what changed

```bash
cd /data/coding/B_thesis_Repo
git status
```

Good signs:

- `README.md` replaced
- `.gitignore` replaced
- `LICENSE` added
- `THIRD_PARTY_NOTICES.md` added
- `CITATION.cff` added
- `docs/` added
- `.github/workflows/pages.yml` added
- `scripts/verify_env.sh` added
- `huggingface_model_cards/` added

---

## 8. Commit the public-release hardening pass

```bash
cd /data/coding/B_thesis_Repo
git add -A
git commit -m "Polish public release: README, Pages, licensing, notices, model cards"
```

---

## 9. Push

```bash
cd /data/coding/B_thesis_Repo
git push -u origin release/public-repo-hardening
```

After review and merge:

```bash
git checkout main
git pull
git merge --ff-only release/public-repo-hardening
git push origin main
```

---

## 10. Enable GitHub Pages

On GitHub:

1. open the repository
2. go to **Settings**
3. go to **Pages**
4. set **Build and deployment -> Source** to **GitHub Actions**

The workflow file in this bundle will then publish the page automatically.

---

## 11. Publish the Hugging Face model cards

The bundle includes one Markdown file per released model under:

```text
huggingface_model_cards/
```

For each public Hugging Face model repo:

1. open the model page
2. edit `README.md`
3. paste the corresponding file content
4. save

Mapping:

```text
AtteConDA-SDE-Scratch-30K    <- huggingface_model_cards/AtteConDA-SDE-Scratch-30K.md
AtteConDA-SDE-UniCon-30K     <- huggingface_model_cards/AtteConDA-SDE-UniCon-30K.md
AtteConDA-SDE-UniCon-60K     <- huggingface_model_cards/AtteConDA-SDE-UniCon-60K.md
AtteConDA-SDE-UniCon-90K     <- huggingface_model_cards/AtteConDA-SDE-UniCon-90K.md
AtteConDA-SDE-UniCon-60K-PAM <- huggingface_model_cards/AtteConDA-SDE-UniCon-60K-PAM.md
AtteConDA-SDE-UniCon-Init    <- huggingface_model_cards/AtteConDA-SDE-UniCon-Init.md
```

---

## 12. Final smoke check after merge

```bash
cd /data/coding/B_thesis_Repo

bash scripts/verify_env.sh atteconda_env

python - <<'PY'
from pathlib import Path
required = [
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "docs/index.html",
    "docs/RESEARCHER_README.md",
    ".github/workflows/pages.yml",
    "scripts/verify_env.sh",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")
print("All release files are present.")
PY
```
