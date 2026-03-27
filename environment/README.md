# Environment notes for AtteConDA

This directory is the environment entry point for the public release.

---

## Design policy

The main AtteConDA training machine is an **RTX 5090** and the repository is intentionally aligned with **CUDA 12.8 / cu128**.
For that reason, the repository should **not** silently drift to a different CUDA build.

At the same time, the public release should be honest about what is already validated and what is still a future cleanup target.

So the policy is:

1. **Keep the author-validated environment file reproducible.**
2. **Document it clearly.**
3. **Do not silently replace working cu128 settings with a different stack just for cosmetic reasons.**

---

## What is currently in `environment.yaml`

The current file is the author-validated environment and is intentionally preserved because it was already used on the RTX 5090 machine.

In particular, the current file includes:

- Python 3.11
- CUDA 12.8 user-space packages
- cu128 PyTorch / torchvision / torchaudio
- diffusers 0.36.0
- pytorch-lightning 1.9.5

This is the right choice for **exact reproducibility** of the current lab machine.

---

## Quick start

```bash
git clone https://github.com/ShogoNoguchi/AtteConDA.git
cd AtteConDA
conda env create -f environment/environment.yaml
conda activate atteconda_env
bash scripts/verify_env.sh atteconda_env
```

---

## Verified machine profile used for this release

This repository was organized around the following validated environment profile:

- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 5090
- CUDA build target: 12.8 / cu128
- PyTorch stack: cu128
- diffusers: 0.36.0

The exact pinned versions remain in `environment.yaml`.

---

## Why this README does **not** silently replace the file with a different stack

For RTX 5090 / Blackwell-class hardware, the most important requirement is that the installed PyTorch build actually matches **cu128** and works reliably on the target machine.

The existing repository already satisfies that.
So for the public release, the safest action is:

- keep the validated file,
- add verification tooling,
- explain the reasoning clearly,
- and only introduce a new stable variant after a full train / infer / eval validation pass.

That is more honest than claiming a new environment file has been validated when it has not.

---

## Optional future cleanup: a stable-only PyTorch trio

If you later decide to publish a second environment variant that uses only officially stable PyTorch wheels, do it as a **separate** file after validation instead of overwriting the current working file.

A good pattern is:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

Important:

- validate **training**
- validate **Waymo inference**
- validate **evaluation Docker overlay**
- validate **xformers compatibility**
- validate **pytorch-lightning compatibility**

before making that stable-only variant the default.

---

## Repository-side verification

Use the repository verification script after every fresh installation:

```bash
bash scripts/verify_env.sh atteconda_env
```

The script checks:

- Python version
- torch / torchvision / torchaudio versions
- CUDA availability
- GPU name
- `torch.version.cuda`
- diffusers
- pytorch-lightning
- transformers
- open_clip
- onnxruntime
- xformers availability

It exits with a non-zero status if CUDA is unavailable.

---

## What to avoid

### 1. Do not install random packages into the system Python

Always use the conda environment or the Docker overlay flow.

### 2. Do not accidentally downgrade the cu128 build

A casual `pip install torch ...` without the proper cu128 index can break the RTX 5090 setup.

### 3. Do not rebuild the Docker image unnecessarily

The project policy is to prefer:

- the common NVIDIA CUDA 12.8 runtime base image
- pip overlay caches under `/data`
- and reproducible runtime commands

instead of repeatedly rebuilding the image for small package changes.

---

## Related files

- `../scripts/verify_env.sh`
- `../README.md`
- `../docs/RESEARCHER_README.md`
