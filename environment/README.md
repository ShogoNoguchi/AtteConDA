# Environment Specification

This directory contains the full reproducible environment specification.

## Files

- `environment.yaml`
  - Primary environment specification.
  - Recommended installation method.

- `linux-64-conda-explicit.txt`
  - Linux-only explicit lock file for conda packages.

- `pip-requirements.txt`
  - pip-only portable requirements.

- `pip-freeze.txt`
  - Full audit record of pip packages.

## Installation Modes

### 1. Standard (recommended)

```bash
conda env create -f environment.yaml
conda activate atteconda_env
```

### 2. Strict Linux reproduction

```bash
conda create -n atteconda_env --file linux-64-conda-explicit.txt
conda activate atteconda_env
pip install -r pip-requirements.txt
```

## Important Constraints

- Do NOT use local editable installs (`-e /home/...`)
- All dependencies must be portable
- This environment was validated on:

  - Ubuntu 24.04.3
  - RTX 5090
  - CUDA 12.8
  - PyTorch 2.10.0.dev+cu128
  - diffusers 0.36.0