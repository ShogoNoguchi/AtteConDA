#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-}"

echo "============================================================"
echo "[AtteConDA] Environment verification"
echo "============================================================"

if [[ -n "${ENV_NAME}" ]]; then
  echo "[Info] Requested environment name: ${ENV_NAME}"
fi

if command -v conda >/dev/null 2>&1; then
  echo "[Info] conda executable: $(command -v conda)"
  CURRENT_ENV="${CONDA_DEFAULT_ENV:-<not activated>}"
  echo "[Info] active conda env: ${CURRENT_ENV}"
else
  echo "[Warn] conda was not found in PATH. Continuing with the current shell Python."
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[Error] python was not found in PATH."
  exit 1
fi

python - <<'PY'
import importlib
import json
import platform
import sys

def get_version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "<no __version__>")
    except Exception as exc:
        return f"<missing: {exc}>"

report = {
    "python_executable": sys.executable,
    "python_version": sys.version.replace("\n", " "),
    "platform": platform.platform(),
    "torch": get_version("torch"),
    "torchvision": get_version("torchvision"),
    "torchaudio": get_version("torchaudio"),
    "pytorch_lightning": get_version("pytorch_lightning"),
    "diffusers": get_version("diffusers"),
    "transformers": get_version("transformers"),
    "open_clip": get_version("open_clip"),
    "onnxruntime": get_version("onnxruntime"),
}

cuda_ok = False
cuda_report = {
    "cuda_available": False,
    "torch_cuda_version": None,
    "device_count": 0,
    "devices": [],
    "warning": None,
}

try:
    import torch  # type: ignore
    cuda_report["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    cuda_report["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        cuda_ok = True
        count = torch.cuda.device_count()
        cuda_report["device_count"] = int(count)
        for idx in range(count):
            props = torch.cuda.get_device_properties(idx)
            cuda_report["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "major": getattr(props, "major", None),
                    "minor": getattr(props, "minor", None),
                }
            )
    else:
        cuda_report["warning"] = "torch.cuda.is_available() returned False"
except Exception as exc:
    cuda_report["warning"] = f"Failed to query torch CUDA state: {exc}"

report["cuda"] = cuda_report

try:
    import xformers  # type: ignore
    report["xformers"] = getattr(xformers, "__version__", "<no __version__>")
except Exception as exc:
    report["xformers"] = f"<missing: {exc}>"

print(json.dumps(report, indent=2, ensure_ascii=False))

if not cuda_ok:
    raise SystemExit(2)
PY

STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
  echo "[Error] CUDA verification failed. AtteConDA expects a working CUDA runtime."
  exit "${STATUS}"
fi

echo "[OK] Environment verification finished successfully."
