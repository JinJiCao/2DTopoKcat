#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Activate the original Conda environment first." >&2
    exit 1
fi

conda env export --from-history | sed '/^prefix:/d' > environment.history.yml
conda env export --no-builds | sed '/^prefix:/d' > environment.lock.yml

python -m pip list --format=freeze > requirements-lock.txt

{
    echo "Exported: $(date -Iseconds)"
    echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
    python --version
    python - <<'PY'
import platform
print("Platform:", platform.platform())
for name in ("torch", "torch_geometric", "torch_sparse", "transformers", "esm", "rdkit"):
    try:
        module = __import__(name)
        print(f"{name}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass
PY
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    fi
} > system-info.txt

echo "Created environment.history.yml, environment.lock.yml, requirements-lock.txt and system-info.txt"
echo "Inspect them for private paths or credentials before uploading."
