#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT_DIR}/pretrained/ESMC-6B"
mkdir -p "${ROOT_DIR}/pretrained/ChemBERTa-77M-MLM"
mkdir -p "${ROOT_DIR}/pretrained/ESM2"
mkdir -p "${ROOT_DIR}/third_party/fair_esm_vendor"

python -m pip install "huggingface-hub[cli]"
hf download biohub/ESMC-6B \
  --local-dir "${ROOT_DIR}/pretrained/ESMC-6B"
hf download DeepChem/ChemBERTa-77M-MLM \
  --local-dir "${ROOT_DIR}/pretrained/ChemBERTa-77M-MLM"

python -m pip install fair-esm==2.0.0 --no-deps \
  --target "${ROOT_DIR}/third_party/fair_esm_vendor"

curl -L \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt \
  -o "${ROOT_DIR}/pretrained/ESM2/esm2_t33_650M_UR50D.pt"
curl -L \
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt \
  -o "${ROOT_DIR}/pretrained/ESM2/esm2_t33_650M_UR50D-contact-regression.pt"

echo "Model files downloaded under ${ROOT_DIR}/pretrained"
