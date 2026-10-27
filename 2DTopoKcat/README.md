# 2DTopoKcat/Km

This repository contains the training implementation used for two enzyme
kinetics regression tasks:

- **kcat (v66):** predicts the supplied `log10kcat_max` label.
- **Km (v67):** predicts the supplied `log10km_mean` label.

Both tasks use the same multimodal architecture: four ESMC-6B residue layers,
an ESM2-650M contact graph, a point/edge/triangle topological deep learning
(TDL) encoder, atom-level and junction-tree ligand graphs, enzyme-ligand
interaction blocks, and a Bayesian MLP regression head. The training scripts
fit ten seeds by default and average their predictions as an ensemble.

## Repository layout

```text
.
├── train_kcat.py                  # short entry point for v66
├── train_km.py                    # short entry point for v67
├── train_ensemble_v66_*.py        # full kcat training/evaluation code
├── train_ensemble_v67_*.py        # full Km training/evaluation code
├── train_ensemble_v36_*.py        # shared dataset/trainer implementation
├── train_v43_*.py                 # shared cache construction helpers
├── substrate_selector_cofactor.py # main-substrate/cofactor selection
├── models/                        # Bayesian head, PNA, TDL and interaction model
├── utils/                         # protein/ligand featurization and ESM2 contacts
├── data_examples/                 # CSV schema examples only
├── scripts/download_models.sh
├── scripts/export_environment.sh
├── environment.yml
└── config.json
```

Large data files, pretrained weights, generated feature caches and experiment
checkpoints are intentionally excluded by `.gitignore`.

## Hardware

The code is intended for Linux with an NVIDIA GPU. Initial ESMC-6B feature
generation is the most demanding step and requires a GPU with BF16 support.
Disk use can be large because residue-level multi-layer features are cached for
every unique enzyme. Training memory also depends strongly on batch size and
the longest sequence in a batch.

The original experiments used Python 3.12, `BATCH_SIZE=32`, `MC_SAMPLES=30`,
and ten seeds (`0` through `9`). If CUDA memory is insufficient, reduce the
batch size before changing the model.

## Installation

### 1. Create the base environment

```bash
conda env create -f environment.yml
conda activate 2dtopokcat
```

PyTorch Geometric binary extensions must match the exact PyTorch and CUDA
versions. Check them first:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

Then install the matching PyG wheels. For example, for PyTorch 2.5.1 and CUDA
12.1:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric
```

Do not copy the example URL blindly if your PyTorch/CUDA versions differ; use
the matching wheel index listed by PyTorch Geometric.

### 2. Download pretrained models

```bash
bash scripts/download_models.sh
```

This downloads:

- `biohub/ESMC-6B` into `pretrained/ESMC-6B`;
- `DeepChem/ChemBERTa-77M-MLM` into `pretrained/ChemBERTa-77M-MLM`;
- Fair-ESM ESM2-650M base and contact-regression checkpoints into
  `pretrained/ESM2`;
- Fair-ESM into `third_party/fair_esm_vendor`.

Fair-ESM is deliberately vendored. Biohub/EvolutionaryScale ESM and Fair-ESM
both import as `esm`, but expose different APIs; installing both into the same
site-packages directory can make ESM2 contact generation import the wrong
package.

Model files are not redistributed in this Git repository. Users must review
and accept the licenses and acceptable-use terms on the original model pages.

### 3. Verify imports

```bash
python scripts/check_install.py
```

## Data format

All labels are already in base-10 logarithmic space. The code does **not** take
another logarithm.

### kcat v66

Place the following files in one directory:

```text
kcat-random_train.csv
kcat-random_val.csv
kcat-random_test.csv
```

Required columns:

```text
sequence,reactant_smiles,log10kcat_max
```

### Km v67

Place the following files in one directory:

```text
km-random_train.csv
km-random_val.csv
km-random_test.csv
```

Required columns:

```text
sequence,substrate_smiles,log10km_mean
```

Small schema-only examples are under `data_examples/`. They are not meaningful
training datasets.

For both tasks, sequences containing non-canonical amino-acid symbols are
filtered. Multi-component SMILES are processed by the cofactor-aware substrate
selector; single-component SMILES are retained unchanged.

## Configuration

Copy `.env.example` to a private shell configuration or export the variables
directly. The training scripts do not automatically parse `.env` files.

```bash
export ESMC6B_MODEL_PATH="$PWD/pretrained/ESMC-6B"
export ESM2_MODEL_PATH="$PWD/pretrained/ESM2"
export CHEMBERTA_MODEL_PATH="$PWD/pretrained/ChemBERTa-77M-MLM"
export FAIR_ESM_VENDOR_PATH="$PWD/third_party/fair_esm_vendor"
```

Important runtime variables:

| Variable | Default | Meaning |
|---|---:|---|
| `CPU_THREADS` | 16 | CPU/thread-pool limit |
| `EPOCHS` | 100 | Epochs per ensemble member |
| `BATCH_SIZE` | 32 | Samples per batch |
| `MC_SAMPLES` | 30 | Bayesian forward samples during final evaluation |
| `MODEL_SEEDS` | `0,...,9` | Comma-separated ensemble seeds |
| `MAX_TDL_TRIANGLES` | 0 | Per-protein triangle cap; 0 means unlimited |
| `DELETE_SOURCE_CACHES_AFTER_MERGE` | 0 | Whether v66 deletes source feature caches |

## Training kcat

```bash
export KCAT_MAX_DATA_DIR=/absolute/path/to/kcat_data
export KCAT_OUTPUT_DIR="$PWD/outputs/kcat_v66"

CUDA_VISIBLE_DEVICES=0 \
CPU_THREADS=16 \
EPOCHS=100 \
BATCH_SIZE=32 \
MC_SAMPLES=30 \
MODEL_SEEDS=0,1,2,3,4,5,6,7,8,9 \
python train_kcat.py
```

## Training Km

```bash
export CATPRED_KM_DATA_DIR=/absolute/path/to/km_data
export KM_OUTPUT_DIR="$PWD/outputs/km_v67"

CUDA_VISIBLE_DEVICES=0 \
CPU_THREADS=16 \
EPOCHS=100 \
BATCH_SIZE=32 \
MC_SAMPLES=30 \
MODEL_SEEDS=0,1,2,3,4,5,6,7,8,9 \
python train_km.py
```

Each seed is trained independently. The best checkpoint is selected by the
training script's validation criterion, then the saved members are evaluated
and averaged. Outputs include per-seed checkpoints/predictions, ensemble
predictions, member metrics, summary CSV files and run metadata.

## Exporting the exact environment used for a paper

The supplied `environment.yml` is a portable starting point. To let others
reproduce the exact server environment, activate the environment that produced
the reported results and run:

```bash
conda activate 2dtopokcat
bash scripts/export_environment.sh
```

The script creates:

- `environment.history.yml`: direct Conda dependencies;
- `environment.lock.yml`: fuller Conda snapshot without build strings;
- `requirements-lock.txt`: installed pip package versions;
- `system-info.txt`: Python, CUDA, GPU and key package versions.

Inspect these files before committing them. Remove private filesystem paths,
credentials, internal package indexes and machine-specific prefixes. For a
paper release, commit the portable environment plus the reviewed lock files.

## Publishing trained weights and data

Do not commit multi-gigabyte `.pt`, `.safetensors`, caches or private datasets
directly to Git. GitHub's normal file limit is unsuitable for these artifacts.
Recommended options are:

1. upload final checkpoints to Zenodo or Hugging Face;
2. create a versioned GitHub Release that links to those files;
3. publish SHA-256 checksums and the exact code commit used for training;
4. document dataset provenance, access conditions and split files separately.

Only final model checkpoints are normally needed for inference. ESMC/ESM2 and
ChemBERTa weights are still needed if users must regenerate input caches.

## Reproducibility checklist

- Record the Git commit hash.
- Commit reviewed environment lock files.
- Record GPU model, CUDA and PyTorch versions.
- Publish the exact train/validation/test row lists or stable identifiers.
- Publish checkpoint SHA-256 hashes.
- State label units and confirm labels are already log10-transformed.
- Record ESMC, ESM2 and ChemBERTa model revisions.
- Record `MODEL_SEEDS`, batch size, epoch count and MC sample count.

## Provenance and licensing

The graph backbone and several utilities are adapted from
[PSICHIC](https://github.com/huankoh/PSICHIC), which is Apache-2.0 licensed.
See `THIRD_PARTY_NOTICES.md`.

No license has been assigned automatically to the new repository. The code
owner must add an appropriate `LICENSE` before public release and ensure that
the selected license is compatible with every adapted component and model.
