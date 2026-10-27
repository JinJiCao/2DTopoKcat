# 2DTopoKcat

2DTopoKcat predicts enzyme kinetic parameters from protein sequences, ligand structures, and protein contact topology. The current release focuses on model evaluation and result reproduction through the unified `test.py` entry point.

The test script supports:

- kcat evaluation with the `log10kcat_max` target;
- Km evaluation with the `log10km_mean` target.

The script loads a previously trained checkpoint. It does not contain an optimizer, backpropagation, epoch loops, or checkpoint-selection logic.

## Project Structure

```text
2DTopoKcat/
├── test.py
├── models/
├── utils/
├── data_examples/
├── substrate_selector_cofactor.py
├── environment.yml
└── README.md
```

- `test.py`: unified kcat and Km evaluation entry point.
- `models/`: model architecture and neural-network layers.
- `utils/`: data loading and graph-processing utilities.
- `data_examples/`: example test-data formats.
- `substrate_selector_cofactor.py`: main-substrate selection for multicomponent SMILES.
- `environment.yml`: runtime environment definition.

## Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate 2dtopokcat
```

Evaluation requires PyTorch, PyTorch Geometric, RDKit, NumPy, Pandas, SciPy, and scikit-learn.

## Required Files

Prepare the following files before running an evaluation:

1. A test CSV file.
2. A ligand-feature cache.
3. A protein-feature and contact-graph cache.
4. A trained `best_model.pt` checkpoint.

The evaluation script does not train a model, generate a checkpoint, or use the test set to select a checkpoint.

## kcat Test Data

The kcat test CSV must contain:

| Column | Description |
| --- | --- |
| `sequence` | Protein amino-acid sequence |
| `reactant_smiles` | Reactant or substrate SMILES |
| `log10kcat_max` | Evaluation target |

Example:

```text
data_examples/kcat/kcat-random_test.csv
```

## Km Test Data

The Km test CSV must contain:

| Column | Description |
| --- | --- |
| `sequence` | Protein amino-acid sequence |
| `substrate_smiles` | Substrate SMILES |
| `log10km_mean` | Evaluation target |

Example:

```text
data_examples/km/km-random_test.csv
```

## Run kcat Evaluation

```bash
python test.py \
  --task kcat \
  --test-csv /path/to/kcat-random_test.csv \
  --ligand-cache /path/to/ligand_dict_cofactor_filtered.pt \
  --protein-cache /path/to/protein_dict_esmc6b_multilayer_esm2contact.pt \
  --checkpoint /path/to/best_model.pt \
  --output-dir /path/to/kcat_test_results
```

## Run Km Evaluation

```bash
python test.py \
  --task km \
  --test-csv /path/to/Km_test.csv \
  --ligand-cache /path/to/ligand_dict.pt \
  --protein-cache /path/to/protein_dict_esmc6b_multilayer_esm2contact.pt \
  --checkpoint /path/to/best_model.pt \
  --output-dir /path/to/km_test_results
```

## Output Files

Each evaluation produces:

- `processed_test.csv`: test data after sequence validation and main-substrate selection;
- `test_predictions.csv`: targets, predictions, and predictive uncertainty;
- `test_summary.csv`: RMSE, MAE, R-squared, Pearson correlation, and Spearman correlation.

## Evaluation Procedure

The script performs the following operations:

1. Validates the required test-CSV columns.
2. Verifies that protein sequences contain only standard amino acids.
3. Selects the main substrate from multicomponent SMILES.
4. Loads the ligand and protein caches.
5. Builds residue, contact-edge, and triangle topology.
6. Loads the specified checkpoint.
7. Runs predictions on the test set and computes metrics.
8. Saves predictions and summary metrics.

The test set is used only for final evaluation.
