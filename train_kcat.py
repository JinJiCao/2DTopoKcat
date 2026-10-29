import json
import os
from collections import Counter


CPU_THREADS = max(1, int(os.environ.get("CPU_THREADS", "16")))
for thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
):
    os.environ[thread_variable] = str(CPU_THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from substrate_selector_cofactor import select_main_substrate
from models.bayesian_layers import BayesianMLP
from models.net_esmc_multilayer_tdl_contact_adapter import (
    ESMCMultiLayerTDLContactModel,
)
from training import (
    KcatData,
    KcatDataset,
    SingleModelTrainer,
    set_seed,
)
from train_original_kcat_seed42 import (
    _load_original_caches,
)
from utils.utils import DataLoader


torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(min(2, CPU_THREADS))
except RuntimeError:
    pass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


SEED = int(os.environ.get("MODEL_SEED", "42"))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "kcat")
EPOCHS = 50
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "30"))
TARGET_COLUMN = "log10kcat_max"
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
RUN_NAME = os.environ.get(
    "KCAT_RUN_NAME",
    "single_model_esmc_multilayer_tdl_contact_adapter_log10kcat_max",
)
PREDICTIONS_FILENAME = os.environ.get(
    "KCAT_PREDICTIONS_FILENAME",
    "single_model_predictions_esmc_tdl_log10kcat_max.csv",
)
SUMMARY_FILENAME = os.environ.get(
    "KCAT_SUMMARY_FILENAME",
    "single_model_summary_esmc_tdl_log10kcat_max.csv",
)
PROTEIN_CONTACT_GRAPH_LABEL = os.environ.get(
    "PROTEIN_CONTACT_GRAPH_LABEL",
    "ESM2-650M_return_contacts_threshold_0.9",
)


def _read_kcat_splits(data_dir):

    paths = {
        "train": os.path.join(data_dir, "kcat-random_train.csv"),
        "validation": os.path.join(data_dir, "kcat-random_val.csv"),
        "test": os.path.join(data_dir, "kcat-random_test.csv"),
    }
    required = {"sequence", "reactant_smiles", TARGET_COLUMN}
    frames = {}
    for split_name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"kcat {split_name} split not found: {path}")
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{split_name} CSV missing columns: {sorted(missing)}"
            )
        frames[split_name] = frame
    return paths, frames


def _validate_and_normalize_sequence(sequence):

    if pd.isna(sequence):
        return "", ["<MISSING>"]
    normalized = str(sequence).strip().upper()
    if not normalized:
        return "", ["<EMPTY>"]
    invalid_residues = sorted(set(normalized).difference(STANDARD_AMINO_ACIDS))
    return normalized, invalid_residues


def _build_triangle_index(edge_index, num_nodes):

    adjacency = [set() for _ in range(int(num_nodes))]
    edges = edge_index.detach().cpu().long()
    if edges.numel() > 0:
        for u, v in zip(edges[0].tolist(), edges[1].tolist()):
            if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)

    triangles = []
    max_triangles = int(os.environ.get("MAX_TDL_TRIANGLES", "0"))
    for u in range(int(num_nodes)):
        for v in sorted(x for x in adjacency[u] if x > u):
            for w in sorted(adjacency[u].intersection(adjacency[v])):
                if w > v:
                    triangles.append((u, v, w))
                    if max_triangles > 0 and len(triangles) >= max_triangles:
                        return torch.tensor(triangles, dtype=torch.long).t().contiguous()
    if not triangles:
        return torch.empty((3, 0), dtype=torch.long)
    return torch.tensor(triangles, dtype=torch.long).t().contiguous()


class KcatTDLData(KcatData):


    def __inc__(self, key, value, *args, **kwargs):
        if key == "prot_triangle_index":
            return self.prot_node_aa.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class KcatTDLDataset(KcatDataset):


    def __init__(self, csv_path, mol_dict, prot_dict, device="cpu"):
        super().__init__(csv_path, mol_dict, prot_dict, device=device)
        sequences = list(dict.fromkeys(self.pairs["sequence"].tolist()))
        missing = [
            sequence for sequence in sequences
            if "_tdl_triangle_index" not in self.prots[sequence]
        ]
        for sequence in tqdm(missing, desc="Building enzyme triangles", leave=False):
            protein = self.prots[sequence]
            protein["_tdl_triangle_index"] = _build_triangle_index(
                protein["edge_index"], protein["seq_feat"].size(0)
            )
        triangle_count = sum(
            self.prots[sequence]["_tdl_triangle_index"].size(1)
            for sequence in sequences
        )
        print(
            f"TDL complex: proteins={len(sequences)}, "
            f"triangles={triangle_count}"
        )

    def __getitem__(self, idx):
        original = super().__getitem__(idx)
        sequence = self.pairs.iloc[idx]["sequence"]
        values = original.to_dict()
        values["prot_triangle_index"] = self.prots[sequence][
            "_tdl_triangle_index"
        ]
        return KcatTDLData(**values)


class TDLTrainer(SingleModelTrainer):
    def _forward(self, data):
        data = data.to(self.device)
        fingerprint = data.fingerprint if hasattr(data, "fingerprint") else None
        return self.model(
            mol_x=data.mol_x,
            mol_x_feat=data.mol_x_feat,
            bond_x=data.mol_edge_attr,
            atom_edge_index=data.mol_edge_index,
            clique_x=data.clique_x,
            clique_edge_index=data.clique_edge_index,
            atom2clique_index=data.atom2clique_index,
            residue_x=data.prot_node_aa,
            residue_evo_x=data.prot_node_evo,
            residue_edge_index=data.prot_edge_index,
            residue_edge_weight=data.prot_edge_weight,
            residue_triangle_index=data.prot_triangle_index,
            fingerprint=fingerprint,
            mol_batch=data.mol_x_batch,
            prot_batch=data.prot_node_aa_batch,
            clique_batch=data.clique_x_batch,
        ), data


def _loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        follow_batch=["mol_x", "clique_x", "prot_node_aa"],
    )


def _create_model(config, device, esmc_shape):
    mol_deg = torch.zeros(10, dtype=torch.long)
    mol_deg[2] = 1000
    prot_deg = torch.zeros(10, dtype=torch.long)
    prot_deg[2] = 2000
    params = config["params"]
    model = ESMCMultiLayerTDLContactModel(
        mol_deg,
        prot_deg,
        mol_in_channels=params["mol_in_channels"],
        prot_in_channels=33,
        prot_evo_channels=esmc_shape[2],
        hidden_channels=params["hidden_channels"],
        pre_layers=params["pre_layers"],
        post_layers=params["post_layers"],
        aggregators=params["aggregators"],
        scalers=params["scalers"],
        total_layer=params["total_layer"],
        K=params["K"],
        heads=params["heads"],
        dropout=params["dropout"],
        dropout_attn_score=params["dropout_attn_score"],
        regression_head=True,
        classification_head=False,
        multiclassification_head=0,
        device=device,
        use_rope=True,
        use_residual_gate=True,
        num_esmc_layers=esmc_shape[1],
    ).to(device)
    hidden = params["hidden_channels"]
    model.reg_out = BayesianMLP([hidden * 4, hidden, 1]).to(device)
    return model


def _predict_single_model(model, data_loader, device, n_samples=30):
    model.eval()
    all_predictions, all_stds, all_targets = [], [], []
    trainer = TDLTrainer(model, device=device)
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Predicting", leave=False):
            samples = []
            for _ in range(n_samples):
                (prediction, _, _, _, _, _, _, _, _), moved = trainer._forward(data)
                samples.append(prediction.reshape(-1).cpu().numpy())
            samples = np.stack(samples, axis=0)
            all_predictions.append(samples.mean(axis=0))
            all_stds.append(samples.std(axis=0))
            all_targets.append(moved.reg_y.reshape(-1).cpu().numpy())
    return (
        np.concatenate(all_predictions),
        np.concatenate(all_stds),
        np.concatenate(all_targets),
    )


def _load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _prepare_cofactor_filtered_splits(data_dir, frames):
    output_dir = os.environ.get(
        "COFACTOR_FILTERED_SPLIT_DIR",
        os.path.join(
            data_dir,
            "cofactor_filtered_splits_log10kcat_max",
        ),
    )
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}
    output_frames = {}
    audit_parts = []
    sequence_filter_records = []
    filename_by_split = {
        "train": "kcat-random_train.csv",
        "validation": "kcat-random_val.csv",
        "test": "kcat-random_test.csv",
    }

    for split_name, frame in frames.items():
        processed = frame.copy()
        if "sequence" not in processed.columns:
            raise ValueError(f"{split_name} CSV missing sequence column")

        normalized_sequences = []
        valid_sequence_mask = []
        for source_row_index, sequence in enumerate(processed["sequence"].tolist()):
            normalized, invalid_residues = _validate_and_normalize_sequence(
                sequence
            )
            normalized_sequences.append(normalized)
            is_valid = not invalid_residues
            valid_sequence_mask.append(is_valid)
            if not is_valid:
                sequence_filter_records.append({
                    "split": split_name,
                    "source_row_index": source_row_index,
                    "sequence": sequence,
                    "sequence_length": len(normalized),
                    "invalid_residues": ",".join(invalid_residues),
                })

        input_count = len(processed)
        processed["sequence"] = normalized_sequences
        processed = processed.loc[valid_sequence_mask].copy().reset_index(drop=True)
        removed_count = input_count - len(processed)
        print(
            f"Standard-amino-acid filter [{split_name}]: "
            f"kept={len(processed)}, removed={removed_count}, "
            f"input={input_count}"
        )
        if processed.empty:
            raise RuntimeError(
                f"No {split_name} samples remain after sequence filtering"
            )

        if TARGET_COLUMN not in processed.columns:
            raise ValueError(
                f"{split_name} CSV missing target column: {TARGET_COLUMN}"
            )
        target = pd.to_numeric(processed[TARGET_COLUMN], errors="raise")
        if target.isnull().any() or not np.isfinite(target.to_numpy()).all():
            raise ValueError(
                f"{split_name} CSV has invalid values in {TARGET_COLUMN}"
            )
                                                                              
                                                                         
        processed["log10_value"] = target.astype(float)
        selected = []
        records = []
        for row_index, reaction_smiles in enumerate(
            processed["reactant_smiles"].tolist()
        ):
            result = select_main_substrate(reaction_smiles)
            selected.append(result.selected_smiles)
            record = result.audit_record()
            record.update({"split": split_name, "row_index": row_index})
            records.append(record)

        processed.insert(
            processed.columns.get_loc("reactant_smiles") + 1,
            "reactant_smiles_original",
            processed["reactant_smiles"],
        )
        processed["reactant_smiles"] = selected
        output_path = os.path.join(output_dir, filename_by_split[split_name])
        processed.to_csv(output_path, index=False)
        output_paths[split_name] = output_path
        output_frames[split_name] = processed
        audit_parts.append(pd.DataFrame(records))

    audit = pd.concat(audit_parts, ignore_index=True)
    audit_path = os.path.join(output_dir, "substrate_selection_audit.csv")
    audit.to_csv(audit_path, index=False)
    sequence_filter_audit_path = os.path.join(
        output_dir, "nonstandard_sequence_filter_audit.csv"
    )
    pd.DataFrame(
        sequence_filter_records,
        columns=[
            "split", "source_row_index", "sequence", "sequence_length",
            "invalid_residues",
        ],
    ).to_csv(sequence_filter_audit_path, index=False)
    summary = Counter(audit["selection_strategy"])
    removed_count = int((audit["removed_cofactors"].str.len() > 0).sum())
    print("Cofactor-aware substrate selection:")
    for strategy, count in sorted(summary.items()):
        print(f"  {strategy}: {count}")
    print(f"  rows with removed known cofactors: {removed_count}")
    print(f"  audit: {audit_path}")
    print(
        "Non-standard sequence filtering: "
        f"removed={len(sequence_filter_records)}; "
        f"audit={sequence_filter_audit_path}"
    )
    return (
        output_paths,
        output_frames,
        audit_path,
        sequence_filter_audit_path,
    )


def _load_cofactor_filtered_ligands(
    data_dir, frames, reusable_ligand_dict, device
):
    cache_path = os.environ.get(
        "COFACTOR_FILTERED_LIGAND_CACHE",
        os.path.join(data_dir, "cache", "ligand_dict_cofactor_filtered.pt"),
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    ligand_dict = {}
    if os.path.isfile(cache_path):
        print(f"Loading cofactor-filtered ligand cache: {cache_path}")
        ligand_dict = torch.load(cache_path, weights_only=False)

    required_smiles = sorted(set().union(*(
        set(frame["reactant_smiles"].tolist()) for frame in frames.values()
    )))
    reused = 0
    for smiles in required_smiles:
        if smiles not in ligand_dict and smiles in reusable_ligand_dict:
            ligand_dict[smiles] = reusable_ligand_dict[smiles]
            reused += 1

    missing = sorted(set(required_smiles).difference(ligand_dict))
    print(
        f"Filtered ligand features: required={len(required_smiles)}, "
        f"reused_now={reused}, missing={len(missing)}"
    )
    if missing:
        from utils.chemberta_init_local import chemberta_ligand_init

        generated = chemberta_ligand_init(missing)
        failed = set(missing).difference(generated)
        if failed:
            raise RuntimeError(
                f"ChemBERTa failed for {len(failed)} selected substrates"
            )
        ligand_dict.update(generated)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

                                                                              
    ligand_dict = {smiles: ligand_dict[smiles] for smiles in required_smiles}
    temporary_path = cache_path + ".tmp"
    torch.save(ligand_dict, temporary_path)
    os.replace(temporary_path, cache_path)
    print(f"Saved cofactor-filtered ligand cache: {cache_path}")
    return ligand_dict, cache_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.environ.get(
        "KCAT_MAX_DATA_DIR",
        os.environ.get("ORIGINAL_KCAT_DATA_DIR", DEFAULT_DATA_DIR),
    )
    print(
        f"CPU threads={CPU_THREADS}; device={device}; epochs={EPOCHS}; "
        f"batch={BATCH_SIZE}; MC samples={MC_SAMPLES}; "
        f"target={TARGET_COLUMN}; seed={SEED}"
    )

    with open(os.path.join(project_dir, "config.json"), "r") as handle:
        config = json.load(handle)

    _, original_frames = _read_kcat_splits(data_dir)
    (
        split_paths,
        filtered_frames,
        audit_path,
        sequence_filter_audit_path,
    ) = _prepare_cofactor_filtered_splits(data_dir, original_frames)

    if os.environ.get("AUDIT_ONLY", "0") == "1":
        print("AUDIT_ONLY=1: preprocessing completed; training was not started")
        return

    protein_dict, reusable_ligand_dict = _load_original_caches(
        project_dir, data_dir, filtered_frames
    )
    ligand_dict, ligand_cache_path = _load_cofactor_filtered_ligands(
        data_dir, filtered_frames, reusable_ligand_dict, device
    )
    del reusable_ligand_dict

    first_protein = next(iter(protein_dict.values()))
    esmc_shape = first_protein["esmc_multi_layer"].shape
    if len(esmc_shape) != 3 or esmc_shape[1] != 4:
        raise RuntimeError(
            f"Expected ESMC cache [L,4,D], got {tuple(esmc_shape)}"
        )
    selected_layers = first_protein["selected_layers"]

    train_dataset = KcatTDLDataset(
        split_paths["train"], ligand_dict, protein_dict, device=device
    )
    validation_dataset = KcatTDLDataset(
        split_paths["validation"], ligand_dict, protein_dict, device=device
    )
    test_dataset = KcatTDLDataset(
        split_paths["test"], ligand_dict, protein_dict, device=device
    )
    print(
        f"log10kcat_max TDL single model: Train={len(train_dataset)}, "
        f"Validation={len(validation_dataset)}, Test={len(test_dataset)}"
    )

    output_dir = os.environ.get(
        "KCAT_OUTPUT_DIR",
        os.path.join(project_dir, "outputs", RUN_NAME),
    )
    set_seed(SEED)
    train_loader = _loader(train_dataset, shuffle=True)
    validation_loader = _loader(validation_dataset, shuffle=False)
    test_loader = _loader(test_dataset, shuffle=False)
    model = _create_model(config, device, esmc_shape)
    model.reset_parameters()
    trainer = TDLTrainer(
        model, device=device, ranking_weight=0.1, beta_kl=1e-4
    )
    result = trainer.fit(
        train_loader,
        validation_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=1e-4,
        weight_decay=1e-5,
        checkpoint_dir=output_dir,
        test_samples=MC_SAMPLES,
    )

    predictions_path = os.path.join(output_dir, "test_predictions.csv")
    predictions_frame = pd.read_csv(predictions_path).rename(columns={
        "y_true": f"{TARGET_COLUMN}_true",
        "y_pred": f"{TARGET_COLUMN}_pred",
    })
    predictions_frame.to_csv(predictions_path, index=False)

    test_metrics = result["test_metrics"]
    summary = {
        "dataset": "kcat_random_split_cofactor_filtered_log10kcat_max",
        "target": TARGET_COLUMN,
        "input_target_transformed_again": False,
        "model_count": 1,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "mc_samples": MC_SAMPLES,
        "best_epoch_by_validation": result["best_epoch"],
        "best_validation_rmse": result["best_validation_rmse"],
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "substrate_selection_audit": audit_path,
        "sequence_filter_audit": sequence_filter_audit_path,
        "ligand_cache": ligand_cache_path,
        "protein_embedding": f"ESMC-6B multi-layer {selected_layers}",
        "protein_contact_graph": PROTEIN_CONTACT_GRAPH_LABEL,
        "protein_topology": "TDL residues/contacts/3-clique triangles",
    }
    summary_path = os.path.join(output_dir, "test_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("=" * 70)
    print(f"Best validation epoch={result['best_epoch']}")
    print(f"Best validation RMSE={result['best_validation_rmse']:.4f}")
    print(f"Test RMSE={test_metrics['rmse']:.4f}")
    print(f"Test R2={test_metrics['r2']:.4f}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

