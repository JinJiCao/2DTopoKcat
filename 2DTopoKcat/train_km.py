import json
import os
import subprocess
import sys
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
from utils.utils import DataLoader


torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(min(2, CPU_THREADS))
except RuntimeError:
    pass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


SEED = int(os.environ.get("MODEL_SEED", "42"))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "km")
ESMC6B_MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained", "ESMC-6B")
ESM2_MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained", "ESM2")
FAIR_ESM_VENDOR_PATH = os.path.join(
    PROJECT_ROOT, "third_party", "fair_esm_vendor"
)
EPOCHS = 50
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "30"))
TARGET_COLUMN = "log10km_mean"
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


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


def _read_km_splits(data_dir):
    paths = {
        "train": os.path.join(data_dir, "km-random_train.csv"),
        "validation": os.path.join(data_dir, "km-random_val.csv"),
        "test": os.path.join(data_dir, "km-random_test.csv"),
    }
    required = {"sequence", "substrate_smiles", TARGET_COLUMN}
    frames = {}
    for split_name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CatPred Km split not found: {path}")
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{split_name} CSV missing columns: {sorted(missing)}"
            )
        frame = frame.copy()
        if frame[list(required)].isnull().any().any():
            raise ValueError(f"{split_name} contains missing required values")
        frame["sequence"] = frame["sequence"].astype(str).str.strip()
        frame["substrate_smiles"] = (
            frame["substrate_smiles"].astype(str).str.strip()
        )
        frame[TARGET_COLUMN] = pd.to_numeric(
            frame[TARGET_COLUMN], errors="raise"
        )
        if frame[list(required)].isnull().any().any():
            raise ValueError(f"{split_name} contains missing required values")
        if (frame["sequence"].str.len() == 0).any():
            raise ValueError(f"{split_name} contains empty protein sequences")
        if not np.isfinite(frame[TARGET_COLUMN].to_numpy()).all():
            raise ValueError(
                f"{split_name} contains non-finite {TARGET_COLUMN}"
            )
        frames[split_name] = frame
    return paths, frames


def _merge_protein_caches(all_sequences, esmc_dict, esm2_dict):
    missing_esmc = set(all_sequences).difference(esmc_dict)
    missing_esm2 = set(all_sequences).difference(esm2_dict)
    if missing_esmc or missing_esm2:
        raise RuntimeError(
            f"Incomplete Km protein caches: ESMC={len(missing_esmc)}, "
            f"ESM2={len(missing_esm2)}"
        )
    merged = {}
    for sequence in all_sequences:
        feature = dict(esmc_dict[sequence])
        contact = esm2_dict[sequence]
        feature["edge_index"] = contact["edge_index"].cpu().long()
        feature["edge_weight"] = contact["edge_weight"].cpu().float()
        feature["contact_source"] = "ESM2_650M_contact_head"
        feature["contact_method"] = "ESM2_return_contacts_threshold_0.9"
        num_nodes = feature["esmc_multi_layer"].size(0)
        if feature["seq_feat"].size(0) != num_nodes:
            raise RuntimeError(f"ESMC/sequence length mismatch: {len(sequence)}")
        if feature["edge_index"].numel() > 0:
            if int(feature["edge_index"].min()) < 0:
                raise RuntimeError("Negative ESM2 contact edge index")
            if int(feature["edge_index"].max()) >= num_nodes:
                raise RuntimeError(f"ESM2 edge out of range: {len(sequence)}")
        merged[sequence] = feature
    return merged


def _esm2_generator_path(project_dir):
    candidates = (
        os.path.join(project_dir, "utils", "generate_esm2_contact_cache_fair_km.py"),
        os.path.join(project_dir, "generate_esm2_contact_cache_fair_km.py"),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("generate_esm2_contact_cache_fair_km.py not found")


def _write_esm2_manifest(cache_dir, frames):
    manifest_dir = os.path.join(cache_dir, "esm2_manifest")
    os.makedirs(manifest_dir, exist_ok=True)
    names = {
        "train": "Km_train.csv",
        "validation": "Km_val.csv",
        "test": "Km_test.csv",
    }
    for split_name, output_name in names.items():
        frames[split_name][["sequence"]].to_csv(
            os.path.join(manifest_dir, output_name), index=False
        )
    return manifest_dir


def _load_or_generate_caches(project_dir, data_dir, frames):

    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    all_sequences = sorted(set().union(*(
        set(frame["sequence"].tolist()) for frame in frames.values()
    )))
    all_smiles = sorted(set().union(*(
        set(frame["substrate_smiles"].tolist()) for frame in frames.values()
    )))
    esmc_path = os.path.join(cache_dir, "protein_dict_esmc6b_multilayer.pt")
    esm2_path = os.path.join(cache_dir, "protein_dict_esm2_contact.pt")
    ligand_path = os.path.join(cache_dir, "ligand_dict.pt")
    force_rebuild = os.environ.get("FORCE_REBUILD_CACHE", "0") == "1"

    if not force_rebuild and os.path.isfile(ligand_path):
        ligand_dict = torch.load(ligand_path, weights_only=False)
    else:
        ligand_dict = {}
    missing_smiles = sorted(set(all_smiles).difference(ligand_dict))
    if missing_smiles:
        print(f"Generating {len(missing_smiles)} missing Km ligand features...")
        from utils.chemberta_init_local import chemberta_ligand_init
        ligand_dict.update(chemberta_ligand_init(missing_smiles))
        torch.save(ligand_dict, ligand_path)
    else:
        print(f"Reusing Km ligand cache: {ligand_path}")

    esm2_dict = {}
    if not force_rebuild and os.path.isfile(esm2_path):
        esm2_dict = torch.load(esm2_path, weights_only=False)
    missing_esm2 = set(all_sequences).difference(esm2_dict)
    if missing_esm2:
        print(f"Generating ESM2 contacts; missing proteins={len(missing_esm2)}")
        vendor_path = os.environ.get("FAIR_ESM_VENDOR_PATH", FAIR_ESM_VENDOR_PATH)
        if not os.path.isfile(os.path.join(vendor_path, "esm", "pretrained.py")):
            raise RuntimeError(
                "Fair-ESM is required. Set FAIR_ESM_VENDOR_PATH to a folder "
                "containing esm/pretrained.py"
            )
        manifest_dir = _write_esm2_manifest(cache_dir, frames)
        environment = os.environ.copy()
        old_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            vendor_path if not old_pythonpath
            else vendor_path + os.pathsep + old_pythonpath
        )
        subprocess.run([
            sys.executable,
            _esm2_generator_path(project_dir),
            "--data-dir", manifest_dir,
            "--model-path", os.environ.get("ESM2_MODEL_PATH", ESM2_MODEL_PATH),
            "--output", esm2_path,
        ], env=environment, check=True)
        esm2_dict = torch.load(esm2_path, weights_only=False)

    if not force_rebuild and os.path.isfile(esmc_path):
        esmc_dict = torch.load(esmc_path, weights_only=False)
    else:
        esmc_dict = {}
    missing_esmc = sorted(set(all_sequences).difference(esmc_dict))
    if missing_esmc:
        print(f"Generating ESMC multilayer features; missing={len(missing_esmc)}")
        os.environ.setdefault(
            "ESMC6B_MODEL_PATH",
            os.environ.get("ESMC6B_MODEL_PATH", ESMC6B_MODEL_PATH),
        )
        from utils.protein_init_esmc6b_multilayer import protein_init as esmc_init
        esmc_dict.update(esmc_init(missing_esmc))
        torch.save(esmc_dict, esmc_path)
        print(f"Saved Km ESMC cache: {esmc_path}")

    protein_dict = _merge_protein_caches(all_sequences, esmc_dict, esm2_dict)
    print("Km protein caches shallow-merged in CPU RAM; no hybrid PT was saved")
    del esmc_dict, esm2_dict
    return protein_dict, ligand_dict


def _prepare_cofactor_filtered_splits(data_dir, frames):
    output_dir = os.environ.get(
        "COFACTOR_FILTERED_SPLIT_DIR",
        os.path.join(
            data_dir,
            "cofactor_filtered_splits_log10km_mean",
        ),
    )
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}
    output_frames = {}
    audit_parts = []
    sequence_filter_records = []
    filename_by_split = {
        "train": "km-random_train.csv",
        "validation": "km-random_val.csv",
        "test": "km-random_test.csv",
    }

    for split_name, frame in frames.items():
        processed = frame.copy()
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
        removed_sequence_count = input_count - len(processed)
        print(
            f"Standard-amino-acid filter [{split_name}]: "
            f"kept={len(processed)}, removed={removed_sequence_count}, "
            f"input={input_count}"
        )
        if processed.empty:
            raise RuntimeError(
                f"No {split_name} samples remain after sequence filtering"
            )

                                                                          
                                                                             
        processed["log10_value"] = processed[TARGET_COLUMN].astype(float)
        selected = []
        records = []
        for row_index, reaction_smiles in enumerate(
            processed["substrate_smiles"].tolist()
        ):
            result = select_main_substrate(reaction_smiles)
            selected.append(result.selected_smiles)
            record = result.audit_record()
            record.update({"split": split_name, "row_index": row_index})
            records.append(record)

        processed.insert(
            processed.columns.get_loc("substrate_smiles") + 1,
            "substrate_smiles_original",
            processed["substrate_smiles"],
        )
        processed["substrate_smiles"] = selected
                                                                      
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
    data_dir = os.environ.get("CATPRED_KM_DATA_DIR", DEFAULT_DATA_DIR)
    print(
        f"CPU threads={CPU_THREADS}; device={device}; epochs={EPOCHS}; "
        f"batch={BATCH_SIZE}; MC samples={MC_SAMPLES}; "
        f"target={TARGET_COLUMN}; seed={SEED}"
    )

    with open(os.path.join(project_dir, "config.json"), "r") as handle:
        config = json.load(handle)

    _, original_frames = _read_km_splits(data_dir)
    (
        split_paths,
        filtered_frames,
        audit_path,
        sequence_filter_audit_path,
    ) = _prepare_cofactor_filtered_splits(data_dir, original_frames)

    if os.environ.get("AUDIT_ONLY", "0") == "1":
        print("AUDIT_ONLY=1: preprocessing completed; training was not started")
        return

    protein_dict, ligand_dict = _load_or_generate_caches(
        project_dir, data_dir, filtered_frames
    )
    ligand_cache_path = os.path.join(data_dir, "cache", "ligand_dict.pt")
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
        f"CatPred Km-mean TDL single model: Train={len(train_dataset)}, "
        f"Validation={len(validation_dataset)}, Test={len(test_dataset)}"
    )

    output_dir = os.environ.get(
        "KM_OUTPUT_DIR",
        os.path.join(
            project_dir,
            "outputs",
            "single_model_esmc_multilayer_tdl_contact_adapter_log10km_mean",
        ),
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
        "dataset": "CatPred_Km_random_split_cofactor_filtered_log10km_mean",
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
        "protein_contact_graph": "ESM2-650M_return_contacts_threshold_0.9",
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

