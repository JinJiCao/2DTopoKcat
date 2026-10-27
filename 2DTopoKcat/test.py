import os
import random
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KcatData(Data):


    def __inc__(self, key, value, *args, **kwargs):
        if key == "mol_edge_index":
            return self.mol_x.size(0)
        if key == "clique_edge_index":
            return self.clique_x.size(0)
        if key == "atom2clique_index":
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        if key == "prot_edge_index":
            return self.prot_node_aa.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class KcatDataset(Dataset):


    def __init__(self, csv_path, mol_dict, prot_dict, device="cpu"):
        super().__init__()
        self.pairs = pd.read_csv(csv_path)
        self.mols = mol_dict
        self.prots = prot_dict
        self.device = device

        for molecule in self.mols.values():
            molecule["atom_idx"] = molecule["atom_idx"].long().view(-1, 1)
            molecule["atom_feature"] = molecule["atom_feature"].float()
            molecule["x_clique"] = molecule["x_clique"].long().view(-1, 1)
            molecule["tree_edge_index"] = molecule["tree_edge_index"].long()
            molecule["atom2clique_index"] = molecule["atom2clique_index"].long()

        for protein in self.prots.values():
            protein["seq_feat"] = protein["seq_feat"].float()
            if protein["esmc_multi_layer"].dim() != 3:
                raise ValueError("esmc_multi_layer must have shape [L, 4, D]")

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _main_component(smiles):
        if "." not in smiles:
            return smiles
        candidates = []
        for component in smiles.split("."):
            molecule = Chem.MolFromSmiles(component)
            if molecule is not None:
                candidates.append((molecule.GetNumAtoms(), component))
        return max(candidates, default=(0, smiles.split(".")[0]))[1]

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        smiles = self._main_component(row["reactant_smiles"])
        sequence = row["sequence"]
        molecule = self.mols[smiles]
        protein = self.prots[sequence]

        mol_x = molecule["atom_idx"]
        mol_x_feat = molecule["atom_feature"]
        num_atoms = mol_x.shape[0]
        bond_feat = molecule.get(
            "bond_feature", torch.zeros(num_atoms, num_atoms, dtype=torch.long)
        )
        bond_feat = torch.as_tensor(bond_feat)
        if bond_feat.dim() != 2 or bond_feat.shape[0] != bond_feat.shape[1]:
            bond_feat = torch.zeros(num_atoms, num_atoms, dtype=torch.long)
        elif bond_feat.shape[0] != num_atoms:
            resized = torch.zeros(num_atoms, num_atoms, dtype=torch.long)
            common_size = min(bond_feat.shape[0], num_atoms)
            resized[:common_size, :common_size] = bond_feat[
                :common_size, :common_size
            ].long()
            bond_feat = resized
        bond_feat = bond_feat.long().clamp(0, 4)

        mol_edge_index = bond_feat.nonzero(as_tuple=False).t().contiguous()
        if mol_edge_index.numel() > 0:
            valid = (
                (mol_edge_index[0] < num_atoms)
                & (mol_edge_index[1] < num_atoms)
            )
            mol_edge_index = mol_edge_index[:, valid]
            mol_edge_attr = bond_feat[
                mol_edge_index[0], mol_edge_index[1]
            ].clamp(1, 4)
        else:
            mol_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            mol_edge_attr = torch.tensor([1], dtype=torch.long)

        clique_x = molecule["x_clique"]
        clique_nodes = clique_x.shape[0]
        clique_edge_index = molecule["tree_edge_index"].long()
        atom2clique_index = molecule["atom2clique_index"].long()
        if num_atoms == 1:
            if clique_nodes == 0:
                clique_x = torch.tensor([3], dtype=torch.long)
                clique_nodes = 1
            if atom2clique_index.numel() == 0:
                atom2clique_index = torch.tensor([[0], [0]], dtype=torch.long)
            if clique_edge_index.numel() == 0:
                clique_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        if clique_edge_index.numel() > 0:
            valid = (
                (clique_edge_index[0] < clique_nodes)
                & (clique_edge_index[1] < clique_nodes)
            )
            clique_edge_index = clique_edge_index[:, valid]
            if clique_edge_index.numel() == 0:
                clique_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        if atom2clique_index.numel() > 0:
            valid = (
                (atom2clique_index[0] < num_atoms)
                & (atom2clique_index[1] < clique_nodes)
            )
            atom2clique_index = atom2clique_index[:, valid]
            if atom2clique_index.numel() == 0:
                atom2clique_index = torch.tensor([[0], [0]], dtype=torch.long)

        fingerprint = molecule.get(
            "fingerprint", torch.zeros(1024, dtype=torch.float32)
        ).unsqueeze(0)
        return KcatData(
            mol_x=mol_x,
            mol_x_feat=mol_x_feat,
            mol_edge_index=mol_edge_index,
            mol_edge_attr=mol_edge_attr,
            clique_x=clique_x,
            clique_edge_index=clique_edge_index,
            atom2clique_index=atom2clique_index,
            fingerprint=fingerprint,
            prot_node_aa=protein["seq_feat"],
            prot_node_evo=protein["esmc_multi_layer"].float(),
            prot_edge_index=protein["edge_index"],
            prot_edge_weight=protein["edge_weight"],
            reg_y=torch.tensor(row["log10_value"], dtype=torch.float32),
        )


import argparse
from substrate_selector_cofactor import select_main_substrate
from models.bayesian_layers import BayesianMLP
from models.net_esmc_multilayer_tdl_contact_adapter import (
    ESMCMultiLayerTDLContactModel,
)
from utils.utils import DataLoader

class ModelEvaluator:
    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device

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
            fingerprint=fingerprint,
            mol_batch=data.mol_x_batch,
            prot_batch=data.prot_node_aa_batch,
            clique_batch=data.clique_x_batch,
        ), data

    def evaluate(self, data_loader, n_samples=30):
        self.model.eval()
        predictions, targets, standard_deviations = [], [], []
        with torch.no_grad():
            for data in tqdm(data_loader, desc="Testing", leave=False):
                samples = []
                for _ in range(n_samples):
                    (prediction, _, _, _, _, _, _, _, _), data = self._forward(data)
                    samples.append(prediction.reshape(-1).cpu().numpy())
                samples = np.stack(samples, axis=0)
                predictions.append(samples.mean(axis=0))
                standard_deviations.append(samples.std(axis=0))
                targets.append(data.reg_y.reshape(-1).cpu().numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        standard_deviations = np.concatenate(standard_deviations)
        mse = mean_squared_error(targets, predictions)
        metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(targets, predictions),
            "r2": r2_score(targets, predictions),
            "pearson": pearsonr(targets, predictions)[0],
            "spearman": spearmanr(targets, predictions)[0],
        }
        return metrics, predictions, standard_deviations, targets


def build_triangle_index(edge_index, num_nodes):

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
            protein["_tdl_triangle_index"] = build_triangle_index(
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


class TDLEvaluator(ModelEvaluator):
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


def create_loader(dataset, shuffle, batch_size=32):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=["mol_x", "clique_x", "prot_node_aa"],
    )


def create_model(config, device, esmc_shape):
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


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MODEL_CONFIG = {
    "params": {
        "mol_in_channels": 384,
        "hidden_channels": 200,
        "aggregators": ["mean", "min", "max", "std"],
        "scalers": ["identity", "amplification", "linear"],
        "pre_layers": 2,
        "post_layers": 1,
        "total_layer": 3,
        "K": [5, 10, 20],
        "dropout": 0,
        "dropout_attn_score": 0.2,
        "heads": 5,
    }
}
TASKS = {
    "kcat": {
        "target": "log10kcat_max",
        "smiles_column": "reactant_smiles",
        "data_subdir": "kcat",
        "test_filename": "kcat-random_test.csv",
        "ligand_filename": "ligand_dict_cofactor_filtered.pt",
        "checkpoint_filename": "kcat_best_model.pt",
    },
    "km": {
        "target": "log10km_mean",
        "smiles_column": "substrate_smiles",
        "data_subdir": "km",
        "test_filename": "Km_test.csv",
        "ligand_filename": "ligand_dict.pt",
        "checkpoint_filename": "km_best_model.pt",
    },
}


def validate_sequence(sequence):
    normalized = str(sequence).strip().upper()
    if not normalized:
        raise ValueError("Empty protein sequence")
    invalid = sorted(set(normalized).difference(STANDARD_AMINO_ACIDS))
    if invalid:
        raise ValueError(
            f"Protein sequence contains unsupported residues: {invalid}"
        )
    return normalized


def require_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_cache_entries(frame, ligand_dict, protein_dict):
    missing_ligands = sorted(
        set(frame["reactant_smiles"]).difference(ligand_dict)
    )
    missing_proteins = sorted(set(frame["sequence"]).difference(protein_dict))
    if missing_ligands:
        raise KeyError(
            f"Ligand cache is missing {len(missing_ligands)} test substrates"
        )
    if missing_proteins:
        raise KeyError(
            f"Protein cache is missing {len(missing_proteins)} test sequences"
        )


def prepare_test_csv(task, source_path, output_path):
    settings = TASKS[task]
    target_column = settings["target"]
    smiles_column = settings["smiles_column"]
    frame = pd.read_csv(source_path)
    required = {"sequence", smiles_column, target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Test CSV missing columns: {sorted(missing)}")
    processed = frame.copy()
    processed["sequence"] = processed["sequence"].map(validate_sequence)
    processed["reactant_smiles"] = processed[smiles_column].map(
        lambda value: select_main_substrate(str(value)).selected_smiles
    )
    processed["log10_value"] = pd.to_numeric(
        processed[target_column], errors="raise"
    )
    if not np.isfinite(processed["log10_value"].to_numpy()).all():
        raise ValueError(f"Test CSV contains non-finite {target_column}")
    processed.to_csv(output_path, index=False)
    return processed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--test-csv")
    parser.add_argument("--ligand-cache")
    parser.add_argument("--protein-cache")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = TASKS[args.task]
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data", settings["data_subdir"])
    output_dir = args.output_dir or os.path.join(
        project_dir, "test_results", args.task
    )
    os.makedirs(output_dir, exist_ok=True)
    test_csv = require_file(
        args.test_csv or os.path.join(data_dir, settings["test_filename"]),
        f"{args.task} test CSV",
    )
    ligand_cache = require_file(
        args.ligand_cache
        or os.path.join(data_dir, "cache", settings["ligand_filename"]),
        f"{args.task} ligand cache",
    )
    protein_cache = require_file(
        args.protein_cache
        or os.path.join(
            data_dir,
            "cache",
            "protein_dict_esmc6b_multilayer_esm2contact.pt",
        ),
        f"{args.task} protein cache",
    )
    checkpoint = require_file(
        args.checkpoint
        or os.path.join(
            project_dir, "checkpoints", settings["checkpoint_filename"]
        ),
        f"{args.task} model checkpoint",
    )
    processed_path = os.path.join(output_dir, "processed_test.csv")
    test_frame = prepare_test_csv(args.task, test_csv, processed_path)
    ligand_dict = torch.load(
        ligand_cache, map_location="cpu", weights_only=False
    )
    protein_dict = torch.load(
        protein_cache, map_location="cpu", weights_only=False
    )
    require_cache_entries(test_frame, ligand_dict, protein_dict)
    first_protein = next(iter(protein_dict.values()))
    esmc_shape = first_protein["esmc_multi_layer"].shape
    if len(esmc_shape) != 3 or esmc_shape[1] != 4:
        raise RuntimeError(
            f"Expected ESMC cache [L,4,D], got {tuple(esmc_shape)}"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    dataset = KcatTDLDataset(
        processed_path, ligand_dict, protein_dict, device=device
    )
    loader = create_loader(
        dataset, shuffle=False, batch_size=args.batch_size
    )
    model = create_model(MODEL_CONFIG, device, esmc_shape)
    model.load_state_dict(load_state_dict(checkpoint, device))
    evaluator = TDLEvaluator(model, device=device)
    metrics, predictions, uncertainties, targets = evaluator.evaluate(
        loader, n_samples=args.mc_samples
    )
    target_column = settings["target"]
    predictions_path = os.path.join(output_dir, "test_predictions.csv")
    pd.DataFrame({
        f"{target_column}_true": targets,
        f"{target_column}_pred": predictions,
        "uncertainty_std": uncertainties,
    }).to_csv(predictions_path, index=False)
    summary_path = os.path.join(output_dir, "test_summary.csv")
    pd.DataFrame([{
        "task": args.task,
        "target": target_column,
        "test_size": len(dataset),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mc_samples": args.mc_samples,
        **metrics,
    }]).to_csv(summary_path, index=False)
    print(f"Test RMSE={metrics['rmse']:.4f}")
    print(f"Test R2={metrics['r2']:.4f}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

