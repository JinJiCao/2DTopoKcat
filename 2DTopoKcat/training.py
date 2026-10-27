import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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


class SingleModelTrainer:


    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ranking_weight=0.1,
        beta_kl=1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.ranking_weight = ranking_weight
        self.beta_kl = beta_kl

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

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        squared_error = 0.0
        sample_count = 0
        total_loss = 0.0
        for data in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            (prediction, _, _, _, o_loss, cl_loss, _, _, _), data = self._forward(data)
            prediction = prediction.reshape(-1)
            target = data.reg_y.reshape(-1).to(self.device)
            loss = self.criterion(prediction, target) + o_loss + cl_loss
            loss = loss + self.beta_kl * self.model.reg_out.kl_divergence()
            if self.ranking_weight > 0 and target.shape[0] > 1:
                prediction_delta = prediction.unsqueeze(0) - prediction.unsqueeze(1)
                target_delta = target.unsqueeze(0) - target.unsqueeze(1)
                rank_loss = torch.nn.functional.softplus(
                    -prediction_delta * torch.sign(target_delta)
                ).mean()
                loss = loss + self.ranking_weight * rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                squared_error += ((prediction - target) ** 2).sum().item()
                sample_count += target.shape[0]
            total_loss += loss.item()
        mse = squared_error / sample_count
        return mse ** 0.5, mse, total_loss / len(train_loader)

    def evaluate(self, data_loader, n_samples=1):
        self.model.eval()
        predictions, targets, standard_deviations = [], [], []
        with torch.no_grad():
            for data in tqdm(data_loader, desc="Evaluating", leave=False):
                if n_samples == 1:
                    (prediction, _, _, _, _, _, _, _, _), data = self._forward(data)
                    prediction = prediction.reshape(-1).cpu().numpy()
                    standard_deviation = np.zeros(prediction.shape[0])
                else:
                    samples = []
                    for _ in range(n_samples):
                        (prediction, _, _, _, _, _, _, _, _), data = self._forward(data)
                        samples.append(prediction.reshape(-1).cpu().numpy())
                    samples = np.stack(samples, axis=0)
                    prediction = samples.mean(axis=0)
                    standard_deviation = samples.std(axis=0)
                predictions.append(prediction)
                standard_deviations.append(standard_deviation)
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

    @staticmethod
    def _format_metrics(metrics):
        return (
            f"MSE={metrics['mse']:.4f} RMSE={metrics['rmse']:.4f} "
            f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} "
            f"Pearson={metrics['pearson']:.4f} "
            f"Spearman={metrics['spearman']:.4f}"
        )

    def fit(
        self,
        train_loader,
        validation_loader,
        test_loader=None,
        epochs=50,
        lr=1e-4,
        weight_decay=1e-5,
        checkpoint_dir="checkpoints/single_model",
        test_samples=30,
    ):

        if epochs != 50:
            raise ValueError("This project is configured for exactly 50 epochs")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        log_path = os.path.join(checkpoint_dir, "training_log.txt")
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        best_validation_rmse = float("inf")
        best_model_state = None
        best_epoch = 0
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("Single-model training log\n" + "=" * 60 + "\n\n")

        for epoch in range(epochs):
            train_rmse, train_mse, train_loss = self.train_epoch(
                train_loader, optimizer
            )
            validation_metrics, _, _, _ = self.evaluate(
                validation_loader, n_samples=1
            )
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"Epoch {epoch + 1}/{epochs}\n")
                log.write(
                    f"  Train: RMSE={train_rmse:.4f} MSE={train_mse:.4f} "
                    f"Loss={train_loss:.4f}\n"
                )
                log.write(
                    f"  Validation: {self._format_metrics(validation_metrics)}\n"
                )
            validation_rmse = validation_metrics["rmse"]
            if validation_rmse < best_validation_rmse:
                best_validation_rmse = validation_rmse
                best_epoch = epoch + 1
                best_model_state = deepcopy(self.model.state_dict())
                torch.save(best_model_state, checkpoint_path)
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(
                        "  -> New best model "
                        f"(validation_rmse={validation_rmse:.4f})\n"
                    )
            scheduler.step()

        if best_model_state is None:
            raise RuntimeError("No validation checkpoint was produced")
        self.model.load_state_dict(best_model_state)
        test_metrics = None
        if test_loader is not None:
            test_metrics, test_predictions, test_stds, test_targets = self.evaluate(
                test_loader, n_samples=test_samples
            )
            pd.DataFrame({
                "y_true": test_targets,
                "y_pred": test_predictions,
                "uncertainty_std": test_stds,
            }).to_csv(
                os.path.join(checkpoint_dir, "test_predictions.csv"), index=False
            )
            with open(log_path, "a", encoding="utf-8") as log:
                log.write("\n" + "=" * 60 + "\n")
                log.write("Final test results (best validation checkpoint):\n")
                log.write(f"Best epoch: {best_epoch}\n")
                log.write(f"Best validation RMSE: {best_validation_rmse:.4f}\n")
                log.write(self._format_metrics(test_metrics) + "\n")
        return {
            "best_epoch": best_epoch,
            "best_validation_rmse": best_validation_rmse,
            "test_metrics": test_metrics,
            "checkpoint_path": checkpoint_path,
        }
