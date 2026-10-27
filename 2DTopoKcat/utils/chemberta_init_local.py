import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os
from pathlib import Path

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

CHEMBERTA_DIM = 384
DEFAULT_CHEMBERTA_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "pretrained" / "ChemBERTa-77M-MLM"
)


def _resolve_model_path(model_path=None):
    path = Path(
        model_path
        or os.environ.get("CHEMBERTA_MODEL_PATH", DEFAULT_CHEMBERTA_MODEL_PATH)
    ).expanduser()
    if (path / "config.json").is_file():
        return str(path)
    snapshots = path / "snapshots"
    candidates = list(snapshots.glob("*/config.json")) if snapshots.is_dir() else []
    if candidates:
        return str(max(candidates, key=lambda item: item.stat().st_mtime).parent)
    raise FileNotFoundError(
        f"ChemBERTa config.json not found below {path}. "
        "Set CHEMBERTA_MODEL_PATH to the local model directory."
    )


def _get_atom_token_map(smiles, tokenizer):

    encoding = tokenizer(smiles, return_tensors="pt", return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    offsets = encoding['offset_mapping'][0].tolist()

    atom_positions = []
    i = 0
    in_bracket = False
    bracket_start = -1

    while i < len(smiles):
        ch = smiles[i]

        if ch == '[':
            in_bracket = True
            bracket_start = i
            i += 1
            continue

        if ch == ']':
                                                        
            positions = []
            for t_idx, (s, e) in enumerate(offsets):
                if s is not None and s >= bracket_start and e <= i + 1:
                    positions.append(t_idx)
            if positions:
                atom_positions.append(positions)
            in_bracket = False
            i += 1
            continue

        if in_bracket:
            i += 1
            continue

                                        
        if ch in ('=', '#', '/', '\\', '.', '(', ')'):
            i += 1
            continue
        if ch.isdigit():
            i += 1
            continue

                                    
        if ch.islower() and ch in ('c', 'n', 'o', 's', 'p'):
            positions = []
            for t_idx, (s, e) in enumerate(offsets):
                if s is not None and s <= i < e:
                    positions.append(t_idx)
            if positions:
                atom_positions.append(positions)
            i += 1
            continue

                                   
        if ch.isupper():
                                       
            if i + 1 < len(smiles) and smiles[i + 1].islower() and smiles[i + 1] not in ('c', 'n', 'o', 's', 'p'):
                element = ch + smiles[i + 1]
                positions = []
                for t_idx, (s, e) in enumerate(offsets):
                    if s is not None and s >= i and e <= i + len(element) + 1:
                        positions.append(t_idx)
                if positions:
                    atom_positions.append(positions)
                i += 2
                continue

            positions = []
            for t_idx, (s, e) in enumerate(offsets):
                if s is not None and s <= i < e:
                    positions.append(t_idx)
            if positions:
                atom_positions.append(positions)
            i += 1
            continue

        i += 1

    return atom_positions, encoding


def chemberta_extract(model, tokenizer, smiles):

    encoding = tokenizer(smiles, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[0].cpu()                  

    atom_positions, _ = _get_atom_token_map(smiles, tokenizer)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    atom_features = torch.zeros(num_atoms, CHEMBERTA_DIM)

    for atom_idx, positions in enumerate(atom_positions):
        if atom_idx >= num_atoms:
            break
        if positions:
                                                                
            valid_positions = [p for p in positions if p < hidden_states.size(0)]
            if valid_positions:
                token_embeds = hidden_states[valid_positions]
                atom_features[atom_idx] = token_embeds.mean(dim=0)

    return atom_features.half()


def chemberta_ligand_init(smiles_list, model_path=None):


    from utils.ligand_init import smiles2graph

    resolved_path = _resolve_model_path(model_path)
    print(f"Loading local ChemBERTa from {resolved_path}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, local_files_only=True)
    model = AutoModel.from_pretrained(resolved_path, local_files_only=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    ligand_dict = {}
    failed = 0

    for smiles in tqdm(smiles_list, desc="ChemBERTa featurization"):
        graph = smiles2graph(smiles)

        chemberta_feat = chemberta_extract(model, tokenizer, smiles)
        if chemberta_feat is not None and chemberta_feat.shape[0] == graph['atom_feature'].shape[0]:
            graph['atom_feature'] = chemberta_feat
        else:
            failed += 1

                                                         
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            graph['fingerprint'] = torch.tensor(list(fp), dtype=torch.float32)
        else:
            graph['fingerprint'] = torch.zeros(1024, dtype=torch.float32)

        ligand_dict[smiles] = graph

    if failed > 0:
        print(f"Warning: {failed}/{len(smiles_list)} molecules failed ChemBERTa extraction, using RDKit features.")

    return ligand_dict
