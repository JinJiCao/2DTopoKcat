
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import esm


def resolve_checkpoint(path_string):
    path = Path(path_string).expanduser()
    if path.is_file():
        return path
    candidates = [
        item for item in path.rglob("*.pt")
        if "contact-regression" not in item.name
    ]
    preferred = [
        item for item in candidates if "esm2_t33_650M_UR50D" in item.name
    ]
    candidates = preferred or candidates
    if not candidates:
        raise FileNotFoundError(f"No ESM2 base .pt checkpoint below {path}")
    return sorted(candidates, key=lambda item: (len(str(item)), str(item)))[0]


def extract_contacts(model, batch_converter, sequence, chunk_size=700):
    device = next(model.parameters()).device

    def extract_chunk(chunk):
        _, _, tokens = batch_converter([("protein", chunk)])
        tokens = tokens.to(device)
        with torch.inference_mode():
            output = model(tokens, repr_layers=[], return_contacts=True)
        contacts = output["contacts"][0].float().cpu()
        if contacts.shape != (len(chunk), len(chunk)):
            raise RuntimeError(
                f"Contact/residue mismatch: {contacts.shape} versus {len(chunk)}"
            )
        return contacts

    if len(sequence) <= chunk_size:
        return extract_chunk(sequence)

    step = chunk_size // 2
    contacts = torch.zeros(len(sequence), len(sequence))
    counts = torch.zeros(len(sequence), len(sequence))
    for start in range(0, len(sequence), step):
        end = min(start + chunk_size, len(sequence))
        chunk_contacts = extract_chunk(sequence[start:end])
        contacts[start:end, start:end] += chunk_contacts
        counts[start:end, start:end] += 1
        if end == len(sequence):
            break
    return contacts / counts.clamp_min(1)


def contact_graph(probability, threshold=0.9):

    num_nodes = probability.size(0)
    weights = torch.maximum(probability, probability.T).clone()
    for offset in (1, 2):
        if num_nodes > offset:
            index = torch.arange(num_nodes - offset)
            local = torch.full_like(index, threshold, dtype=weights.dtype)
            weights[index, index + offset] = torch.maximum(
                weights[index, index + offset], local
            )
            weights[index + offset, index] = torch.maximum(
                weights[index + offset, index], local
            )
    weights.fill_diagonal_(1.0)
    edge_index = (weights >= threshold).nonzero(as_tuple=False).T.contiguous()
    edge_weight = weights[edge_index[0], edge_index[1]].float()
    return edge_index.long(), edge_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split_files = ("Km_train.csv", "Km_val.csv", "Km_test.csv")
    frames = [
        pd.read_csv(os.path.join(args.data_dir, filename))
        for filename in split_files
    ]
    sequences = sorted(set().union(*(
        set(frame["sequence"].astype(str).str.strip().tolist())
        for frame in frames
    )))
    checkpoint = resolve_checkpoint(args.model_path)
    print(f"Fair-ESM checkpoint: {checkpoint}")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    result = {}
    for sequence in tqdm(sequences, desc="Fair-ESM2 Km contact graphs"):
        probability = extract_contacts(model, batch_converter, sequence)
        edge_index, edge_weight = contact_graph(probability, threshold=0.9)
        result[sequence] = {
            "seq": sequence,
            "num_nodes": len(sequence),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "encoder": "ESM2_t33_650M_UR50D",
            "encoder_path": str(checkpoint),
            "contact_method": "ESM2_return_contacts_threshold_0.9",
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(result, args.output)
    print(f"Saved {len(result)} ESM2 Km contact graphs to {args.output}")


if __name__ == "__main__":
    main()
