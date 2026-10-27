import numpy as np
import pandas as pd
import sys

                                                    
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch
from torch_geometric.utils import add_self_loops, to_undirected, remove_self_loops, coalesce
import esm

import math

ESM2_DIM = 1280
ESM2_LAYER = 33

def protein_init(seqs):
    result_dict = {}
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    for seq in tqdm(seqs):
        seq_feat = seq_feature(seq)
        token_repr, contact_map_proba = esm2_extract(model, batch_converter, seq)
        edge_index, edge_weight = contact_map(contact_map_proba, contact_threshold=0.9)

        result_dict[seq] = {
            'seq': seq,
            'seq_feat': torch.from_numpy(seq_feat),
            'token_representation': token_repr.half(),
            'num_nodes': len(seq),
            'num_pos': torch.arange(len(seq)).reshape(-1, 1),
            'edge_index': edge_index,
            'edge_weight': edge_weight,
        }

    return result_dict


def esm2_extract(model, batch_converter, seq, chunk_size=700):

    if len(seq) <= chunk_size:
        data = [('protein', seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(next(model.parameters()).device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[ESM2_LAYER], return_contacts=True)
        token_repr = results['representations'][ESM2_LAYER][0, 1:len(seq)+1].cpu()
        contact_prob = results['contacts'][0].cpu()
        return token_repr, contact_prob
    else:
        token_repr = torch.zeros(len(seq), ESM2_DIM)
        contact_prob = torch.zeros(len(seq), len(seq))
        counts_repr = torch.zeros(len(seq), 1)
        counts_contact = torch.zeros(len(seq), len(seq))
        interval = chunk_size // 2
        for start in range(0, len(seq), interval):
            end = min(start + chunk_size, len(seq))
            sub_seq = seq[start:end]
            data = [('protein', sub_seq)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(next(model.parameters()).device)
            with torch.no_grad():
                results = model(tokens, repr_layers=[ESM2_LAYER], return_contacts=True)
            sub_repr = results['representations'][ESM2_LAYER][0, 1:len(sub_seq)+1].cpu()
            token_repr[start:end] += sub_repr
            counts_repr[start:end] += 1
            sub_contact = results['contacts'][0].cpu()
            contact_prob[start:end, start:end] += sub_contact
            counts_contact[start:end, start:end] += 1
        token_repr = token_repr / counts_repr.clamp(min=1)
        contact_prob = contact_prob / counts_contact.clamp(min=1)
        return token_repr, contact_prob


def _hidden_to_contact(hidden):

    norm = hidden / (hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    sim = torch.mm(norm, norm.t())           
    return (sim + 1) / 2                 


           
def dic_normalize(dic):
                
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
                      
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
                                                          
    return np.array(res_property1 + res_property2)


                  
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
                  
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def seq_feature(pro_seq):    
    if 'U' in pro_seq or 'B' in pro_seq:
        print('U or B in Sequence')
    pro_seq = pro_seq.replace('U','X').replace('B','X')
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
                            
                            
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def contact_map(contact_map_proba, contact_threshold=0.5):
    num_residues = contact_map_proba.shape[0]
    prot_contact_adj = (contact_map_proba >= contact_threshold).long()
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index
    edge_weight = contact_map_proba[row, col].float()
                                                                                                 
    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1],(torch.arange(num_residues)+1)[:-1]])
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:],(torch.arange(num_residues)-1)[1:]])
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1],dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1],dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2],(torch.arange(num_residues)+2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:],(torch.arange(num_residues)-2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) *contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2],dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2],dim=-1)
                                                                                                 

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight,fill_value=1)
    
    return edge_index, edge_weight



def esm_extract(model, batch_converter, seq, layer=36, approach='mean',dim=2560):
    pro_id = 'A'
    if len(seq) <= 700:
        data = []
        data.append((pro_id, seq))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

        logits = results["logits"][0].cpu().numpy()[1: len(seq) + 1]
        contact_prob_map = results["contacts"][0].cpu().numpy()
        token_representation = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
        assert token_representation.size(0) == layer

        if approach == 'last':
            token_representation = token_representation[-1]
        elif approach == 'sum':
            token_representation = token_representation.sum(dim=0)
        elif approach == 'mean':
            token_representation = token_representation.mean(dim=0)

        token_representation = token_representation.cpu().numpy()
        token_representation = token_representation[1: len(seq) + 1]
    else:
        contact_prob_map = np.zeros((len(seq), len(seq)))                                 
        token_representation = np.zeros((len(seq), dim))
        logits = np.zeros((len(seq),layer))
        interval = 350
        i = math.ceil(len(seq) / interval)
                                
                                
                                
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                         
                                               
        for s in range(i):
            start = s * interval                         
            end = min((s + 2) * interval, len(seq))                       
            sub_seq_len = end - start

                        
            temp_seq = seq[start:end]
            temp_data = []
            temp_data.append((pro_id, temp_seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
            batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

                                                
            row, col = np.where(contact_prob_map[start:end, start:end] != 0)
            row = row + start
            col = col + start
            contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                0].cpu().numpy()
            contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
            
            logits[start:end] += results['logits'][0].cpu().numpy()[1: len(temp_seq) + 1]
            logits[row] = logits[row]/2.0

                    
            subtoken_repr = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
            assert subtoken_repr.size(0) == layer
            if approach == 'last':
                subtoken_repr = subtoken_repr[-1]
            elif approach == 'sum':
                subtoken_repr = subtoken_repr.sum(dim=0)
            elif approach == 'mean':
                subtoken_repr = subtoken_repr.mean(dim=0)

            subtoken_repr = subtoken_repr.cpu().numpy()
            subtoken_repr = subtoken_repr[1: len(temp_seq) + 1]

            trow = np.where(token_representation[start:end].sum(axis=-1) != 0)[0]
            trow = trow + start
            token_representation[start:end] = token_representation[start:end] + subtoken_repr
            token_representation[trow] = token_representation[trow] / 2.0

            if end == len(seq):
                break

    return torch.from_numpy(token_representation), torch.from_numpy(contact_prob_map), torch.from_numpy(logits)




def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad                    
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


from Bio.PDB import PDBParser
biopython_parser = PDBParser()

one_to_three = {"A" : "ALA",
              "C" : "CYS",
              "D" : "ASP",
              "E" : "GLU",
              "F" : "PHE",
              "G" : "GLY",
              "H" : "HIS",
              "I" : "ILE",
              "K" : "LYS",
              "L" : "LEU",
              "M" : "MET",
              "N" : "ASN",
              "P" : "PRO",
              "Q" : "GLN",
              "R" : "ARG",
              "S" : "SER",
              "T" : "THR",
              "V" : "VAL",
              "W" : "TRP",
              "Y" : "TYR",
              "B" : "ASX",
              "Z" : "GLX",
              "X" : "UNK",
              "*" : " * "}

three_to_one = {}
for _key, _value in one_to_three.items():
    three_to_one[_value] = _key
three_to_one["SEC"] = "C"
three_to_one["MSE"] = "M"


def extract_pdb_seq(protein_path):

    structure = biopython_parser.get_structure('random_id', protein_path)[0]
    seq = ''
    chain_str = ''
    for i, chain in enumerate(structure):
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:                                                      
                try:
                    seq += three_to_one[residue.get_resname()]
                    chain_str += str(chain.id)
                except Exception as e:
                    seq += 'X'
                    chain_str += str(chain.id)
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash X.')
        
    return seq, chain_str