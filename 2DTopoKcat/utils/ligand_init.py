from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
import numpy as np

import torch

fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
import sys

                                                    
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm




def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    encoding = one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10])
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'])
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]

    return np.array(encoding)



class MoleculeGraphDataset():
    def __init__(self,atom_classes=None, halogen_detail=False, save_path=None):
                          
        self.ATOM_CODES = {}
        if atom_classes is None:
            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))

            self.FEATURE_NAMES = []
            if halogen_detail:
                atom_classes = [
                    (5, 'B'),
                    (6, 'C'),
                    (7, 'N'),
                    (8, 'O'),
                    (15, 'P'),
                    (16, 'S'),
                    (34, 'Se'),
                              
                    (9, 'F'),
                    (17, 'Cl'),
                    (35, 'Br'),
                    (53, 'I'),
                              
                    (metals, 'metal')
                ]
            else:
                atom_classes = [
                    (5, 'B'),
                    (6, 'C'),
                    (7, 'N'),
                    (8, 'O'),
                    (15, 'P'),
                    (16, 'S'),
                    (34, 'Se'),
                              
                    ([9, 17, 35, 53], 'halogen'),
                              
                    (metals, 'metal')
                ]


        self.NUM_ATOM_CLASSES = len(atom_classes)
        for code, (atom, name) in enumerate(atom_classes):
            if type(atom) is list:
                for a in atom:
                    self.ATOM_CODES[a] = code
            else:
                self.ATOM_CODES[atom] = code
            self.FEATURE_NAMES.append(name)

                                        
        self.feat_types = ['Donor', 'Acceptor', 'Hydrophobe', 'LumpedHydrophobe']

                       
        self.edge_dict = {BondType.SINGLE: 1, BondType.DOUBLE: 2,
                     BondType.TRIPLE: 3, BondType.AROMATIC: 4,
                         BondType.UNSPECIFIED: 1}
                     
        self.save_path = save_path

    def hybridization_onehot(self,hybrid_type):
        hybrid_type = str(hybrid_type)
        types = {'S': 0, 'SP': 1, 'SP2': 2, 'SP3': 3, 'SP3D': 4, 'SP3D2': 5}

        encoding = np.zeros(len(types))
        try:
            encoding[types[hybrid_type]] = 1.0
        except:
            pass
        return encoding

    def encode_num(self,atomic_num):














        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def atom_feature_extract(self,atom):












        feat = []

        feat.append(atom.GetDegree())
        feat.append(atom.GetTotalValence())
        feat += self.hybridization_onehot(atom.GetHybridization()).tolist()
        feat.append(atom.GetNumRadicalElectrons())
        feat.append(atom.GetFormalCharge())
        feat.append(int(atom.GetIsAromatic()))
        feat.append(int(atom.IsInRing()))

        return feat

    def mol_feature(self,mol):
        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            feat = self.atom_feature_extract(atom)
            atom_feats.append(feat)

        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])

        return feature

    def mol_extra_feature(self, mol):
        atom_num = len(mol.GetAtoms())
        feature = np.zeros((atom_num, len(self.feat_types)))

        fact_feats = factory.GetFeaturesForMol(mol)
        for f in fact_feats:
            f_type = f.GetFamily()
            if f_type in self.feat_types:
                f_index = self.feat_types.index(f_type)
                atom_ids = f.GetAtomIds()
                feature[atom_ids, f_index] = 1

        return feature

    def mol_simplified_feature(self,mol):
        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            atomic_num = atom.GetAtomicNum()

            if atomic_num in self.ATOM_CODES.keys():
                atom_feats.append([self.ATOM_CODES[atomic_num] + 1])
            else:
                atom_feats.append([0])
        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])

        return feature

    def mol_sequence_simplified_feature(self,mol):

        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():

            atom_ids.append(atom.GetIdx())
            onehot_label = one_of_k_encoding_unk(atom.GetSymbol(),
                                  ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
                                   'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                   'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                                   'Unknown'])
            out = np.array(onehot_label).nonzero()[0]
            atom_feats.append(out)

        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])

        return feature

    def mol_full_feature(self, mol):
        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            feature = atom_features(atom)
            atom_feats.append(feature)
        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])

        return feature

    def bond_feature(self,mol):



        atom_num = len(mol.GetAtoms())
        adj = np.zeros((atom_num,atom_num))

        for b in mol.GetBonds():
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            b_type = self.edge_dict[b.GetBondType()]
            adj[begin_idx, end_idx] = b_type
            adj[end_idx, begin_idx] = b_type

        return adj

    def junction_tree(self,mol):
        tree_edge_index, atom2clique_index, num_cliques, x_clique = tree_decomposition(mol,return_vocab=True)
                                                                 
        if atom2clique_index.nelement() == 0:
            num_cliques = len(mol.GetAtoms())
            x_clique = torch.tensor([3]*num_cliques)
            atom2clique_index = torch.stack([torch.arange(num_cliques),
                                             torch.arange(num_cliques)])
        tree = dict(tree_edge_index=tree_edge_index,
             atom2clique_index=atom2clique_index,
             num_cliques=num_cliques,
             x_clique=x_clique)

        return tree


    def featurize(self,mol,type='atom_type'):
        if type=='atom_type':
            atom_feature = self.mol_simplified_feature(mol)
        elif type =='detailed_atom_type':
            atom_feature = self.mol_sequence_simplified_feature(mol)
        elif type=='atom_feature':
            base_feat = self.mol_feature(mol)
            extra_feat = self.mol_extra_feature(mol)
            atom_feature = np.concatenate((base_feat, extra_feat), axis=1)

        elif type=='atom_full_feature':
            atom_feature = self.mol_full_feature(mol)
        else:
            raise Exception('atom_type or atom_feature')
        bond_feature = self.bond_feature(mol)

        return atom_feature, bond_feature




                                         
from itertools import chain
from typing import Any, Tuple, Union

import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import Tensor

from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    to_scipy_sparse_matrix,
    to_undirected,
)


def tree_decomposition(
    mol: Any,
    return_vocab: bool = False,
) -> Union[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, int, Tensor]]:
















    import rdkit.Chem as Chem

                                
    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    xs = [0] * len(cliques)
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            xs.append(1)

                                      
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

                                                                              
    for c1 in range(len(cliques)):
        for atom in cliques[c1]:
            for c2 in atom2clique[atom]:
                if c1 >= c2 or len(cliques[c1]) <= 2 or len(cliques[c2]) <= 2:
                    continue
                if len(set(cliques[c1]) & set(cliques[c2])) > 2:
                    cliques[c1] = set(cliques[c1]) | set(cliques[c2])
                    xs[c1] = 2
                    cliques[c2] = []
                    xs[c2] = -1
    cliques = [c for c in cliques if len(c) > 0]
    xs = [x for x in xs if x >= 0]

                                    
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

                                                                      
                                                             
    edges = {}
    for atom in range(mol.GetNumAtoms()):
        cs = atom2clique[atom]
        if len(cs) <= 1:
            continue

                                                        
        bonds = [c for c in cs if len(cliques[c]) == 2]
                                                        
        rings = [c for c in cs if len(cliques[c]) > 4]

        if len(bonds) > 2 or (len(bonds) == 2 and len(cs) > 2):
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 1

        elif len(rings) > 2:
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 99

        else:
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    c1, c2 = cs[i], cs[j]
                    count = len(set(cliques[c1]) & set(cliques[c2]))
                    edges[(c1, c2)] = min(count, edges.get((c1, c2), 99))

                                    
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

    if len(edges) > 0:
        edge_index_T, weight = zip(*edges.items())
        edge_index = torch.tensor(edge_index_T).t()
        inv_weight = 100 - torch.tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(cliques))
        junc_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(junc_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(cliques))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    rows = [[i] * len(atom2clique[i]) for i in range(mol.GetNumAtoms())]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(atom2clique)))
    atom2clique = torch.stack([row, col], dim=0).to(torch.long)

    if return_vocab:
        vocab = torch.tensor(xs, dtype=torch.long)
        return edge_index, atom2clique, len(cliques), vocab
    else:
        return edge_index, atom2clique, len(cliques)


   

def smiles2graph(m_str):
    mgd = MoleculeGraphDataset(halogen_detail=False)
    mol = Chem.MolFromSmiles(m_str)
    atom_feature, bond_feature = mgd.featurize(mol,'atom_full_feature')
    atom_idx, _ = mgd.featurize(mol,'atom_type')
    tree = mgd.junction_tree(mol)

    out_dict = {
        'smiles':m_str,
        'atom_feature':torch.tensor(atom_feature),
        'atom_types':'|'.join([i.GetSymbol() for i in mol.GetAtoms()]),
        'atom_idx':torch.tensor(atom_idx),
        'bond_feature':torch.tensor(bond_feature),
    }
    tree['tree_edge_index'] = tree['tree_edge_index']
    tree['atom2clique_index'] = tree['atom2clique_index']
    tree['x_clique'] = tree['x_clique']

    out_dict.update(tree)

    return out_dict

def ligand_init(smiles_list):
    ligand_dict = {}
    for smiles in tqdm(smiles_list):
        ligand_dict[smiles] = smiles2graph(smiles)

    return ligand_dict
