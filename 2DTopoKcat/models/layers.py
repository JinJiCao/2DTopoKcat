import math
from typing import Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from models.pna import PNAConv
import torch

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing, GCNConv, SAGEConv, APPNP, SGConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax, degree, subgraph, to_scipy_sparse_matrix, segregate_self_loops, add_remaining_self_loops
import numpy as np 
import scipy.sparse as sp



class SGCluster(torch.nn.Module):
    def __init__(self, in_dim, out_dim, K, in_norm=False):                    
        super().__init__()
        self.sgc = SGConv(in_dim, out_dim, K=K)
        self.in_norm = in_norm
        if self.in_norm:
            self.in_ln = nn.LayerNorm(in_dim)

    def reset_parameters(self):
        self.sgc.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x, edge_index):
        y = x
                          
        if self.in_norm:
            y = self.in_ln(y)

        y = self.sgc(y, edge_index)

        return y

class APPNPCluster(torch.nn.Module):
    def __init__(self, in_dim, out_dim, a, K, in_norm=False):                    
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)
        self.propagate = APPNP(alpha=a, K=K, dropout=0)
        self.in_norm = in_norm
        if self.in_norm:
            self.in_ln = nn.LayerNorm(in_dim)

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x, edge_index):
        y = x
                          
        if self.in_norm:
            y = self.in_ln(y)
        y = self.lin(y)

        y = self.propagate(y, edge_index)
        
        return y

class GCNCluster(torch.nn.Module):
    def __init__(self, dims, out_norm=False, in_norm=False):                    
        super().__init__()
        list_Conv_layers = [ GCNConv(dims[idx-1], dims[idx]) for idx in range(1,len(dims)) ]
        self.Conv_layers = nn.ModuleList(list_Conv_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.Conv_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x, edge_index):
        y = x
                          
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.Conv_layers[idx](y, edge_index)
            y = F.relu(y)
        y = self.Conv_layers[-1](y, edge_index)

        if self.out_norm:
            y = self.out_ln(y)

        return y

class SAGECluster(torch.nn.Module):
    def __init__(self, dims, in_norm=False, add_self_loops=True, root_weight=False, 
                normalize=False, temperature=False):                    
        super().__init__()
        list_Conv_layers = [ SAGEConv(dims[idx-1], dims[idx], root_weight=root_weight) for idx in range(1,len(dims)) ]
        self.Conv_layers = nn.ModuleList(list_Conv_layers)
        self.hidden_layers = len(dims) - 2

        self.in_norm = in_norm
        self.temperature = temperature
        self.normalize = normalize 

        if self.temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])
            
        self.add_self_loops = add_self_loops

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.Conv_layers[idx].reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x, edge_index):
        if self.add_self_loops:
            edge_index, _ = add_remaining_self_loops(edge_index=edge_index, num_nodes=x.size(0))
        y = x
                          
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.Conv_layers[idx](y, edge_index)
            y = F.relu(y)
        y = self.Conv_layers[-1](y, edge_index)
        
        if self.normalize:
            y = F.normalize(y, p=2., dim=-1)

        if self.temperature:
            logit_scale = self.logit_scale.exp()
            y = y * logit_scale
        
        return y

class AtomEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(AtomEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):
            self.embeddings.append(torch.nn.Embedding(100, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out


class BondEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BondEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(3):
            self.embeddings.append(torch.nn.Embedding(10, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            out += self.embeddings[i](edge_attr[:, i])
        return out


class PosLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_value=0.2,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
                                   
                                                      
                                                      

        lower_bound = init_value/2
        upper_bound = init_value
        weight = nn.init.uniform_(torch.empty((out_features, in_features),**factory_kwargs), a=lower_bound, b=upper_bound)
                                                                                                                      
        weight = torch.abs(weight)
        self.weight = nn.Parameter(weight.log())
                                                                                                
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



    def reset_parameters(self) -> None:
                                                               
                                                    
        if self.bias is not None:
            nn.init.uniform_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.exp(), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MLP(nn.Module):

    def __init__(self, dims, out_norm=False, in_norm=False, bias=True):                    
        super().__init__()
        list_FC_layers = [ nn.Linear(dims[idx-1], dims[idx], bias=bias) for idx in range(1,len(dims)) ]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.FC_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x):
        y = x
                          
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.FC_layers[idx](y)
            y = F.relu(y)
        y = self.FC_layers[-1](y)

        if self.out_norm:
            y = self.out_ln(y)

        return y

class Drug_PNAConv(nn.Module):
    def __init__(self, mol_deg, hidden_channels, edge_channels,
                 pre_layers=2, post_layers=2,
                 aggregators=['sum', 'mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 num_towers=4,
                 dropout=0.1):
        super(Drug_PNAConv, self).__init__()

        self.bond_encoder = torch.nn.Embedding(5, hidden_channels)

        self.atom_conv = PNAConv(
            in_channels=hidden_channels, out_channels=hidden_channels,
            edge_dim=edge_channels, aggregators=aggregators,
            scalers=scalers, deg=mol_deg, pre_layers=pre_layers,
            post_layers=post_layers,towers=num_towers,divide_input=True,
        )
        self.atom_norm = torch.nn.LayerNorm(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        self.atom_conv.reset_parameters()
        self.atom_norm.reset_parameters()


    def forward(self, atom_x, bond_x, atom_edge_index):
        atom_in = atom_x
                                              
        bond_x_flat = bond_x.view(-1)
        if bond_x_flat.min() < 0 or bond_x_flat.max() > 4:
            print(f"[Drug_PNAConv] bond_x out of range! min={bond_x_flat.min().item()}, max={bond_x_flat.max().item()}")
            bond_x_flat = torch.clamp(bond_x_flat, 0, 4)
        torch.cuda.synchronize()                                    
        bond_x = self.bond_encoder(bond_x_flat)
        torch.cuda.synchronize()                        
        atom_x = atom_in + F.relu(self.atom_norm(self.atom_conv(atom_x, atom_edge_index, bond_x)))
        atom_x = F.dropout(atom_x, self.dropout, training=self.training)

        return atom_x


class Protein_PNAConv(nn.Module):
    def __init__(self, prot_deg, hidden_channels, edge_channels,
                 pre_layers=2, post_layers=2,
                 aggregators=['sum', 'mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 num_towers=4,
                 dropout=0.1):
        super(Protein_PNAConv, self).__init__()

        self.conv = PNAConv(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            edge_dim=edge_channels,
                            aggregators=aggregators,
                            scalers=scalers,
                            deg=prot_deg,
                            pre_layers=pre_layers,
                            post_layers=post_layers,
                            towers=num_towers,
                            divide_input=True,
                            )
                            
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.dropout = dropout
        
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, prot_edge_index, prot_edge_attr):
        x_in = x
        x = x_in + F.relu(self.norm(self.conv(x, prot_edge_index, prot_edge_attr)))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class DrugProteinConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        atom_channels: int,
        residue_channels: int,
        heads: int = 1,
        t = 0.2,
        dropout_attn_score = 0.2,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(DrugProteinConv, self).__init__(node_dim=0, **kwargs)
        
        assert residue_channels%heads == 0 
        assert atom_channels%heads == 0
        
        self.residue_out_channels = residue_channels//heads
        self.atom_out_channels = atom_channels//heads
        self.heads = heads
        self.edge_dim = edge_dim
        self._alpha = None
        
                                       
        self.lin_key = nn.Linear(residue_channels, heads * self.atom_out_channels, bias=False)
        self.lin_query = nn.Linear(atom_channels, heads * self.atom_out_channels, bias=False)
        self.lin_value = nn.Linear(residue_channels, heads * self.atom_out_channels, bias=False)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * self.atom_out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)
        
                                       
        self.lin_atom_value = nn.Linear(atom_channels, heads * self.residue_out_channels, bias=False)
        
                        
        self.drug_in_norm = torch.nn.LayerNorm(atom_channels)
        self.residue_in_norm = torch.nn.LayerNorm(residue_channels)

        self.drug_out_norm = torch.nn.LayerNorm(heads * self.atom_out_channels)
        self.residue_out_norm = torch.nn.LayerNorm(heads * self.residue_out_channels)
              
        self.clique_mlp = MLP([atom_channels*2, atom_channels*2, atom_channels], out_norm=True)
        self.residue_mlp = MLP([residue_channels*2, residue_channels*2, residue_channels], out_norm=True)
                      
        self.t = t
                                                                               

                                 
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
                         
        self.lin_atom_value.reset_parameters()
                         
        self.drug_in_norm.reset_parameters()
        self.residue_in_norm.reset_parameters()
        self.drug_out_norm.reset_parameters()
        self.residue_out_norm.reset_parameters()

                    
        self.clique_mlp.reset_parameters()
        self.residue_mlp.reset_parameters()

    def forward(self, drug_x, clique_x, clique_batch, residue_x, edge_index: Adj):

                                      
        H, aC = self.heads, self.atom_out_channels
        residue_hx = self.residue_in_norm(residue_x)                 
        query = self.lin_query(drug_x).view(-1, H, aC)
        key = self.lin_key(residue_hx).view(-1, H, aC)
        value = self.lin_value(residue_hx).view(-1, H, aC)
        
                                                                                                 
        drug_out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=None, size=None)
        alpha = self._alpha
        self._alpha = None

        drug_out = drug_out.view(-1, H * aC)
        drug_out = self.drug_out_norm(drug_out)
        clique_out = torch.cat([clique_x, drug_out[clique_batch]], dim=-1)
        clique_out = self.clique_mlp(clique_out)

                                       
        H, rC = self.heads, self.residue_out_channels
        drug_hx = self.drug_in_norm(drug_x)                 
        residue_value = self.lin_atom_value(drug_hx).view(-1, H, rC)[edge_index[1]]
        residue_out = residue_value * alpha.view(-1, H, 1) 
        residue_out = residue_out.view(-1, H * rC)
        residue_out = self.residue_out_norm(residue_out)
        residue_out = torch.cat([residue_out, residue_x], dim=-1)
        residue_out = self.residue_mlp(residue_out)

        return clique_out, residue_out, (edge_index, alpha)


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.atom_out_channels)
        alpha = alpha / self.t               
                                              
                                     
        
        alpha = F.dropout(alpha, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(alpha , index, ptr, size_i)  
        self._alpha = alpha

        out = value_j
        out = out * alpha.view(-1, self.heads, 1)
        
        return out


def unbatch(src, batch, dim: int = 0):




















    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)



def unbatch_edge_index(edge_index, batch):





















    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def compute_connectivity(edge_index, batch):                                                         

    edges_by_batch = unbatch_edge_index(edge_index, batch)

    nodes_counts = torch.unique(batch, return_counts=True)[1]

    connectivity = torch.tensor([nodes_in_largest_graph(e, n) for e, n in zip(edges_by_batch, nodes_counts)])
    isolation = torch.tensor([isolated_nodes(e, n) for e, n in zip(edges_by_batch, nodes_counts)])

    return connectivity, isolation


def nodes_in_largest_graph(edge_index, num_nodes):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    num_components, component = sp.csgraph.connected_components(adj)

    _, count = np.unique(component, return_counts=True)
    subset = np.in1d(component, count.argsort()[-1:])

    return subset.sum() / num_nodes


def isolated_nodes(edge_index, num_nodes):

    edge_attr = None

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    mask[edge_index.view(-1)] = 0

    return mask.sum() / num_nodes

def dropout_node(edge_index, p, num_nodes, batch, training):































    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask
    
    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    
                                             
    batch_tf = global_add_pool(node_mask.view(-1,1),batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []
    
    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)
            
    node_mask = torch.cat(node_mask_list)
    
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:






































    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask