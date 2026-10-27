
import torch
from torch_geometric.nn import global_add_pool
from torch.nn import Embedding, Linear, LayerNorm, MultiheadAttention, Dropout
import torch.nn.functional as F
from models.layers import MLP, Drug_PNAConv, Protein_PNAConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import GraphConv
import math

try:
    from torch_geometric.nn import GATv2Conv as ContactConv
    CONTACT_CONV_NAME = "GATv2Conv"
except ImportError:
    from torch_geometric.nn import GATConv as ContactConv
    CONTACT_CONV_NAME = "GATConv"


class ESMCMultiLayerFusion(torch.nn.Module):

    def __init__(self, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.layer_weights = torch.nn.Parameter(torch.zeros(num_layers))
        self.gamma = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.layer_weights)
        with torch.no_grad():
            self.gamma.fill_(1.0)

    def forward(self, hidden_states):
        if isinstance(hidden_states, (list, tuple)):
            hidden_states = torch.stack(hidden_states, dim=1)
        if hidden_states.dim() != 3 or hidden_states.size(1) != self.num_layers:
            raise ValueError(
                f"Expected ESMC states [residues,{self.num_layers},hidden], "
                f"got {tuple(hidden_states.shape)}"
            )
        weights = torch.softmax(self.layer_weights, dim=0)
        fused = torch.sum(weights[None, :, None] * hidden_states, dim=1)
        return self.gamma * fused


class ContactGraphAdapter(torch.nn.Module):

    def __init__(self, hidden_dim=200, dropout=0.1):
        super().__init__()
        self.contact_gnn = ContactConv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            heads=1, concat=False, add_self_loops=False
        )
        self.contact_norm = LayerNorm(hidden_dim)
        self.output_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)
        self.gate_linear = Linear(hidden_dim * 2, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.contact_gnn.reset_parameters()
        self.contact_norm.reset_parameters()
        self.output_norm.reset_parameters()
        self.gate_linear.reset_parameters()
        torch.nn.init.constant_(self.gate_linear.bias, -1.0)

    def forward(self, x, edge_index):
        contact_feat = self.contact_gnn(x, edge_index)
        contact_feat = self.contact_norm(self.dropout(F.gelu(contact_feat)))
        gate = torch.sigmoid(self.gate_linear(torch.cat([x, contact_feat], dim=-1)))
        return self.output_norm(x + gate * contact_feat)


def _scatter_mean(values, index, size):

    out = values.new_zeros((size, values.size(-1)))
    count = values.new_zeros((size, 1))
    if index.numel() > 0:
        out.index_add_(0, index, values)
        count.index_add_(
            0, index,
            values.new_ones((index.numel(), 1)),
        )
    return out / count.clamp(min=1)


class PointEdgeTriangleTDL(torch.nn.Module):








    def __init__(self, hidden_dim=200, dropout=0.1):
        super().__init__()
        self.edge_encoder = Linear(hidden_dim * 2 + 1, hidden_dim)
        self.triangle_encoder = Linear(hidden_dim * 2, hidden_dim)
        self.triangle_to_edge = Linear(hidden_dim, hidden_dim)
        self.edge_to_node = Linear(hidden_dim, hidden_dim)
        self.triangle_to_node = Linear(hidden_dim, hidden_dim)
        self.node_update = Linear(hidden_dim * 3, hidden_dim)
        self.update_gate = Linear(hidden_dim * 2, hidden_dim)
        self.edge_norm = LayerNorm(hidden_dim)
        self.triangle_norm = LayerNorm(hidden_dim)
        self.node_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for module in (
            self.edge_encoder, self.triangle_encoder,
            self.triangle_to_edge, self.edge_to_node,
            self.triangle_to_node, self.node_update, self.update_gate,
        ):
            module.reset_parameters()
        self.edge_norm.reset_parameters()
        self.triangle_norm.reset_parameters()
        self.node_norm.reset_parameters()
        torch.nn.init.constant_(self.update_gate.bias, -1.0)

    def forward(self, node_x, directed_edge_index, directed_edge_weight,
                triangle_index):
        num_nodes = node_x.size(0)
        if directed_edge_index.numel() == 0 or num_nodes == 0:
            return node_x

        src, dst = directed_edge_index.long()
        non_self = src != dst
        src, dst = src[non_self], dst[non_self]
        if src.numel() == 0:
            return node_x

        edge_u = torch.minimum(src, dst)
        edge_v = torch.maximum(src, dst)
        directed_keys = edge_u * num_nodes + edge_v
        edge_keys, inverse = torch.unique(
            directed_keys, sorted=True, return_inverse=True
        )
        edge_u = torch.div(edge_keys, num_nodes, rounding_mode="floor")
        edge_v = torch.remainder(edge_keys, num_nodes)
        num_edges = edge_keys.numel()

        weights = directed_edge_weight.reshape(-1).to(node_x.dtype)
        if weights.numel() == directed_edge_index.size(1):
            weights = weights[non_self]
        else:
            weights = node_x.new_ones(src.numel())
        weight_sum = node_x.new_zeros(num_edges)
        weight_count = node_x.new_zeros(num_edges)
        weight_sum.index_add_(0, inverse, weights)
        weight_count.index_add_(0, inverse, torch.ones_like(weights))
        edge_weight = (weight_sum / weight_count.clamp(min=1)).unsqueeze(-1)

        endpoint_mean = 0.5 * (node_x[edge_u] + node_x[edge_v])
        endpoint_delta = torch.abs(node_x[edge_u] - node_x[edge_v])
        edge_x = self.edge_norm(F.gelu(self.edge_encoder(torch.cat(
            [endpoint_mean, endpoint_delta, edge_weight], dim=-1
        ))))

        triangle_node_message = node_x.new_zeros(node_x.shape)
        if triangle_index is not None and triangle_index.numel() > 0:
            tri = triangle_index.long()
            valid = (tri >= 0).all(dim=0) & (tri < num_nodes).all(dim=0)
            tri = tri[:, valid]
        else:
            tri = node_x.new_empty((3, 0), dtype=torch.long)

        if tri.numel() > 0:
            a, b, c = tri[0], tri[1], tri[2]
            boundary_keys = torch.cat([
                torch.minimum(a, b) * num_nodes + torch.maximum(a, b),
                torch.minimum(a, c) * num_nodes + torch.maximum(a, c),
                torch.minimum(b, c) * num_nodes + torch.maximum(b, c),
            ])
            boundary_edge_ids = torch.searchsorted(edge_keys, boundary_keys)
            boundary_ok = boundary_edge_ids < num_edges
            boundary_ok = boundary_ok & (
                edge_keys[boundary_edge_ids.clamp(max=num_edges - 1)]
                == boundary_keys
            )
            if not bool(boundary_ok.all()):
                raise RuntimeError("Triangle boundary is absent from contact edges")

            num_triangles = tri.size(1)
            boundary_feat = edge_x[boundary_edge_ids].view(
                3, num_triangles, -1
            ).mean(dim=0)
            triangle_node_mean = node_x[tri].mean(dim=0)
            triangle_x = self.triangle_norm(F.gelu(self.triangle_encoder(
                torch.cat([triangle_node_mean, boundary_feat], dim=-1)
            )))

            tri_to_edge = _scatter_mean(
                triangle_x.repeat(3, 1), boundary_edge_ids, num_edges
            )
            edge_x = self.edge_norm(
                edge_x + self.dropout(self.triangle_to_edge(tri_to_edge))
            )
            triangle_node_message = _scatter_mean(
                triangle_x.repeat(3, 1), tri.reshape(-1), num_nodes
            )

        edge_node_index = torch.cat([edge_u, edge_v])
        edge_node_message = _scatter_mean(
            edge_x.repeat(2, 1), edge_node_index, num_nodes
        )
        update = F.gelu(self.node_update(torch.cat([
            node_x,
            self.edge_to_node(edge_node_message),
            self.triangle_to_node(triangle_node_message),
        ], dim=-1)))
        update = self.dropout(update)
        gate = torch.sigmoid(self.update_gate(torch.cat([node_x, update], dim=-1)))
        return self.node_norm(node_x + gate * update)


class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len):
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        cos = self.cos_cached[:, :, :seq_len, :].to(x.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(x.dtype)
        x_rot = self._rotate_half(x)
        return x * cos + x_rot * sin

    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class RoPESelfAttention(torch.nn.Module):

    def __init__(self, hidden_dim=200, heads=5, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = x.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = x.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        q = q.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        k = k.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        v = x
        attn_output = self._rope_attention(q, k, v, mask)
        return self.norm(x + attn_output)

    def _rope_attention(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)


class ResidualGatedCrossAttentionBlock(torch.nn.Module):















    def __init__(self, hidden_dim=200, heads=5, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

                                                                     
        self.res_proj = Linear(hidden_dim, hidden_dim)
        self.atom_proj = Linear(hidden_dim, hidden_dim)
        self.cross_attn = MultiheadAttention(hidden_dim, heads, dropout=0.1, batch_first=True)
        self.res_norm = LayerNorm(hidden_dim)

                          
        self.res_gate_linear = Linear(hidden_dim * 2, hidden_dim)
                                                         
        torch.nn.init.zeros_(self.res_gate_linear.weight)
        torch.nn.init.constant_(self.res_gate_linear.bias, -1.0)

                                                                  
        self.atom_proj_rev = Linear(hidden_dim, hidden_dim)
        self.res_proj_rev = Linear(hidden_dim, hidden_dim)
        self.cross_attn_rev = MultiheadAttention(hidden_dim, heads, dropout=0.1, batch_first=True)
        self.atom_norm = LayerNorm(hidden_dim)

                       
        self.atom_gate_linear = Linear(hidden_dim * 2, hidden_dim)
                                                         
        torch.nn.init.zeros_(self.atom_gate_linear.weight)
        torch.nn.init.constant_(self.atom_gate_linear.bias, -1.0)

                 
        self.dropout = Dropout(dropout)

                                
        self.res_gate_mean = 0.0
        self.atom_gate_mean = 0.0
        self.res_gate_std = 0.0
        self.atom_gate_std = 0.0

    def forward(self, residue_padded, atom_padded):











                                              
        Q = self.res_proj(residue_padded)
        KV = self.atom_proj(atom_padded)
        attn_out_prot, attn_weights_prot = self.cross_attn(Q, KV, KV)

              
        gate_input_res = torch.cat([residue_padded, attn_out_prot], dim=-1)                    
        gate_res = torch.sigmoid(self.res_gate_linear(gate_input_res))                  
        gated_attn_res = gate_res * attn_out_prot

                              
        residue_padded = self.res_norm(residue_padded + self.dropout(gated_attn_res))

                               
        self.res_gate_mean = gate_res.mean().item()
        self.res_gate_std = gate_res.std().item()

                                           
        Q_rev = self.atom_proj_rev(atom_padded)
        KV_rev = self.res_proj_rev(residue_padded)
        attn_out_mol, attn_weights_mol = self.cross_attn_rev(Q_rev, KV_rev, KV_rev)

              
        gate_input_atom = torch.cat([atom_padded, attn_out_mol], dim=-1)                    
        gate_atom = torch.sigmoid(self.atom_gate_linear(gate_input_atom))                  
        gated_attn_atom = gate_atom * attn_out_mol

                              
        atom_padded = self.atom_norm(atom_padded + self.dropout(gated_attn_atom))

                               
        self.atom_gate_mean = gate_atom.mean().item()
        self.atom_gate_std = gate_atom.std().item()

        return residue_padded, atom_padded, attn_weights_prot, attn_weights_mol


class ESMCMultiLayerTDLContactModel(torch.nn.Module):



    def __init__(self, mol_deg, prot_deg,
                 mol_in_channels=384,
                 prot_in_channels=34, prot_evo_channels=1280,
                 hidden_channels=200,
                 pre_layers=2, post_layers=1,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'linear'],
                 total_layer=3,
                 K=[5, 10, 20],
                 t=1,
                 heads=5,
                 dropout=0,
                 dropout_attn_score=0.2,
                 regression_head=True,
                 classification_head=False,
                 multiclassification_head=0,
                 device='cuda:0',
                 use_rope=True,
                 use_residual_gate=True,
                 num_esmc_layers=4):
        super().__init__()
        self.total_layer = total_layer
        self.use_rope = use_rope
        self.use_residual_gate = use_residual_gate
        self.hidden_channels = hidden_channels

        if (regression_head or classification_head) == False:
            raise Exception('must have one objective')

        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head

                                                            
        self.atom_type_encoder = Embedding(20, hidden_channels)
        self.atom_feat_encoder = MLP([mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True)

                                               
        self.esmc_layer_fusion = ESMCMultiLayerFusion(num_layers=num_esmc_layers)
        self.prot_evo = MLP([prot_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_aa = MLP([prot_in_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.contact_adapter = ContactGraphAdapter(hidden_channels, dropout=0.1)
        self.topological_adapter = PointEdgeTriangleTDL(hidden_channels, dropout=0.1)

                                                            
        if self.use_rope:
            self.rope_self_attn = RoPESelfAttention(
                hidden_dim=hidden_channels,
                heads=heads,
                dropout=dropout,
                max_seq_len=2048
            )

                                           
        self.mol_convs = torch.nn.ModuleList()
        self.prot_convs = torch.nn.ModuleList()
        self.mol_gn2 = torch.nn.ModuleList()
        self.prot_gn2 = torch.nn.ModuleList()
        self.prot_edge_dim = hidden_channels

        for idx in range(total_layer):
            self.mol_convs.append(Drug_PNAConv(
                mol_deg, hidden_channels, edge_channels=hidden_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators, scalers=scalers,
                num_towers=heads, dropout=dropout
            ))
            self.prot_convs.append(Protein_PNAConv(
                prot_deg, hidden_channels, edge_channels=hidden_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators, scalers=scalers,
                num_towers=heads, dropout=dropout
            ))
            self.mol_gn2.append(GraphNorm(hidden_channels))
            self.prot_gn2.append(GraphNorm(hidden_channels))

                                               
        self.clique_encoder = Embedding(4, hidden_channels)
        self.fragment_conv = GraphConv(hidden_channels, hidden_channels)
        self.fragment_norm = GraphNorm(hidden_channels)
        self.res_frag_proj = Linear(hidden_channels, hidden_channels)
        self.frag_proj = Linear(hidden_channels, hidden_channels)
        self.res_frag_cross_attn = MultiheadAttention(hidden_channels, heads, dropout=0.1, batch_first=True)
        self.res_frag_norm = LayerNorm(hidden_channels)
        self.frag_importance_proj = MLP([hidden_channels, hidden_channels, 1])
        self.clique_global_proj = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)

                                               
        self.num_interactions = 3
        if self.use_residual_gate:
                                                 
            self.cross_attn_blocks = torch.nn.ModuleList()
            for _ in range(self.num_interactions):
                self.cross_attn_blocks.append(ResidualGatedCrossAttentionBlock(
                    hidden_dim=hidden_channels, heads=heads, dropout=dropout
                ))
        else:
                                          
            self.res_proj_list = torch.nn.ModuleList()
            self.atom_proj_list = torch.nn.ModuleList()
            self.cross_attn_list = torch.nn.ModuleList()
            self.atom_proj_rev_list = torch.nn.ModuleList()
            self.res_proj_rev_list = torch.nn.ModuleList()
            self.cross_attn_rev_list = torch.nn.ModuleList()
            self.interaction_norm_prot = torch.nn.ModuleList()
            self.interaction_norm_mol = torch.nn.ModuleList()

            for _ in range(self.num_interactions):
                self.res_proj_list.append(Linear(hidden_channels, hidden_channels))
                self.atom_proj_list.append(Linear(hidden_channels, hidden_channels))
                self.cross_attn_list.append(MultiheadAttention(hidden_channels, heads, dropout=0.1, batch_first=True))
                self.atom_proj_rev_list.append(Linear(hidden_channels, hidden_channels))
                self.res_proj_rev_list.append(Linear(hidden_channels, hidden_channels))
                self.cross_attn_rev_list.append(MultiheadAttention(hidden_channels, heads, dropout=0.1, batch_first=True))
                self.interaction_norm_prot.append(LayerNorm(hidden_channels))
                self.interaction_norm_mol.append(LayerNorm(hidden_channels))

                                                      
        self.pair_res_proj = Linear(hidden_channels, hidden_channels)
        self.pair_atom_proj = Linear(hidden_channels, hidden_channels)
        self.pair_interaction_mlp = MLP([hidden_channels * 4, hidden_channels * 2, hidden_channels], out_norm=True)
        self.pair_score_head = MLP([hidden_channels, hidden_channels, 1])

                             
        self.fp_encoder = MLP([1024, hidden_channels * 2, hidden_channels], out_norm=True)

                                      
        self.binding_head = MLP([hidden_channels, hidden_channels, 1])

                      
        self.mol_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)

                              
        if self.regression_head:
            self.reg_out = MLP([hidden_channels * 4, hidden_channels, 1])
        if self.classification_head:
            self.cls_out = MLP([hidden_channels * 4, hidden_channels, 1])
        if self.multiclassification_head:
            self.mcls_out = MLP([hidden_channels * 4, hidden_channels, multiclassification_head])

        self.dropout = dropout
        self.device = device

    def reset_parameters(self):
        self.atom_type_encoder.reset_parameters()
        self.atom_feat_encoder.reset_parameters()
        self.prot_evo.reset_parameters()
        self.prot_aa.reset_parameters()
        self.esmc_layer_fusion.reset_parameters()
        self.contact_adapter.reset_parameters()
        self.topological_adapter.reset_parameters()

        for idx in range(self.total_layer):
            self.mol_convs[idx].reset_parameters()
            self.prot_convs[idx].reset_parameters()
            self.mol_gn2[idx].reset_parameters()
            self.prot_gn2[idx].reset_parameters()

        self.fragment_conv.reset_parameters()
        self.fragment_norm.reset_parameters()
        self.res_frag_proj.reset_parameters()
        self.frag_proj.reset_parameters()
        self.res_frag_norm.reset_parameters()
        self.frag_importance_proj.reset_parameters()
        self.clique_global_proj.reset_parameters()

        self.pair_res_proj.reset_parameters()
        self.pair_atom_proj.reset_parameters()
        self.pair_interaction_mlp.reset_parameters()
        self.pair_score_head.reset_parameters()

                                      
        if self.use_residual_gate:
            for block in self.cross_attn_blocks:
                                                                  
                block.res_proj.reset_parameters()
                block.atom_proj.reset_parameters()
                block.cross_attn._reset_parameters()
                block.atom_proj_rev.reset_parameters()
                block.res_proj_rev.reset_parameters()
                block.cross_attn_rev._reset_parameters()
        else:
            for i in range(self.num_interactions):
                self.res_proj_list[i].reset_parameters()
                self.atom_proj_list[i].reset_parameters()
                self.atom_proj_rev_list[i].reset_parameters()
                self.res_proj_rev_list[i].reset_parameters()

        self.fp_encoder.reset_parameters()
        self.binding_head.reset_parameters()
        self.mol_out.reset_parameters()
        self.prot_out.reset_parameters()

        if self.regression_head:
            self.reg_out.reset_parameters()
        if self.classification_head:
            self.cls_out.reset_parameters()
        if self.multiclassification_head:
            self.mcls_out.reset_parameters()

    def forward(self,
                mol_x, mol_x_feat, bond_x, atom_edge_index,
                clique_x, clique_edge_index, atom2clique_index,
                residue_x, residue_evo_x, residue_edge_index, residue_edge_weight,
                fingerprint=None,
                mol_batch=None, prot_batch=None, clique_batch=None,
                residue_triangle_index=None):

        reg_pred = None
        cls_pred = None
        mcls_pred = None
        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device)

                                                 
                                                                    
        esmc_fused = self.esmc_layer_fusion(residue_evo_x)
        assert torch.isfinite(esmc_fused).all(), "Non-finite ESMC fused features"
        assert esmc_fused.size(0) == residue_x.size(0), "ESMC/residue row mismatch"
        residue_x = self.prot_aa(residue_x) + self.prot_evo(esmc_fused)
        assert residue_x.size(0) == residue_evo_x.size(0)

                                                                        
        assert residue_edge_index.dtype == torch.long
        assert residue_edge_index.device == residue_x.device
        if residue_edge_index.numel() > 0:
            assert int(residue_edge_index.min()) >= 0
            assert int(residue_edge_index.max()) < residue_x.size(0)
            assert torch.equal(
                prot_batch[residue_edge_index[0]],
                prot_batch[residue_edge_index[1]],
            ), "Cross-protein edges detected in batched contact graph"
        residue_x = self.contact_adapter(residue_x, residue_edge_index)
        residue_x = self.topological_adapter(
            residue_x,
            residue_edge_index,
            residue_edge_weight,
            residue_triangle_index,
        )
        assert torch.isfinite(residue_x).all(), "Non-finite contact-adapted features"

                                                     
        if self.use_rope:
            residue_padded_for_rope, rope_mask = to_dense_batch(residue_x, prot_batch)
            padding_mask = ~rope_mask
            residue_padded_for_rope = self.rope_self_attn(residue_padded_for_rope, mask=padding_mask)
            residue_x = residue_padded_for_rope[rope_mask]

                                                  
                                                                             
                                                                           
        atom_x = self.atom_type_encoder(mol_x.view(-1)) + self.atom_feat_encoder(mol_x_feat)

                                                            
        for idx in range(self.total_layer):
            atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index)
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)
            atom_x = self.mol_gn2[idx](atom_x, mol_batch)
            residue_x = self.prot_gn2[idx](residue_x, prot_batch)

                                               
                                                              
        clique_feat = self.clique_encoder(clique_x.view(-1))
        if clique_edge_index.numel() > 0:
            clique_feat = self.fragment_conv(clique_feat, clique_edge_index)
        clique_feat = self.fragment_norm(clique_feat, clique_batch)

        residue_padded, prot_mask = to_dense_batch(residue_x, prot_batch)
        clique_padded, clique_mask = to_dense_batch(clique_feat, clique_batch)

        Q_rf = self.res_frag_proj(residue_padded)
        KV_rf = self.frag_proj(clique_padded)
        attn_out_rf, _ = self.res_frag_cross_attn(Q_rf, KV_rf, KV_rf)
        residue_padded = self.res_frag_norm(residue_padded + attn_out_rf)

        frag_importance = self.frag_importance_proj(clique_feat)
        frag_importance = torch.sigmoid(frag_importance).squeeze(-1)

        if atom2clique_index.numel() > 0:
            atom2clique_row = atom2clique_index[0]
            atom2clique_col = atom2clique_index[1]
            atom_weight_per_edge = frag_importance[atom2clique_col]
            atom_weight_sum = torch.zeros(atom_x.size(0), device=self.device)
            atom_weight_sum.scatter_add_(0, atom2clique_row, atom_weight_per_edge)
            atom_count = torch.zeros(atom_x.size(0), device=self.device)
            atom_count.scatter_add_(0, atom2clique_row, torch.ones_like(atom_weight_per_edge))
            atom_weight_avg = atom_weight_sum / atom_count.clamp(min=1)
        else:
            atom_weight_avg = torch.ones(atom_x.size(0), device=self.device)

        atom_x = atom_x * (1 + atom_weight_avg.unsqueeze(-1))

        batch_size = mol_batch.max().item() + 1 if mol_batch is not None else 1
        clique_global_feat = global_add_pool(clique_feat, clique_batch, size=batch_size)
        clique_global_feat = self.clique_global_proj(clique_global_feat)

                                               
        atom_padded, atom_mask = to_dense_batch(atom_x, mol_batch, batch_size=batch_size)

                               
        gate_stats = {}

        if self.use_residual_gate:
                                             
            for i in range(self.num_interactions):
                residue_padded, atom_padded, attn_weights_prot, attn_weights_mol = \
                    self.cross_attn_blocks[i](residue_padded, atom_padded)

                                       
                block = self.cross_attn_blocks[i]
                gate_stats[f'res_gate_mean_{i}'] = block.res_gate_mean
                gate_stats[f'atom_gate_mean_{i}'] = block.atom_gate_mean
                gate_stats[f'res_gate_std_{i}'] = block.res_gate_std
                gate_stats[f'atom_gate_std_{i}'] = block.atom_gate_std
        else:
                                      
            for i in range(self.num_interactions):
                Q = self.res_proj_list[i](residue_padded)
                KV = self.atom_proj_list[i](atom_padded)
                attn_out_prot, attn_weights_prot = self.cross_attn_list[i](Q, KV, KV)

                Q_rev = self.atom_proj_rev_list[i](atom_padded)
                KV_rev = self.res_proj_rev_list[i](residue_padded)
                attn_out_mol, attn_weights_mol = self.cross_attn_rev_list[i](Q_rev, KV_rev, KV_rev)

                residue_padded = self.interaction_norm_prot[i](residue_padded + attn_out_prot)
                atom_padded = self.interaction_norm_mol[i](atom_padded + attn_out_mol)

                                       
        prot_mask_f = prot_mask.unsqueeze(-1).float()
        atom_mask_f = atom_mask.unsqueeze(-1).float()
        prot_graph_feat = (residue_padded * prot_mask_f).sum(dim=1) / prot_mask_f.sum(dim=1).clamp(min=1)
        mol_pool_feat = (atom_padded * atom_mask_f).sum(dim=1) / atom_mask_f.sum(dim=1).clamp(min=1)

        mol_feat = self.mol_out(mol_pool_feat + clique_global_feat)
        prot_graph_feat = self.prot_out(prot_graph_feat)

                                                      
        B, L, D = residue_padded.shape
        _, A, _ = atom_padded.shape

        res_proj = self.pair_res_proj(residue_padded)
        atom_proj = self.pair_atom_proj(atom_padded)

        pair_score = torch.einsum('bld,bad->bla', res_proj, atom_proj)
        pair_score = pair_score / (D ** 0.5)

        prot_mask_expanded = prot_mask.unsqueeze(2).expand(B, L, A)
        atom_mask_expanded = atom_mask.unsqueeze(1).expand(B, L, A)
        pair_mask = prot_mask_expanded & atom_mask_expanded

        pair_score = pair_score.masked_fill(~pair_mask, float('-inf'))
        pair_weight = F.softmax(pair_score.view(B, -1), dim=-1).view(B, L, A)
        pair_weight = torch.nan_to_num(pair_weight, nan=0.0)

        res_weighted = torch.einsum('bla,bld->bd', pair_weight, residue_padded)
        atom_weighted = torch.einsum('bla,bad->bd', pair_weight, atom_padded)
        res_at_atom = torch.einsum('bla,bld->bad', pair_weight, residue_padded)
        interaction_prod = torch.einsum('bad,bad->bd', res_at_atom, atom_padded)
        interaction_diff = torch.abs(res_weighted - atom_weighted)

        interaction_feat = torch.cat([res_weighted, atom_weighted, interaction_prod, interaction_diff], dim=-1)
        interaction_feat = self.pair_interaction_mlp(interaction_feat)

                              
        if fingerprint is not None:
            fp_feat = self.fp_encoder(fingerprint)
        else:
            fp_feat = torch.zeros(mol_pool_feat.size(0), 200, device=self.device)

                              
        final_feat = torch.cat([mol_feat, prot_graph_feat, fp_feat, interaction_feat], dim=-1)

                                 
        binding_logits = self.binding_head(residue_padded).squeeze(-1)

        if self.regression_head:
            reg_pred = self.reg_out(final_feat)
        if self.classification_head:
            cls_pred = self.cls_out(final_feat)
        if self.multiclassification_head:
            mcls_pred = self.mcls_out(final_feat)

        zero = torch.tensor(0.).to(self.device)
        attention_dict = {
            'cross_attn_weights_prot': attn_weights_prot,
            'cross_attn_weights_mol': attn_weights_mol,
            'drug_atom_index': mol_batch,
            'drug_clique_index': clique_batch,
            'protein_residue_index': prot_batch,
            'mol_feature': mol_feat,
            'prot_feature': prot_graph_feat,
            'interaction_fingerprint': final_feat,
            'gate_stats': gate_stats,
        }

        return reg_pred, cls_pred, mcls_pred, zero, zero, zero, binding_logits, attention_dict, prot_mask


def _rbf(D, D_max, D_count, device):
    D_min = 0.
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, -1)
    D_mu = D_mu[0]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
