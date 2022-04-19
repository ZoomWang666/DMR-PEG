import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj

device = torch.device('cuda:0')

class GNNat(torch.nn.Module):
    def __init__(self, n_output=1, num_feature_miRNA=2759, num_feature_drug=1005, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GNNat, self).__init__()

        print('GNNNat Loaded')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv2 = GCNConv(num_features_mol * 2, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 2)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 2, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.miRNA_fc1 = torch.nn.Linear(num_feature_miRNA, 1024)
        self.miRNA_fc2 = torch.nn.Linear(1024, 512)
        self.miRNA_fc3 = torch.nn.Linear(512, output_dim)

        self.drug_fc1 = torch.nn.Linear(num_feature_drug, 1024)
        self.drug_fc2 = torch.nn.Linear(1024, 512)
        self.drug_fc3 = torch.nn.Linear(512, output_dim)

        self.asso_conv1 = PEG_conv(in_feats_dim = 860, pos_dim = 128, out_feats_dim = 128,
                                use_formerinfo = False)
        self.asso_conv2 = PEG_conv(in_feats_dim = 128, pos_dim = 128, out_feats_dim = 128,
                                use_formerinfo = False)
        self.asso_conv3 = PEG_conv(in_feats_dim = 128, pos_dim = 128, out_feats_dim = 128,
                                use_formerinfo = False)
        #self.asso_conv1 = GCNConv(860, 128)
        #self.asso_conv2 = GCNConv(128, 128)
        #self.asso_conv3 = GCNConv(128, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(3 * output_dim, 512)
        #self.fc2 = nn.Linear(1024, 512)

        self.fc3 = nn.Linear(2 * output_dim, 512)
        #self.fc4 = nn.Linear(1024, 512)

        self.fc5 = nn.Linear(1024, 256)
        #self.fc6 = nn.Linear(512, 256)

        self.gtf1 = nn.Linear(512, 256, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.ctf1 = nn.Linear(512, 128)
        self.ctf2 = nn.Linear(512, 128)
        self.ctf3 = nn.Linear(128, 16)
        self.ctf4 = nn.Linear(128, 16)

        self.final_layer1 = nn.Linear(256 * 3, 128)
        self.final_layer2 = nn.Linear(128, 1)
        self.out = nn.Linear(2, self.n_output)

        self.mol_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.mol_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.mol_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.asso_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.asso_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.asso_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # 初始化
        self.mol_weight_1.data.fill_(0.5)
        self.mol_weight_2.data.fill_(0.333)
        self.mol_weight_3.data.fill_(0.25)
        self.asso_weight_1.data.fill_(0.5)
        self.asso_weight_2.data.fill_(0.333)
        self.asso_weight_3.data.fill_(0.25)

    def forward(self, data_mol, asso_adj, miRNA_feature, drug_feature,positional_encoding, id_map):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        mol_batch = mol_batch.cuda(device)
        # get protein input
        asso_x, asso_edge_index = asso_adj.x, asso_adj.edge_index

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_batch.size(), 'batch', target_batch.size())

        x1 = self.mol_conv1(mol_x, mol_edge_index)
        x1 = self.relu(x1)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x2 = self.mol_conv2(x1, mol_edge_index)
        x2 = self.relu(x2)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x3 = self.mol_conv3(x2, mol_edge_index)
        x3 = self.relu(x3)

        x = self.mol_weight_1 * x1 + self.mol_weight_2 * x2 + self.mol_weight_3 * x3
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x[id_map[0]]

        PEG_input = torch.cat((positional_encoding, asso_x), 1)


        asso1 = self.asso_conv1(PEG_input, asso_edge_index)
        #asso1 = self.relu(asso1)
        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        asso2 = self.asso_conv2(asso1, asso_edge_index)
        #asso2 = self.relu(asso2)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        asso3 = self.asso_conv3(asso2, asso_edge_index)
        #asso3 = self.relu(asso3)
        #print(self.asso_weight_1)
        #print(self.asso_weight_2)
        #print(self.asso_weight_3)

        asso1 = asso1[ : , 128: ]
        asso2 = asso2[ : , 128: ]
        asso3 = asso3[ : , 128: ]

        asso = self.asso_weight_1 * asso1 + self.asso_weight_2 * asso2 + self.asso_weight_3 * asso3



        mi = self.miRNA_fc1(miRNA_feature)
        mi = self.relu(mi)
        mi = self.dropout(mi)

        mi = self.miRNA_fc2(mi)
        mi = self.relu(mi)
        mi = self.dropout(mi)
        mi = self.miRNA_fc3(mi)
        mi = self.relu(mi)
        mi = self.dropout(mi)

        drug = self.drug_fc1(drug_feature)
        drug = self.relu(drug)
        drug = self.dropout(drug)

        drug = self.drug_fc2(drug)
        drug = self.relu(drug)
        drug = self.dropout(drug)
        drug = self.drug_fc3(drug)
        drug = self.relu(drug)
        drug = self.dropout(drug)

        # print(x.size(), xt.size())
        # concat
        drug_asso = torch.cat((x, asso[id_map[0]]), 1)
        drug_asso = torch.cat((drug_asso, drug[id_map[0]]), 1)
        mi_asso = torch.cat((mi[id_map[1]], asso[106 + id_map[1]]), 1)
        #drug_asso = torch.cat((x, drug[id_map[0]]), 1)
        #mi_asso = mi[id_map[1]]

        drug_pos = positional_encoding[id_map[0]]
        miRNA_pos = positional_encoding[id_map[1] + 106]
        #drug_asso = x + asso[id_map[0]] + drug[id_map[0]]
        #mi_asso = mi[id_map[1]] + asso[106 + id_map[1]]

        # add some dense layers
        drug_asso = self.fc1(drug_asso)
        drug_asso = self.relu(drug_asso)
        drug_asso = self.dropout(drug_asso)
        #drug_asso = self.fc2(drug_asso)
        #drug_asso = self.relu(drug_asso)
        #drug_asso = self.dropout(drug_asso)

        mi_asso = self.fc3(mi_asso)
        mi_asso = self.relu(mi_asso)
        mi_asso = self.dropout(mi_asso)
        #mi_asso = self.fc4(mi_asso)
        #mi_asso = self.relu(mi_asso)
        #mi_asso = self.dropout(mi_asso)

        drug_mi_mlp = torch.cat((drug_asso, mi_asso), 1)
        drug_mi_mlp = self.fc5(drug_mi_mlp)
        drug_mi_mlp = self.relu(drug_mi_mlp)
        drug_mi_mlp = self.dropout(drug_mi_mlp)
        #drug_mi_mlp = self.fc6(drug_mi_mlp)
        #drug_mi_mlp = self.relu(drug_mi_mlp)
        #drug_mi_mlp = self.dropout(drug_mi_mlp)

        drug_mi_gtf = torch.mul(drug_asso, mi_asso)
        drug_mi_gtf = self.gtf1(drug_mi_gtf)
        drug_mi_gtf = self.sigmoid(drug_mi_gtf)
        drug_mi_gtf = self.dropout(drug_mi_gtf)

        drug_mi_ctf = torch.bmm(drug_asso.unsqueeze(2), mi_asso.unsqueeze(1))

        drug_mi_ctf = self.ctf1(drug_mi_ctf)
        drug_mi_ctf = torch.transpose(drug_mi_ctf, 1, 2)
        drug_mi_ctf = self.ctf2(drug_mi_ctf)
        drug_mi_ctf = torch.transpose(drug_mi_ctf, 1, 2)
        drug_mi_ctf = self.ctf3(drug_mi_ctf)
        drug_mi_ctf = torch.transpose(drug_mi_ctf, 1, 2)
        drug_mi_ctf = self.ctf4(drug_mi_ctf)
        drug_mi_ctf = torch.transpose(drug_mi_ctf, 1, 2)

        drug_mi_ctf = torch.flatten(drug_mi_ctf, start_dim=1)
        all_drug_mi = torch.cat((drug_mi_mlp, drug_mi_gtf), 1)
        all_drug_mi = torch.cat((all_drug_mi, drug_mi_ctf), 1)

        embedding = self.final_layer1(all_drug_mi)
        embedding = self.final_layer2(embedding)
        rel_dis = ((drug_pos - miRNA_pos)**2).sum(dim=-1, keepdim=True)

        mix = torch.cat((embedding, rel_dis), 1)

        out = self.out(mix)
        return out


import math

from typing import Optional, Tuple
from torch_geometric.typing import Adj, Size, OptTensor, Tensor
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing


    
class PEG_conv(MessagePassing):
    
    r"""The PEG layer from the `"Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks" <https://arxiv.org/abs/2203.00199>`_ paper
    
    
    Args:
        in_feats_dim (int): Size of input node features.
        pos_dim (int): Size of positional encoding.
        out_feats_dim (int): Size of output node features.
        edge_mlp_dim (int): We use MLP to make one to one mapping between the relative information and edge weight. 
                            edge_mlp_dim represents the hidden units dimension in the MLP. (default: 32)
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        use_formerinfo (bool): Whether to use previous layer's output to update node features.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_feats_dim: int, pos_dim: int, out_feats_dim: int, edge_mlp_dim: int = 32,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, use_formerinfo: bool = False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(PEG_conv, self).__init__(**kwargs)

        self.in_feats_dim = in_feats_dim
        self.out_feats_dim = out_feats_dim
        self.pos_dim = pos_dim
        self.use_formerinfo = use_formerinfo
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.edge_mlp_dim = edge_mlp_dim

        self._cached_edge_index = None
        self._cached_adj_t = None
        
        self.weight_withformer = Parameter(torch.Tensor(in_feats_dim + in_feats_dim, out_feats_dim))
        self.weight_noformer = Parameter(torch.Tensor(in_feats_dim, out_feats_dim))
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_mlp_dim),
            nn.Linear(edge_mlp_dim, 1),
            nn.Sigmoid()
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_feats_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_withformer)
        glorot(self.weight_noformer)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self,x: Tensor, edge_index: Adj, 
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, feats.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, feats.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        else:
            print('We normalize the adjacent matrix in PEG.')
        
        
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # pos: l2 norms
        hidden_out, coors_out = self.propagate(edge_index, x = feats, edge_weight=edge_weight, pos=rel_dist, coors=coors,
                             size=None)
        
        

        if self.bias is not None:
            hidden_out += self.bias

        return torch.cat([coors_out, hidden_out], dim=-1)


    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor, pos) -> Tensor:
        PE_edge_weight = self.edge_mlp(pos)
        return x_j if edge_weight is None else PE_edge_weight * edge_weight.view(-1, 1) * x_j
    
    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)


        m_i = self.aggregate(m_ij, **aggr_kwargs)
        
        
        coors_out = kwargs["coors"]
        hidden_feats = kwargs["x"]
        if self.use_formerinfo:
            hidden_out = torch.cat([hidden_feats, m_i], dim = -1)
            hidden_out = hidden_out @ self.weight_withformer
        else:
            hidden_out = m_i
            hidden_out = hidden_out @ self.weight_noformer
        



        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)



    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

        
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)