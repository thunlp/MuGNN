import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_completion.layers import GraphConvolution, GraphMultiHeadAttLayer, RelAttGCN


class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, dim):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(num_ent, dim)
        self.rel_embeddings = nn.Embedding(num_rel, dim)
        nn.init.normal_(self.ent_embeddings, std=1. / math.sqrt(num_ent))
        nn.init.xavier_uniform_(self.rel_embeddings)

    @property
    def weight(self):
        self.ent_embeddings.weight = F.normalize(self.ent_embeddings.weight, dim=-1, p=2)
        self.rel_embeddings.weight = F.normalize(self.ent_embeddings.weight, dim=-1, p=2)
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=1, dim=-1)

    def forward(self, ent_embeddings, h_list, t_list, r_list):
        # shape = [num, 2*nega+1, embedding_dim]
        h = ent_embeddings(h_list)
        t = ent_embeddings(t_list)
        r = self.rel_embeddings(r_list)
        return h + r - t


class GAT(nn.Module):
    def __init__(self, dim_in, dim_out, nheads, layers, dropout_rate, alpha, sp, w_adj, cuda):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        assert dim_out % nheads == 0
        self.dropout = nn.Dropout(dropout_rate)
        self.multi_head_att_layers = nn.ModuleList()
        concat = True
        for i in range(layers):
            if i != 0:
                dim_in = dim_out
            if i == layers - 1:
                concat = False
            self.multi_head_att_layers.append(
                GraphMultiHeadAttLayer(dim_in, dim_out, nheads, dropout_rate, alpha, concat, sp, w_adj, cuda))

    def forward(self, x, adj):
        for att_layer in self.multi_head_att_layers:
            x = self.dropout(x)
            x = att_layer(x, adj)
        return x


class GATmGCN(GAT):
    def __init__(self, dim_in, dim_out, nheads, layers, dropout_rate, alpha, sp, w_adj, cuda):
        super(GATmGCN, self).__init__(dim_in, dim_out, nheads, layers, dropout_rate, alpha, sp, w_adj, cuda)
        self.relation_attention_gcns = nn.ModuleList()
        self.layers = layers
        for i in range(layers):
            self.relation_attention_gcns.append(RelAttGCN(dim_in, dim_out, cuda))
        self.pool = nn.MaxPool1d(2)


    def forward(self, x, adj):
        x1 = x
        x2 = x
        for i in range(self.layers):
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
            x1 = self.multi_head_att_layers[i](x1, adj)
            x2 = self.relation_attention_gcns[i](x2, adj)
        x = torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), dim=-1)
        x = self.pool(x).squeeze(-1)
        # x = (x1 + x2) / 2
        return x

class GCN(nn.Module):
    def __init__(self, dim, num_layer, dropout_rate=0.5, bias=False):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn_list = nn.ModuleList()
        for i in range(num_layer):
            self.gcn_list.append(GraphConvolution(dim, dim, bias))

    def forward(self, inputs, adj):
        graph_embedding = inputs
        for gcn in self.gcn_list:
            graph_embedding = self.dropout(gcn(graph_embedding, adj))
        return graph_embedding
