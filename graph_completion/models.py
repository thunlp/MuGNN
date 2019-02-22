import torch
import torch.nn as nn
from graph_completion.layers import GraphConvolution, GraphMultiHeadAttLayer, DoubleEmbedding


class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, dim):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(num_ent, dim, _weight=torch.zeros((num_ent, dim), dtype=torch.double))
        self.rel_embeddings = nn.Embedding(num_rel, dim, _weight=torch.zeros((num_rel, dim), dtype=torch.double))
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=1, dim=-1)

    def loss(self, p_score, n_score):
        y = torch.Tensor([-1]).cuda()
        return self.criterion(p_score, n_score, y)

    def forward(self, h_list, t_list, r_list):
        # shape = [num, 2*nega+1, embedding_dim]
        h = self.ent_embeddings(h_list)
        t = self.ent_embeddings(t_list)
        r = self.rel_embeddings(r_list)
        return h + r - t


class GAT(nn.Module):
    def __init__(self, dim_in, dim_out, nheads, layers, dropout_rate, alpha, sp, cuda):
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
                GraphMultiHeadAttLayer(dim_in, dim_out // nheads, nheads, dropout_rate, alpha, concat, sp, cuda))

    def forward(self, x, adj):
        for att_layer in self.multi_head_att_layers:
            x = self.dropout(x)
            x = att_layer(x, adj)
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
