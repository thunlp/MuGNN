import torch.nn as nn
from graph_completion.layers import GraphConvolution, SpGraphMultiHeadAttLayer


class SpGAT(nn.Module):
    def __init__(self, dim_in, dim_out, nheads, layers, dropout_rate, alpha, cuda):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        assert dim_out % nheads == 0
        self.dropout = nn.Dropout(dropout_rate)
        self.multi_head_att_layers = nn.ModuleList()
        for i in range(layers):
            if i != 0:
                dim_in = dim_out
            self.multi_head_att_layers.append(
                SpGraphMultiHeadAttLayer(dim_in, dim_out // nheads, nheads, dropout_rate, alpha, True, cuda))

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
