import torch.nn as nn
from graph_completion.layers import GraphConvolution


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

