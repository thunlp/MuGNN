import torch
import torch.nn as nn
import torch.nn.functional as F
# from tools.print_time_info import print_time_info

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(GraphConvolution, self).__init__()
        self.act_func = F.relu
        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.weights.data)

    def forward(self, inputs, adjacency_matrix):
        '''
        inputs: shape = [num_entity, embedding_dim]
        '''
        outputs = torch.chain_matmul(adjacency_matrix, inputs, self.weights)
        # support = torch.mm(inputs, self.weights)
        # outputs = torch.spmm(inputs, support)
        if self.bias is not None:
            outputs += self.bias
        return outputs

class DoubleEmbedding(nn.Module):
    def __init__(self, num_sr, num_tg, embedding_dim):
        super(DoubleEmbedding, self).__init__()
        self.embedding_sr = nn.Embedding(num_sr, embedding_dim)
        self.embedding_tg = nn.Embedding(num_tg, embedding_dim)
        nn.init.xavier_normal_(self.embedding_sr.weight.data)
        nn.init.xavier_normal_(self.embedding_tg.weight.data)

    def forward(self, sr_data, tg_data):
        return self.embedding_sr(sr_data), self.embedding_tg(tg_data)

    @property
    def weight(self):
        return self.embedding_sr.weight, self.embedding_tg.weight