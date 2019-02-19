import torch
import torch.nn as nn
import torch.nn.functional as F
# from tools.print_time_info import print_time_info

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, act_func=F.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.act_func = act_func
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
