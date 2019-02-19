import torch
import torch.nn as nn
import torch.nn.functional as F
# from tools.print_time_info import print_time_info

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5, act_func=F.relu, bias=False, sparse=True):
        super(GraphConvolution, self).__init__()
        '''
        暂时忽略sparse情况
        todo: sparse input support to save memory
        '''
        self.sparse = sparse
        self.act_func = act_func
        self.weights = nn.Parameter(torch.zeros([input_dim, output_dim], dtype=torch.float), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros([output_dim], dtype=torch.float), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.weights.data)

    def forward(self, inputs, adjacency_matrix):
        '''
        inputs: shape = [num_entity, embedding_dim]
        '''
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        outputs = torch.chain_matmul(adjacency_matrix, inputs, self.weights)
        # support = torch.mm(inputs, self.weights)
        # outputs = torch.spmm(inputs, support)
        if self.bias is not None:
            outputs += self.bias
        return outputs