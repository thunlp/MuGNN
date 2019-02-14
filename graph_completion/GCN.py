import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5, act_func=F.relu, bias=False,):
        super(GCN, self).__init__()
        '''
        暂时忽略sparse情况
        todo: sparse input support to save memory
        '''
        self.bias = bias
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.weights = torch.zeros([input_dim, output_dim], requires_grad=True)
        # nn.init.xavier_uniform_(self.weights.data)
        if self.bias:
            self.bias_weights = torch.zeros([output_dim], requires_grad=True)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs, adjacent_matrix):
        '''
        inputs: shape = [num_entity, embedding_dim]
        '''
        if self.dropout_rate > 0:
            inputs = self.dropout(inputs)
        outputs = torch.chain_matmul(adjacent_matrix, inputs, self.weights)
        if self.bias:
            bias = self.bias_weights.expand(
                [1, self.bias_weights.size()[0]]).repeat([inputs.size()[0], 1])
            outputs += bias
        return outputs