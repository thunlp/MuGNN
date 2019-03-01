import torch, math
import torch.nn as nn
import torch.nn.functional as F


# [gat(dim_in, dim_out_s, dropout, alpha, concat, cuda) for _ in range(nheads)])

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, cuda=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, inputs, adj):
        h = torch.mm(inputs, self.W)
        N = h.size()[0]

        a_input = torch.cat((h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)), dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GraphMultiHeadAttLayer(nn.Module):
    def __init__(self, dim_in, dim_out_s, nheads, dropout, alpha, concat, sp, w_adj, cuda):
        super(GraphMultiHeadAttLayer, self).__init__()
        self.attentions = nn.ModuleList(
            [SpGraphAttentionLayer(dim_in, dim_out_s, dropout, alpha, concat, w_adj, cuda) for _ in range(nheads)])

    def forward(self, inputs, adj):
        # inputs shape = [num, dim]
        outputs = torch.cat(tuple([att(inputs, adj).unsqueeze(-1) for att in self.attentions]), dim=-1)
        # shape = [num, dim, nheads]
        outputs =  torch.mean(outputs, dim=-1)
        return outputs


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, w_adj=True, cuda=True, residual=False):
        super(SpGraphAttentionLayer, self).__init__()
        assert in_features == out_features
        self.w_adj = w_adj
        self.is_cuda = cuda
        self.concat = concat
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(out_features,), dtype=torch.float))
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features), dtype=torch.float32))
        nn.init.ones_(self.W.data)
        stdv = 1. / math.sqrt(in_features * 2)
        nn.init.uniform_(self.a.data, -stdv, stdv)
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.special_spmm = SpecialSpmm()


    def forward(self, inputs, adj):
        # input shape shape [num_entity, embedding_dim]
        N = inputs.size()[0]

        # edge = adj.nonzero().t()
        edge = adj.indices()
        # h = torch.mm(inputs, self.W)
        # h shape: [num_entity, out_dim]
        h = torch.mul(inputs, self.W) # todo: dot product
        # h: N x out
        assert not torch.isnan(h).any()
        # print(self.a.data[0])
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # h[edge[0, :], :] shape = [N, outdim] # cat后 [N, outdim*2]
        # edge_h shape [outdim *2, N]
        # edge: 2*D x E

        edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))

        # for the weighted adj
        if self.w_adj:
            assert edge_e.size() == adj.values().size()
            edge_e = edge_e * adj.values()
            assert edge_e.size() == adj.values().size()

        assert not torch.isnan(edge_e).any()
        # edge_e: E
        ones = torch.ones(size=(N, 1), dtype=torch.float32)

        if self.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), ones)

        # e_rowsum: N x 1

        # edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        # add bias
        # h_prime = h_prime + self.bias

        if self.concat:
            # if this layer is not last layer,
            output =  F.elu(h_prime)
        else:
            # if this layer is last layer,
            output =  h_prime

        if self.residual:
            output = inputs + output
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(GraphConvolution, self).__init__()
        self.act_func = F.relu
        self.weights = nn.Parameter(torch.DoubleTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.DoubleTensor(output_dim))
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
    def __init__(self, num_sr, num_tg, embedding_dim, type='entity'):
        super(DoubleEmbedding, self).__init__()
        self.embedding_sr = nn.Embedding(num_sr, embedding_dim, _weight=torch.zeros((num_sr, embedding_dim), dtype=torch.float))
        self.embedding_tg = nn.Embedding(num_tg, embedding_dim, _weight=torch.zeros((num_tg, embedding_dim), dtype=torch.float))
        # nn.init.xavier_uniform_(self.embedding_sr.weight.data)
        # nn.init.xavier_uniform_(self.embedding_tg.weight.data)
        if type == 'entity':
            nn.init.normal_(self.embedding_sr.weight.data, std=1. / math.sqrt(num_sr))
            nn.init.normal_(self.embedding_tg.weight.data, std=1. / math.sqrt(num_tg))
        elif type=='relation':
            nn.init.xavier_uniform_(self.embedding_sr.weight.data)
            nn.init.xavier_uniform_(self.embedding_tg.weight.data)
        else:
            raise NotImplementedError

    def normalize(self):
        self.embedding_sr.weight.data = F.normalize(self.embedding_sr.weight, dim=-1, p=2)
        self.embedding_tg.weight.data = F.normalize(self.embedding_tg.weight, dim=-1, p=2)

    def forward(self, sr_data, tg_data):
        return self.embedding_sr(sr_data), self.embedding_tg(tg_data)

    @property
    def weight(self):
        return self.embedding_sr.weight, self.embedding_tg.weight


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
