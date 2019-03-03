import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch import sparse


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
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, inputs, adj):
        h = torch.mm(inputs, self.W)
        N = h.size()[0]
        a_input = torch.cat((h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)), dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
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
        outputs = torch.mean(outputs, dim=-1)
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

    def forward(self, inputs, adj):
        N = inputs.size()[0]
        ones = torch.ones(size=(N, 1), dtype=torch.float32)
        if self.is_cuda:
            ones = ones.cuda()

        edge = adj.indices()
        h = torch.mul(inputs, self.W)  # todo: dot product

        # for relation weighting
        # h_prime2 = sparse.mm(adj, h)
        # adj_row_sum = torch.mm(adj, ones)
        # h_prime2 = h_prime2.div(adj_row_sum)

        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        # for relation weighting
        # edge_e = edge_e * adj.values()

        e_rowsum = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), ones)
        h_prime = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), h)

        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        # h_prime = (h_prime2 + h_prime) / 2
        if self.concat:
            output = F.elu(h_prime)
        else:
            output = h_prime

        if self.residual:
            output = inputs + output
            assert output.size() == inputs.size()
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RelAttGCN(nn.Module):
    def __init__(self, infeatures, out_features, cuda=True):
        super(RelAttGCN, self).__init__()
        assert infeatures == out_features
        self.is_cuda = cuda
        self.W = nn.Parameter(torch.zeros(size=(out_features,), dtype=torch.float))
        nn.init.ones_(self.W.data)

    def forward(self, inputs, adj):
        N = inputs.size()[0]
        ones = torch.ones(size=(N, 1), dtype=torch.float32)
        if self.is_cuda:
            ones = ones.cuda()
        adj_exp = torch.sparse_coo_tensor(adj.indices(), torch.exp(adj.values()), size=torch.Size((N, N)))
        inputs = torch.mul(inputs, self.W)  # todo: dot product
        # for relation weighting
        hidden = sparse.mm(adj_exp, inputs)
        # print('max', torch.max(adj_exp), 'min', torch.min(adj_exp))
        rowsum = sparse.mm(adj_exp, ones)
        # print('rowsum', rowsum.size())
        hidden = hidden.div(rowsum)
        # print('hidden: ', hidden.size())
        output = F.elu(hidden)
        return output


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
        self.embedding_sr = nn.Embedding(num_sr, embedding_dim,
                                         _weight=torch.zeros((num_sr, embedding_dim), dtype=torch.float))
        self.embedding_tg = nn.Embedding(num_tg, embedding_dim,
                                         _weight=torch.zeros((num_tg, embedding_dim), dtype=torch.float))
        # nn.init.xavier_uniform_(self.embedding_sr.weight.data)
        # nn.init.xavier_uniform_(self.embedding_tg.weight.data)
        if type == 'entity':
            nn.init.normal_(self.embedding_sr.weight.data, std=1. / math.sqrt(num_sr))
            nn.init.normal_(self.embedding_tg.weight.data, std=1. / math.sqrt(num_tg))
        elif type == 'relation':
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
