import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def set_random_seed(seed_value=999):
    torch.manual_seed(seed_value)  # cpu  vars
    np.random.seed(seed_value)  # cpu vars
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

batch = 0
class SpecialLoss(nn.Module):
    def __init__(self, margin, re_scale=1.0, cuda=True):
        super(SpecialLoss, self).__init__()
        self.re_scale = re_scale
        self.criterion = nn.MarginRankingLoss(margin)
        self.is_cuda = cuda

    def forward(self, score):
        '''
        score shape: [batch_size, 1 + nega_sample_num, embedding_dim]
        '''
        distance = torch.abs(score).sum(dim=-1) * self.re_scale
        pos_score = distance[:, :1]
        nega_score = distance[:, 1:]
        y = torch.DoubleTensor([-1.0])
        if self.is_cuda:
            y = y.cuda()
        loss = self.criterion(pos_score, nega_score, y)
        return loss


class RelationWeighting(nn.Module):
    def __init__(self, shape, cuda, solution='max_pooling'):
        super(RelationWeighting, self).__init__()
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        self.is_cuda = cuda
        self.shape = shape
        self.reverse = False
        self.solution = solution
        if shape[0] > shape[1]:
            self.reverse = True
            self.shape = (shape[1], shape[0])
        self.pad_len = self.shape[1] - self.shape[0]

    def forward(self, a, b):
        '''
        a shape: [num_relation_a, embed_dim]
        b shape: [num_relation_b, embed_dim]
        return shape: [num_relation_a], [num_relation_b]
        '''
        pad_len = self.pad_len
        reverse = self.reverse
        if reverse:
            a, b = b, a
        if pad_len > 0:
            a = F.pad(a, (0, 0, 0, pad_len))
        sim = cosine_similarity_nbyn(a, b)
        if self.solution == 'max_pooling':
            r_sim_sr, r_sim_tg = self._max_pool_solution(sim)
        else:
            r_sim_sr, r_sim_tg = self._lap_solultion(sim)
        if pad_len > 0:
            r_sim_sr = r_sim_sr[:-pad_len]
        if reverse:
            r_sim_sr, r_sim_tg = r_sim_tg, r_sim_sr
        return r_sim_sr, r_sim_tg

    def _lap_solultion(self, sim):
        def scipy_solution(sim):
            rows, cols = linear_sum_assignment(-sim.detach().cpu().numpy())
            rows = torch.from_numpy(rows)
            cols = torch.from_numpy(cols)
            if self.is_cuda:
                rows = rows.cuda()
                cols = cols.cuda()
            return rows, cols

        rows, cols = scipy_solution(sim)
        r_sim_sr = torch.gather(sim, -1, cols.view(-1, 1)).squeeze(1)
        cols, cols_index = cols.sort()
        rows = rows[cols_index]
        r_sim_tg = torch.gather(sim.t(), -1, rows.view(-1, 1)).squeeze(1)
        return r_sim_sr, r_sim_tg

    def _max_pool_solution(self, sim):
        '''
        sim: shape = [num_relation_sr, num_relation_sr]
        '''
        dim = self.shape[1]
        sim = sim.expand(1, 1, dim, dim)
        sr_score = F.max_pool2d(sim, (1, dim)).view(-1)
        tg_score = F.max_pool2d(sim, (dim, 1)).view(-1)
        return sr_score, tg_score


def normalize_adj_torch(adj):
    adj = adj.to_dense()
    adj = torch.clamp(adj, max=1.0)
    row_sum = torch.sum(adj, 1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    result = ((adj * d_inv_sqrt).t() * d_inv_sqrt)  # the result of gcn code
    return result.t()  # .to_sparse()


def cosine_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.mm(a, b.transpose(0, 1))
