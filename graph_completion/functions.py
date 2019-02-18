import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_completion.auction_lap import auction_lap
from scipy.optimize import linear_sum_assignment
from tools.timeit import timeit
from tools.print_time_info import print_time_info

class GCNAlignLoss(nn.Module):
    def __init__(self, margin, re_scale=1.0, cuda=True):
        super(GCNAlignLoss, self).__init__()
        self.re_scale = re_scale
        self.criterion = nn.MarginRankingLoss(margin)
        self.is_cuda = cuda

    def forward(self, repre_sr, repre_tg):
        '''
        repre_sr shape: [batch_size, 1 + nega_sample_num, embedding_dim]
        repre_tg shpae: [batch_size, 1 + nega_sample_num, embedding_dim]
        '''
        result_1 = torch.abs(repre_sr - repre_tg)
        result_2 = torch.sum(result_1, dim=-1)
        score = result_2 * self.re_scale
        pos_score = score[:, :1]
        nega_score = score[:, 1:]
        y = torch.FloatTensor([-1.0])
        # if next(self.parameters()).is_cuda:
        if self.is_cuda:
            y = y.cuda()
        loss = self.criterion(pos_score, nega_score, y)
        return loss


class RelationWeighting(nn.Module):
    def __init__(self, shape, cuda):
        super(RelationWeighting, self).__init__()
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        self.is_cuda = cuda
        self.shape = shape
        self.reverse = False
        if shape[0] > shape[1]:
            self.reverse = True
            self.shape = (shape[1], shape[0])
        self.pad_len = self.shape[1] - self.shape[0]

    def lap_solultion(self, sim):
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

    def max_pool_solution(self, sim):
        '''
        sim: shape = [num_relation_sr, num_relation_sr]
        '''
        dim = self.shape[1]
        sim = sim.expand(1, 1, dim, dim)
        sr_score = F.max_pool2d(sim, (1, dim)).view(-1)
        tg_score = F.max_pool2d(sim, (dim, 1)).view(-1)
        return sr_score, tg_score

    # @timeit
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
        r_sim_sr, r_sim_tg = self.max_pool_solution(sim)

        if pad_len > 0:
            r_sim_sr = r_sim_sr[:-pad_len]
        if reverse:
            r_sim_sr, r_sim_tg = r_sim_tg, r_sim_sr
        return r_sim_sr, r_sim_tg


def cosine_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.mm(a, b.transpose(0, 1))
