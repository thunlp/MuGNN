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


class SpecialLoss(nn.Module):
    def __init__(self, margin, p=2, re_scale=1.0, reduction='mean', cuda=True):
        super(SpecialLoss, self).__init__()
        self.p = p
        self.re_scale = re_scale
        self.criterion = nn.MarginRankingLoss(margin, reduction=reduction)
        self.is_cuda = cuda

    def forward(self, score):
        '''
        score shape: [batch_size, 2, embedding_dim]
        '''
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        # score shape = [num, 2, dim]
        distance = torch.pow(score, 0.5).sum(-1)
        pos_distance = distance[:, 0, :]
        neg_distance = distance[:, 1, :]
        y = torch.FloatTensor([-1, 0])
        if self.is_cuda:
            y = y.cuda()
        loss = self.criterion(pos_distance, neg_distance, y)
        return loss

class SpecialLossRule(nn.Module):
    def __init__(self, margin, re_scale=1.0, cuda=True):
        super(SpecialLossRule, self).__init__()
        self.p = 2
        self.re_scale = re_scale
        self.criterion = nn.MarginRankingLoss(margin)
        self.is_cuda = cuda

    def forward(self, score):
        '''
        score shape: [batch_size, 1 + nega_sample_num, embedding_dim]
        '''
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        pos_score = score[:, 0]
        nega_score = score[:, 1]
        y = torch.FloatTensor([1.0])
        if self.is_cuda:
            y = y.cuda()
        loss = self.criterion(pos_score, nega_score, y)
        loss = loss * self.re_scale
        return loss

class LimitBasedLoss(nn.Module):
    def __init__(self, gamma1=0.1, gamma2=0.8, micro=1.0):
        super(LimitBasedLoss, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.micro = micro

    def forward(self, score):
        # for my truth value transe
        score = 1 - score

        pos_score = score[:, 0]
        nega_score = score[:, 1]
        loss = torch.clamp(pos_score - self.gamma1, min=0) + self.micro * torch.clamp(self.gamma2 - nega_score, min=0)
        loss = loss.mean()
        return loss

class SpecialLossTransE(nn.Module):
    def __init__(self, margin, p=2, re_scale=1.0, reduction='mean', cuda=True):
        super(SpecialLossTransE, self).__init__()
        self.p = 2
        self.re_scale = re_scale
        self.criterion = nn.MarginRankingLoss(margin, reduction=reduction)
        self.is_cuda = cuda

    def forward(self, score):
        '''
        score shape: [batch_size, 1 + nega_sample_num, embedding_dim]
        '''
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        pos_score = score[:, :1] #.sum(-1, keepdim=True)
        nega_score = score[:, 1:] #.sum(-1, keepdim=True)
        y = torch.FloatTensor([-1.0])
        if self.is_cuda:
            y = y.cuda()
        loss = self.criterion(pos_score, nega_score, y)
        loss = loss * self.re_scale
        return loss


class SpecialLossAlign(nn.Module):
    def __init__(self, margin, p=2, re_scale=1.0, reduction='mean', cuda=True):
        super(SpecialLossAlign, self).__init__()
        self.p = p
        self.re_scale = re_scale
        self.criterion = nn.TripletMarginLoss(margin, p=p, reduction=reduction)
        self.is_cuda = cuda

    def forward(self, repre_sr, repre_tg):
        '''
        score shape: [batch_size, 2, embedding_dim]
        '''
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        sr_true = repre_sr[:, 0, :]
        sr_nega = repre_sr[:, 1, :]
        tg_true = repre_tg[:, 0, :]
        tg_nega = repre_tg[:, 1, :]
        loss = self.criterion(torch.cat((sr_true, tg_true), dim=0), torch.cat((tg_true, sr_true), dim=0),
                              torch.cat((tg_nega, sr_nega), dim=0))
        loss = loss * self.re_scale
        return loss


class RelationWeighting(object):
    def __init__(self, shape):
        # super(RelationWeighting, self).__init__()
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        self.shape = shape
        self.reverse = False
        if shape[0] > shape[1]:
            self.reverse = True
            self.shape = (shape[1], shape[0])
        self.pad_len = self.shape[1] - self.shape[0]

    def __call__(self, a, b):
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
        # sim = torch_l2distance(a, b)
        r_sim_sr, r_sim_tg = self._max_pool_solution(sim)
        if pad_len > 0:
            r_sim_sr = r_sim_sr[:-pad_len]
        if reverse:
            r_sim_sr, r_sim_tg = r_sim_tg, r_sim_sr
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
    a = a / torch.clamp(a.norm(dim=-1, keepdim=True), min=1e-8)
    b = b / torch.clamp(b.norm(dim=-1, keepdim=True), min=1e-8)
    return torch.mm(a, b.t())


def torch_l2distance(a, b):
    x1 = torch.sum(a * a, dim=-1).view(-1, 1)
    x2 = torch.sum(b * b, dim=-1).view(-1, 1)
    x3 = -2 * torch.matmul(a, b.t())
    sim = x1 + x2.t() + x3  # .pow(0.5)
    return sim.pow(0.5)
