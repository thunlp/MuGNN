import torch
import torch.nn as nn


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
        pos_score = score[:, :1]  # .sum(-1, keepdim=True)
        nega_score = score[:, 1:]  # .sum(-1, keepdim=True)
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
