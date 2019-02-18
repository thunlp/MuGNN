import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class GCNAlignLoss(nn.Module):
    def __init__(self, margin, re_scale=1.0):
        super(GCNAlignLoss, self).__init__()
        self.re_scale = re_scale
        self.criterion = nn.MarginRankingLoss(margin)

    def forward(self, repre_sr, repre_tg):
        '''
        repre_sr shape: [batch_size, 1 + nega_sample_num, embedding_dim]
        repre_tg shpae: [batch_size, 1 + nega_sample_num, embedding_dim]
        '''
        print(repre_sr)
        print(repre_tg)
        result_1 = torch.abs(repre_sr - repre_tg)
        # print(result_1)
        result_2 = torch.sum(result_1, dim=-1)
        # print(result_2)
        score = result_2 * self.re_scale

        pos_score = score[:, :1]
        nega_score = score[:, 1:]
        print(pos_score)
        print(nega_score)
        y = torch.FloatTensor([-1.0])
        # if next(self.parameters()).is_cuda:
        # y = y.cuda()
        loss = self.criterion(pos_score, nega_score, y)
        print('losses:', loss)
        return loss


class RelationWeighting(nn.Module):
    def __init__(self, shape, cuda):
        super(RelationWeighting, self).__init__()
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        self.cuda = cuda
        self.shape = shape
        self.reverse = False
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


        rows, cols = linear_sum_assignment(-sim.detach().cpu().numpy())
        rows = torch.from_numpy(rows)
        cols = torch.from_numpy(cols)
        # if next(self.parameters()).is_cuda:
        if self.cuda:
            rows = rows.cuda()
            cols = cols.cuda()
        # print('rows1: ', rows.is_cuda)
        # print('cols: ', cols.is_cuda)
        r_sim_sr = torch.gather(sim, -1, cols.view(-1, 1)).squeeze(1)
        # print('r_sim_sr', r_sim_sr.is_cuda)
        cols, cols_index = cols.sort()
        # print('cols2:', cols.is_cuda)
        # print('cols_index:', cols_index.is_cuda)
        rows = rows[cols_index]
        # print('rows2', rows.is_cuda)
        r_sim_tg = torch.gather(sim.t(), -1, rows.view(-1, 1)).squeeze(1)
        # print('r_sim_tg', r_sim_tg.is_cuda)
        if pad_len > 0:
            r_sim_sr = r_sim_sr[:-pad_len]
            # print('r_sim_sr2', r_sim_sr.is_cuda)
        if reverse:
            r_sim_sr, r_sim_tg = r_sim_tg, r_sim_sr
            # print('r_sim_sr3', r_sim_sr.is_cuda)
            # print('r_sim_tg2', r_sim_tg.is_cuda)
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
