import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from graph_completion.cross_graph_completion import CrossGraphCompletion
from models.torch_functions import cosine_similarity_nbyn


class SpTwinAdj(object):
    def __init__(self, cgc, non_acylic, cuda=True):
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.is_cuda = cuda
        self.non_acylic = non_acylic
        self.entity_num_sr = len(cgc.id2entity_sr)
        self.entity_num_tg = len(cgc.id2entity_tg)
        self.triples_sr = cgc.triples_sr  # + list(cgc.new_triple_confs_sr.keys())
        self.triples_tg = cgc.triples_tg  # + list(cgc.new_triple_confs_tg.keys())
        self.init()

    def init(self):
        def _triple2sp_m(triples, size):
            heads, tails, relations = list(zip(*triples))
            pos = list(zip(heads, tails))
            if self.non_acylic:
                pos += list(zip(tails, heads))
            pos += [(i, i) for i in range(size)]  # unit matrix
            pos = set(pos)
            heads, tails = list(zip(*pos))
            pos = torch.tensor([heads, tails], dtype=torch.int64)
            value = torch.ones((len(heads),), dtype=torch.int64)
            return torch.sparse_coo_tensor(pos, value, size=torch.Size((size, size)))

        self.sp_adj_sr = _triple2sp_m(self.triples_sr,
                                      self.entity_num_sr).coalesce()  # .detach() #.to_dense()
        self.sp_adj_tg = _triple2sp_m(self.triples_tg,
                                      self.entity_num_tg).coalesce()  # .detach() #.to_dense()
        if self.is_cuda:
            self.sp_adj_sr = self.sp_adj_sr.cuda()
            self.sp_adj_tg = self.sp_adj_tg.cuda()

    def __call__(self, *args):
        return self.sp_adj_sr, self.sp_adj_tg


class SpRelWeiADJ(nn.Module):
    def __init__(self, cgc, non_acylic, cuda=True):
        super(SpRelWeiADJ, self).__init__()
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.is_cuda = cuda
        self.non_acylic = non_acylic
        self.entity_num_sr = len(cgc.id2entity_sr)
        self.entity_num_tg = len(cgc.id2entity_tg)
        self.triples_sr = cgc.triples_sr  # + list(cgc.new_triple_confs_sr.keys())
        self.triples_tg = cgc.triples_tg  # + list(cgc.new_triple_confs_tg.keys())
        self.reverse = False
        self.shape = (len(cgc.id2relation_sr), len(cgc.id2relation_tg))
        if self.shape[0] > self.shape[1]:
            self.reverse = True
            self.shape = (self.shape[1], self.shape[0])
        self.pad_len = self.shape[1] - self.shape[0]
        self.init()

    def init(self):
        def _triple2non_acylic(triples, size):
            pos2r = {(h, t): r for h, t, r in triples}
            for h, t, r in triples:
                pos2r[(t, h)] = r
            for i in range(size):
                if (i, i) in pos2r:
                    pos2r.pop((i, i))
            heads, tails = torch.from_numpy(np.asarray(list(zip(*list(pos2r.keys()))), dtype=np.int64))
            relations = torch.tensor(list(pos2r.values()), dtype=torch.int64)
            return heads, tails, relations

        head_sr, tail_sr, relation_sr = _triple2non_acylic(self.triples_sr, self.entity_num_sr)
        head_tg, tail_tg, relation_tg = _triple2non_acylic(self.triples_tg, self.entity_num_tg)
        self.relation_sr = relation_sr
        self.relation_tg = relation_tg
        self.pos_sr = torch.cat((head_sr.view(1, -1), tail_sr.view(1, -1)), dim=0)
        self.pos_tg = torch.cat((head_tg.view(1, -1), tail_tg.view(1, -1)), dim=0)
        self.unit_matrix_sr = get_sparse_unit_matrix(self.entity_num_sr)
        self.unit_matrix_tg = get_sparse_unit_matrix(self.entity_num_tg)
        if self.is_cuda:
            self.unit_matrix_sr = self.unit_matrix_sr.cuda()
            self.unit_matrix_tg = self.unit_matrix_tg.cuda()

    def forward(self, rel_embedding_sr, rel_embedding_tg):
        rel_att_sr, rel_att_tg = self._max_pool_attetion_solution(rel_embedding_sr, rel_embedding_tg)
        rel_att_sr = rel_att_sr[self.relation_sr]  # sparse support
        rel_att_tg = rel_att_tg[self.relation_tg]
        sp_rel_att_sr = torch.sparse_coo_tensor(self.pos_sr, rel_att_sr, size=(self.entity_num_sr, self.entity_num_sr))
        sp_rel_att_tg = torch.sparse_coo_tensor(self.pos_tg, rel_att_tg, size=(self.entity_num_tg, self.entity_num_tg))
        sp_rel_att_sr = sp_clamp(sp_rel_att_sr + self.unit_matrix_sr, max=1.0).coalesce()
        sp_rel_att_tg = sp_clamp(sp_rel_att_tg + self.unit_matrix_tg, max=1.0).coalesce()
        return sp_rel_att_sr, sp_rel_att_tg

    def _max_pool_attetion_solution(self, a, b):
        '''
        sim: shape = [num_relation_sr, num_relation_sr]
        '''
        pad_len = self.pad_len
        reverse = self.reverse
        if reverse:
            a, b = b, a
        if pad_len > 0:
            a = F.pad(a, (0, 0, 0, pad_len))
        sim = cosine_similarity_nbyn(a, b)
        dim = self.shape[1]
        sim = sim.expand(1, 1, dim, dim)
        a = F.max_pool2d(sim, (1, dim)).view(-1)
        b = F.max_pool2d(sim, (dim, 1)).view(-1)
        if pad_len > 0:
            a = a[:-pad_len]
        if reverse:
            a, b = b, a
        return a, b


def get_sparse_unit_matrix(size):
    values = torch.from_numpy(np.ones([size], dtype=np.float32))
    poses = torch.from_numpy(np.asarray([[i for i in range(size)] for _ in range(2)], dtype=np.int64))
    return torch_trans2sp(poses, values, (size, size))


def torch_trans2sp(indices, values, size):
    '''
    2019-2-26 safely create sparse tensor, max method is implemented for duplicates
    '''
    is_cuda = values.is_cuda
    assert indices.size()[1] == values.size()[0]
    indices = indices.cpu().numpy()
    values = values.cpu().numpy()
    poses = {}
    for i, indice in enumerate(zip(*indices)):
        if indice not in poses:
            poses[indice] = values[i]
        else:
            poses[indice] = max(values[i], poses[indice])
    new_indices = torch.tensor(list(zip(*poses.keys())))
    new_values = torch.tensor(list(poses.values()))
    if is_cuda:
        new_values = new_values.cuda()
        new_indices = new_indices.cuda()
    return torch.sparse.FloatTensor(new_indices, new_values, size=torch.Size(size))


def sp_clamp(sparse_tensor, min=None, max=None):
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    values = torch.clamp(values, min=min, max=max)
    return torch.sparse_coo_tensor(indices, values)


def watch_sp(sp, row_num):
    try:
        sp = sp.coo_matrix(sp)
        sp = sp.coalesce()
        row = sp.indices()[0]
        col = sp.indices()[1]
        values = sp.values()
        for i, ele in enumerate(row):
            if ele == row_num:
                print('(', row_num, int(col[i]), ')', float(values[i]))
    except:
        row = sp[row_num]
        for i, ele in enumerate(row):
            if ele != 0:
                print('(', row_num, i, ')', float(ele))
