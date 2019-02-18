import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from tools.print_time_info import print_time_info
from tools.timeit import timeit
from pprint import pprint


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # format transform
    adj = sp.coo_matrix(adj)

    # norm
    rowsum = torch.sum(adj, 1)
    rowsum = torch.pow()

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def g_func_template(a, b, c, d, e):
    '''
    all input: sparse tensor shape = [num_entity, num_entity]
    :return: sparse tensor shape = [num_entity, num_entity]
    '''
    return a * b * (0.3*c + 0.3*d + 0.4*e)

def get_sparse_unit_matrix(size):
    values = torch.from_numpy(np.ones([size], dtype=np.float32))
    poses = torch.from_numpy(np.asarray([[i for i in range(size)] for _ in range(2)]))
    return torch_trans2sp(poses, values, (size, size))

class CrossAdjacencyMatrix(nn.Module):
    def __init__(self, embedding_dim, cgc):
        '''
        '''
        super(CrossAdjacencyMatrix, self).__init__()
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.embedding_dim = embedding_dim
        self.entity_num_sr = len(cgc.id2entity_sr)
        self.entity_num_tg = len(cgc.id2entity_tg)
        self.entity_embedding_sr = nn.Embedding(self.entity_num_sr, embedding_dim)
        self.entity_embedding_tg = nn.Embedding(self.entity_num_tg, embedding_dim)
        self.relation_embedding_sr = nn.Embedding(len(cgc.id2relation_sr), embedding_dim)
        self.relation_embedding_tg = nn.Embedding(len(cgc.id2relation_tg), embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding_sr.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding_tg.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_sr.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_tg.weight.data)

        self.init_constant_part()

    def init_constant_part(self):
        cgc = self.cgc
        # for the transe
        head_sr, tail_sr, relation_sr = torch.from_numpy(np.asarray(
            list(zip(*str2int4triples(cgc.triples_sr)))))
        head_tg, tail_tg, relation_tg = torch.from_numpy(np.asarray(
            list(zip(*str2int4triples(cgc.triples_tg)))))
        self.head_sr = nn.Parameter(head_sr, requires_grad=False)
        self.head_tg = nn.Parameter(head_tg, requires_grad=False)
        self.tail_sr = nn.Parameter(tail_sr, requires_grad=False)
        self.tail_tg = nn.Parameter(tail_tg, requires_grad=False)
        self.relation_sr = nn.Parameter(relation_sr, requires_grad=False)
        self.relation_tg = nn.Parameter(relation_tg, requires_grad=False)
        self.pos_sr = nn.Parameter(torch.cat([self.head_sr.view(1, -1), self.tail_sr.view(1, -1)], dim=0),
                                   requires_grad=False)
        self.pos_tg = nn.Parameter(torch.cat([self.head_tg.view(1, -1), self.tail_tg.view(1, -1)], dim=0),
                                   requires_grad=False)

        # part of the matrix
        sp_rel_conf_sr, sp_rel_imp_sr, sp_triple_pca_sr = build_adms_rconf_imp_pca(cgc.triples_sr,
                                                                                   cgc.new_triple_confs_sr,
                                                                                   self.entity_num_sr,
                                                                                   cgc.relation2conf_sr,
                                                                                   cgc.relation2imp_sr)
        # self.ad_constant_matrix_sr = nn.Parameter(torch.from_numpy(
        sp_rel_conf_tg, sp_rel_imp_tg, sp_triple_pca_tg = build_adms_rconf_imp_pca(cgc.triples_tg,
                                                                                   cgc.new_triple_confs_tg,
                                                                                   self.entity_num_tg,
                                                                                   cgc.relation2conf_tg,
                                                                                   cgc.relation2imp_tg)
        self.sp_rel_conf_sr = nn.Parameter(sp_rel_conf_sr, requires_grad=False)
        self.sp_rel_conf_tg = nn.Parameter(sp_rel_conf_tg, requires_grad=False)
        self.sp_rel_imp_sr = nn.Parameter(sp_rel_imp_sr, requires_grad=False)
        self.sp_rel_imp_tg = nn.Parameter(sp_rel_imp_tg, requires_grad=False)
        self.sp_triple_pca_sr = nn.Parameter(sp_triple_pca_sr, requires_grad=False)
        self.sp_triple_pca_tg = nn.Parameter(sp_triple_pca_tg, requires_grad=False)

        unit_matrix_sr = get_sparse_unit_matrix(self.entity_num_sr)
        unit_matrix_tg = get_sparse_unit_matrix(self.entity_num_tg)
        self.unit_matrix_sr = nn.Parameter(unit_matrix_sr, requires_grad=False)
        self.unit_matrix_tg = nn.Parameter(unit_matrix_tg, requires_grad=False)

    def forward(self, g_func=g_func_template):
        relation_w_sr, relation_w_tg = relation_weighting(
            self.relation_embedding_sr.weight, self.relation_embedding_tg.weight)
        sp_rel_att_sr, sp_rel_att_tg = self._forward_relation(relation_w_sr, relation_w_tg)
        sp_tv_sr, sp_tv_tg = self._forward_transe_tv()
        adjacency_matrix_sr = g_func(self.sp_rel_conf_sr, self.sp_rel_imp_sr, self.sp_triple_pca_sr, sp_tv_sr,
                                     sp_rel_att_sr)
        adjacency_matrix_tg = g_func(self.sp_rel_conf_tg, self.sp_rel_imp_tg, self.sp_triple_pca_tg, sp_tv_tg,
                                     sp_rel_att_tg)

        return adjacency_matrix_sr + self.unit_matrix_sr, adjacency_matrix_tg + self.unit_matrix_tg

    def _forward_relation(self, relation_w_sr, relation_w_tg):
        rel_att_sr = F.embedding(self.relation_sr, relation_w_sr)  # sparse support
        rel_att_tg = F.embedding(self.relation_tg, relation_w_tg)
        sp_rel_att_sr = torch_trans2sp(self.pos_sr, rel_att_sr, [self.entity_num_sr] * 2)
        sp_rel_att_tg = torch_trans2sp(self.pos_tg, rel_att_tg, [self.entity_num_tg] * 2)
        return sp_rel_att_sr, sp_rel_att_tg

    def _forward_transe_tv(self):
        def _score_func(h, t, r):
            return 1 - torch.norm(h + r - t, dim=1) / 3 / math.sqrt(self.embedding_dim)

        h_sr = self.entity_embedding_sr(self.head_sr)
        h_tg = self.entity_embedding_tg(self.head_tg)
        t_sr = self.entity_embedding_sr(self.tail_sr)
        t_tg = self.entity_embedding_tg(self.tail_tg)
        r_sr = self.relation_embedding_sr(self.relation_sr)
        r_tg = self.relation_embedding_tg(self.relation_tg)
        score_sr = _score_func(h_sr, t_sr, r_sr)
        score_tg = _score_func(h_tg, t_tg, r_tg)
        sp_score_sr = torch_trans2sp(self.pos_sr, score_sr, [self.entity_num_sr] * 2)  # .todense()
        sp_score_tg = torch_trans2sp(self.pos_tg, score_tg, [self.entity_num_tg] * 2)  # .todense()
        return sp_score_sr, sp_score_tg


def str2int4triples(triples):
    return [(int(head), int(tail), int(relation)) for head, tail, relation in triples]


def torch_trans2sp(indices, values, size):
    return torch.sparse.FloatTensor(indices, values, size=torch.Size(size))


def build_adms_rconf_imp_pca(triples, new_triple_confs, num_entity, relation2conf, relation2imp):
    # the last dimension: (rel_conf, rel_imp, triple_pca)
    # print(num_entity)
    sp_matrix = {0: {}, 1: {}, 2: {}}
    for triple in triples:
        head, tail, relation = triple
        pos = (int(head), int(tail))
        sp_matrix[1][pos] = relation2imp[relation]
        if triple in new_triple_confs:
            if pos in sp_matrix[0]:
                sp_matrix[0][pos] = max(relation2conf[relation], sp_matrix[0][pos])
            else:
                sp_matrix[0][pos] = relation2conf[relation]
            if pos in sp_matrix[2]:
                sp_matrix[2][pos] = max(new_triple_confs[triple], sp_matrix[2][pos])
            else:
                sp_matrix[2][pos] = new_triple_confs[triple]
        else:
            sp_matrix[0][pos] = 1
            sp_matrix[2][pos] = 1
    for key, sp_m in sp_matrix.items():
        poses = torch.from_numpy(np.asarray(list(zip(*sp_m.keys())), dtype=np.int))
        values = torch.from_numpy(np.asarray(list(sp_m.values()), dtype=np.float32))
        assert len(values) == len(poses[0]) == len(poses[-1])
        sp_matrix[key] = torch_trans2sp(poses, values, [num_entity, num_entity])
    # print_time_info('The duplicate triple num: %d/%d.'%(i, len(triples)))
    return sp_matrix[0], sp_matrix[1], sp_matrix[2]


# att(r, r'): relation, shape = [num_relation,]
def cosine_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.mm(a, b.transpose(0, 1))


@timeit
def relation_weighting(a, b):
    '''
    a shape: [num_relation_a, embed_dim]
    b shape: [num_relation_b, embed_dim]
    return shape: [num_relation_a], [num_relation_b]
    '''
    reverse = False
    if a.size()[0] > b.size()[0]:
        a, b = b, a
        reverse = True
    pad_len = b.size()[0] - a.size()[0]
    if pad_len > 0:
        a = F.pad(a, (0, 0, 0, pad_len))
    sim = cosine_similarity_nbyn(a, b)
    rows, cols = linear_sum_assignment(-sim.detach().cpu().numpy())
    rows = torch.from_numpy(rows)
    cols = torch.from_numpy(cols)
    if sim.is_cuda:
        rows = rows.cuda(sim.get_device())
        cols = cols.cuda(sim.get_device())
    r_sim_sr = torch.gather(sim, -1, cols.view(-1, 1)).squeeze(1)
    cols, cols_index = cols.sort()
    rows = rows[cols_index]
    r_sim_tg = torch.gather(sim.t(), -1, rows.view(-1, 1)).squeeze(1)

    if pad_len > 0:
        r_sim_sr = r_sim_sr[:-pad_len]
    if reverse:
        r_sim_sr, r_sim_tg = r_sim_tg, r_sim_sr
    return r_sim_sr, r_sim_tg
