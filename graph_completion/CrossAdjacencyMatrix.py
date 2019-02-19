import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from graph_completion.functions import RelationWeighting, cosine_similarity_nbyn, normalize_adj_torch


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

def g_func_template(a, b, c, e):
    '''
    all input: sparse tensor shape = [num_entity, num_entity]
    :return: sparse tensor shape = [num_entity, num_entity]
    '''
    # return a * b * (1.0*c + 0.0*e)
    return a * b * c

def get_sparse_unit_matrix(size):
    values = torch.from_numpy(np.ones([size], dtype=np.float32))
    poses = torch.from_numpy(np.asarray([[i for i in range(size)] for _ in range(2)], dtype=np.int64))
    return torch_trans2sp(poses, values, (size, size))

class CrossAdjacencyMatrix(nn.Module):
    def __init__(self, embedding_dim, cgc, cuda, non_acylic=False):
        '''
        '''
        super(CrossAdjacencyMatrix, self).__init__()
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.non_acylic = non_acylic
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

        self.relation_weighting = RelationWeighting((len(cgc.id2relation_sr), len(cgc.id2relation_tg)), cuda)
        self.init_constant_part()

    def init_constant_part(self):
        cgc = self.cgc
        # for the transe
        head_sr, tail_sr, relation_sr = torch.from_numpy(np.asarray(
            list(zip(*str2int4triples(cgc.triples_sr))), dtype=np.int64))
        head_tg, tail_tg, relation_tg = torch.from_numpy(np.asarray(
            list(zip(*str2int4triples(cgc.triples_tg))), dtype=np.int64))
        self.head_sr = nn.Parameter(head_sr, requires_grad=False)
        self.head_tg = nn.Parameter(head_tg, requires_grad=False)
        self.tail_sr = nn.Parameter(tail_sr, requires_grad=False)
        self.tail_tg = nn.Parameter(tail_tg, requires_grad=False)
        self.relation_sr = nn.Parameter(relation_sr, requires_grad=False)
        self.relation_tg = nn.Parameter(relation_tg, requires_grad=False)
        self.pos_sr = nn.Parameter(torch.cat([self.head_sr.view(1, -1), self.tail_sr.view(1, -1)], dim=0), requires_grad=False)
        self.pos_tg = nn.Parameter(torch.cat([self.head_tg.view(1, -1), self.tail_tg.view(1, -1)], dim=0), requires_grad=False)

        # part of the matrix
        sp_rel_conf_sr, sp_rel_imp_sr, sp_triple_pca_sr = build_adms_rconf_imp_pca(cgc.triples_sr,
                                                                                   cgc.new_triple_confs_sr,
                                                                                   self.entity_num_sr,
                                                                                   cgc.relation2conf_sr,
                                                                                   cgc.relation2imp_sr,
                                                                                   self.non_acylic)
        # self.ad_constant_matrix_sr = nn.Parameter(torch.from_numpy(
        sp_rel_conf_tg, sp_rel_imp_tg, sp_triple_pca_tg = build_adms_rconf_imp_pca(cgc.triples_tg,
                                                                                   cgc.new_triple_confs_tg,
                                                                                   self.entity_num_tg,
                                                                                   cgc.relation2conf_tg,
                                                                                   cgc.relation2imp_tg,
                                                                                   self.non_acylic)
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
        relation_w_sr, relation_w_tg = self.relation_weighting(
            self.relation_embedding_sr.weight, self.relation_embedding_tg.weight)
        # print(relation_w_sr)
        # return
        sp_rel_att_sr, sp_rel_att_tg = self._forward_relation(relation_w_sr, relation_w_tg)
        # sp_tv_sr, sp_tv_tg = self._forward_transe_tv()
        adjacency_matrix_sr = g_func(self.sp_rel_conf_sr, self.sp_rel_imp_sr, self.sp_triple_pca_sr, sp_rel_att_sr)
        adjacency_matrix_tg = g_func(self.sp_rel_conf_tg, self.sp_rel_imp_tg, self.sp_triple_pca_tg, sp_rel_att_tg)
        # watch_sp((adjacency_matrix_sr + self.unit_matrix_sr).cpu().detach().to_dense().numpy(), 0)
        adjacency_matrix_sr = normalize_adj_torch(adjacency_matrix_sr + self.unit_matrix_sr)
        # watch_sp(adjacency_matrix_sr.cpu().detach().numpy(), 0)
        adjacency_matrix_tg = normalize_adj_torch(adjacency_matrix_tg + self.unit_matrix_tg)
        return adjacency_matrix_sr, adjacency_matrix_tg

    def _forward_relation(self, relation_w_sr, relation_w_tg):
        rel_att_sr = F.embedding(self.relation_sr, relation_w_sr)  # sparse support
        rel_att_tg = F.embedding(self.relation_tg, relation_w_tg)
        sp_rel_att_sr = torch_trans2sp(self.pos_sr, rel_att_sr, [self.entity_num_sr] * 2)
        sp_rel_att_tg = torch_trans2sp(self.pos_tg, rel_att_tg, [self.entity_num_tg] * 2)
        return sp_rel_att_sr, sp_rel_att_tg

    def _forward_transe_tv(self):
        # 这里是optimize到不了的
        def _score_func(h, t, r):
            return 1 - torch.norm(h + r - t, p=1, dim=-1) / 3 / math.sqrt(self.embedding_dim)

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
        # print('--------', sp_score_sr.requires_grad)
        # sp_score_sr.register_hook(lambda x: print('---------' + str(torch.isnan(x).sum())))
        return sp_score_sr, sp_score_tg


def str2int4triples(triples):
    return [(int(head), int(tail), int(relation)) for head, tail, relation in triples]


def torch_trans2sp(indices, values, size):
    return torch.sparse.FloatTensor(indices, values, size=torch.Size(size))


def build_adms_rconf_imp_pca(triples, new_triple_confs, num_entity, relation2conf, relation2imp, non_acylic=False):
    # the last dimension: (rel_conf, rel_imp, triple_pca)
    # print(num_entity)
    sp_matrix = {0: {}, 1: {}, 2: {}}
    for triple in triples:
        head, tail, relation = triple
        pos = (int(head), int(tail))
        reverse_pos = (int(tail), int(head))
        sp_matrix[1][pos] = relation2imp[relation]
        if non_acylic:
            sp_matrix[1][reverse_pos] = relation2imp[relation]
        if triple in new_triple_confs:
            if pos in sp_matrix[0]:
                sp_matrix[0][pos] = max(relation2conf[relation], sp_matrix[0][pos])
                if non_acylic:
                    sp_matrix[0][reverse_pos] = max(relation2conf[relation], sp_matrix[0][reverse_pos])
            else:
                sp_matrix[0][pos] = relation2conf[relation]
                if non_acylic:
                    sp_matrix[0][reverse_pos] = relation2conf[relation]
            if pos in sp_matrix[2]:
                sp_matrix[2][pos] = max(new_triple_confs[triple], sp_matrix[2][pos])
                if non_acylic:
                    sp_matrix[2][reverse_pos] = max(new_triple_confs[triple], sp_matrix[2][reverse_pos])
            else:
                sp_matrix[2][pos] = new_triple_confs[triple]
                if non_acylic:
                    sp_matrix[2][reverse_pos] = new_triple_confs[triple]
        else:
            sp_matrix[0][pos] = 1
            sp_matrix[2][pos] = 1
            if non_acylic:
                sp_matrix[0][reverse_pos] = 1
                sp_matrix[2][reverse_pos] = 1

    for key, sp_m in sp_matrix.items():
        poses = torch.from_numpy(np.asarray(list(zip(*sp_m.keys())), dtype=np.int64))
        values = torch.from_numpy(np.asarray(list(sp_m.values()), dtype=np.float32))
        assert len(values) == len(poses[0]) == len(poses[-1])
        sp_matrix[key] = torch_trans2sp(poses, values, [num_entity, num_entity])
    # print_time_info('The duplicate triple num: %d/%d.'%(i, len(triples)))
    return sp_matrix[0], sp_matrix[1], sp_matrix[2]