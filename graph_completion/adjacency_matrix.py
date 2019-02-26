import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from graph_completion.functions import str2int4triples
from graph_completion.torch_functions import RelationWeighting, normalize_adj_torch


def g_func_template(a, b, c, e):
    '''
    # sp_rel_conf_sr, sp_rel_imp_sr, sp_triple_pca_sr, sp_rel_att_sr
    # a * b * (0.5 * c + 0.5 * e)
    all input: sparse tensor shape = [num_entity, num_entity]
    :return: sparse tensor shape = [num_entity, num_entity]
    '''
    # return a * b * (1.0*c + 0.0*e)
    return a * b * (0.5 * c + 0.5 * e)


class SpTwinAdj(object):
    def __init__(self, cgc, non_acylic, cuda=True):
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.is_cuda = cuda
        self.non_acylic = non_acylic
        self.entity_num_sr = len(cgc.id2entity_sr)
        self.entity_num_tg = len(cgc.id2entity_tg)
        self.init()

    def init(self):
        cgc = self.cgc

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

        self.sp_adj_sr = _triple2sp_m(str2int4triples(cgc.triples_sr),
                                      self.entity_num_sr).coalesce()  # .detach() #.to_dense()
        self.sp_adj_tg = _triple2sp_m(str2int4triples(cgc.triples_tg),
                                      self.entity_num_tg).coalesce()  # .detach() #.to_dense()
        if self.is_cuda:
            self.sp_adj_sr = self.sp_adj_sr.cuda()
            self.sp_adj_tg = self.sp_adj_tg.cuda()

    def __call__(self, *args):
        return self.sp_adj_sr, self.sp_adj_tg


class SpTwinCAW(SpTwinAdj):
    def __init__(self, rule_scale, cgc, non_acylic, g_func=g_func_template, cuda=True):
        assert isinstance(cgc, CrossGraphCompletion)
        self.g_func = g_func
        self.rule_scale = rule_scale
        super(SpTwinCAW, self).__init__(cgc, non_acylic, cuda)

    def init(self):
        cgc = self.cgc
        self.relation_weighting = RelationWeighting((len(cgc.id2relation_sr), len(cgc.id2relation_tg)))
        head_sr, tail_sr, relation_sr = torch.from_numpy(np.asarray(
            list(zip(*str2int4triples(cgc.triples_sr))), dtype=np.int64))
        head_tg, tail_tg, relation_tg = torch.from_numpy(np.asarray(
            list(zip(*str2int4triples(cgc.triples_tg))), dtype=np.int64))
        self.relation_sr = relation_sr
        self.relation_tg = relation_tg
        self.pos_sr = torch.cat((head_sr.view(1, -1), tail_sr.view(1, -1)), dim=0)
        self.pos_tg = torch.cat((head_tg.view(1, -1), tail_tg.view(1, -1)), dim=0)
        self.unit_matrix_sr = get_sparse_unit_matrix(self.entity_num_sr)
        self.unit_matrix_tg = get_sparse_unit_matrix(self.entity_num_tg)
        self.sp_rel_conf_sr, self.sp_rel_imp_sr, self.sp_triple_pca_sr = build_adms_rconf_imp_pca(cgc.triples_sr,
                                                                                                  cgc.new_triple_confs_sr,
                                                                                                  self.entity_num_sr,
                                                                                                  cgc.relation2conf_sr,
                                                                                                  cgc.relation2imp_sr,
                                                                                                  self.rule_scale,
                                                                                                  self.non_acylic)
        self.sp_rel_conf_tg, self.sp_rel_imp_tg, self.sp_triple_pca_tg = build_adms_rconf_imp_pca(cgc.triples_tg,
                                                                                                  cgc.new_triple_confs_tg,
                                                                                                  self.entity_num_tg,
                                                                                                  cgc.relation2conf_tg,
                                                                                                  cgc.relation2imp_tg,
                                                                                                  self.rule_scale,
                                                                                                  self.non_acylic)
        if self.is_cuda:
            self.pos_sr = self.pos_sr.cuda()
            self.pos_tg = self.pos_tg.cuda()
            self.unit_matrix_sr = self.unit_matrix_sr.cuda()
            self.unit_matrix_tg = self.unit_matrix_tg.cuda()
            self.sp_rel_conf_sr, self.sp_rel_imp_sr, self.sp_triple_pca_sr = self.sp_rel_conf_sr.cuda(), self.sp_rel_imp_sr.cuda(), self.sp_triple_pca_sr.cuda()
            self.sp_rel_conf_tg, self.sp_rel_imp_tg, self.sp_triple_pca_tg = self.sp_rel_conf_tg.cuda(), self.sp_rel_imp_tg.cuda(), self.sp_triple_pca_tg.cuda()

    def __call__(self, rel_embedding_sr, rel_embedding_tg):
        sp_rel_att_sr, sp_rel_att_tg = self.__forward__relation_weight__(rel_embedding_sr, rel_embedding_tg)
        adjacency_matrix_sr = self.g_func(self.sp_rel_conf_sr, self.sp_rel_imp_sr, self.sp_triple_pca_sr, sp_rel_att_sr)
        adjacency_matrix_tg = self.g_func(self.sp_rel_conf_tg, self.sp_rel_imp_tg, self.sp_triple_pca_tg, sp_rel_att_tg)
        adjacency_matrix_sr = sp_clamp(adjacency_matrix_sr + self.unit_matrix_sr, max=1.0).coalesce()
        adjacency_matrix_tg = sp_clamp(adjacency_matrix_tg + self.unit_matrix_tg, max=1.0).coalesce()
        return adjacency_matrix_sr, adjacency_matrix_tg

    def __forward__relation_weight__(self, rel_embedding_sr, rel_embedding_tg):
        relation_w_sr, relation_w_tg = self.relation_weighting(rel_embedding_sr, rel_embedding_tg)
        rel_att_sr = relation_w_sr[self.relation_sr]  # sparse support
        rel_att_tg = relation_w_tg[self.relation_tg]
        sp_rel_att_sr = torch_trans2sp(self.pos_sr, rel_att_sr, [self.entity_num_sr] * 2)
        sp_rel_att_tg = torch_trans2sp(self.pos_tg, rel_att_tg, [self.entity_num_tg] * 2)
        return sp_rel_att_sr, sp_rel_att_tg


def get_sparse_unit_matrix(size):
    values = torch.from_numpy(np.ones([size], dtype=np.float32))
    poses = torch.from_numpy(np.asarray([[i for i in range(size)] for _ in range(2)], dtype=np.int64))
    return torch_trans2sp(poses, values, (size, size))


class CrossAdjacencyMatrix(nn.Module):
    def __init__(self, embedding_dim, cgc, cuda, non_acylic=False):
        super(CrossAdjacencyMatrix, self).__init__()
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.non_acylic = non_acylic
        self.embedding_dim = embedding_dim
        self.entity_num_sr = len(cgc.id2entity_sr)
        self.entity_num_tg = len(cgc.id2entity_tg)
        self.relation_weighting = RelationWeighting((len(cgc.id2relation_sr), len(cgc.id2relation_tg)), cuda)
        self.init_constant_part()

    def init_constant_part(self):
        cgc = self.cgc
        # for the transe
        head_sr, tail_sr, relation_sr = torch.from_numpy(np.asarray(
            list(zip(*str2int4triples(cgc.triples_sr))), dtype=np.int64))
        head_tg, tail_tg, relation_tg = torch.from_numpy(np.asarray(
            list(zip(*str2int4triples(cgc.triples_tg))), dtype=np.int64))
        self.relation_sr = nn.Parameter(relation_sr, requires_grad=False)
        self.relation_tg = nn.Parameter(relation_tg, requires_grad=False)
        self.pos_sr = nn.Parameter(torch.cat((head_sr.view(1, -1), tail_sr.view(1, -1)), dim=0), requires_grad=False)
        self.pos_tg = nn.Parameter(torch.cat((head_tg.view(1, -1), tail_tg.view(1, -1)), dim=0), requires_grad=False)

        # part of the matrix
        sp_rel_conf_sr, sp_rel_imp_sr, sp_triple_pca_sr = build_adms_rconf_imp_pca(cgc.triples_sr,
                                                                                   cgc.new_triple_confs_sr,
                                                                                   self.entity_num_sr,
                                                                                   cgc.relation2conf_sr,
                                                                                   cgc.relation2imp_sr,
                                                                                   self.non_acylic)

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

    def forward(self, rel_sr_weight, rel_tg_weight, g_func=g_func_template):
        relation_w_sr, relation_w_tg = self.relation_weighting(rel_sr_weight, rel_tg_weight)
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


def torch_trans2sp(indices, values, size):
    return torch.sparse.DoubleTensor(indices, values, size=torch.Size(size))


def build_adms_rconf_imp_pca(triples, new_triple_confs, num_entity, relation2conf, relation2imp, rule_scale=0.9, non_acylic=False):
    # sp_rel_conf_sr, sp_rel_imp_sr, sp_triple_pca_sr, sp_rel_att_sr
    # a * b * (0.5 * c + 0.5 * e)
    # print(num_entity)
    new_triple_confs = {triple: conf * rule_scale for triple, conf in new_triple_confs.items()}
    relation2conf = {relation: conf * rule_scale for relation, conf in relation2conf.items()}

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

def sp_clamp(sparse_tensor, min=None, max=None):
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    values = torch.clamp(values, min=min, max=max)
    return torch.sparse_coo_tensor(indices, values)