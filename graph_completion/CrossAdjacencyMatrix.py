import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from tools.print_time_info import print_time_info
from pprint import pprint


def build_adms_rconf_imp_pca(triples, new_triple_confs, num_entity, relation2conf, relation2imp):
    # the last dimension: (relation_conf, rel_imp, pca_conf)
    # print(num_entity)
    matrix = np.zeros([num_entity, num_entity, 3], dtype=torch.float)
    for triple in triples:
        head, tail, relation = triple
        head, tail = int(head), int(tail)
        matrix[head, tail][1] = relation2imp[relation]
        if triple in new_triple_confs:
            matrix[head, tail][0] = max(
                relation2conf[relation], matrix[head, tail][0])
            matrix[head, tail][2] = max(
                new_triple_confs[triple], matrix[head, tail][2])
        else:
            matrix[head, tail][0] = 1
            matrix[head, tail][2] = 1
    # print_time_info('The duplicate triple num: %d/%d.'%(i, len(triples)))
    return matrix


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
    rows, cols = linear_sum_assignment(-sim.numpy())
    rows = torch.from_numpy(rows)
    cols = torch.from_numpy(cols)
    r_sim_sr = torch.gather(sim, -1, cols.view(-1, 1)).squeeze(1)
    cols, cols_index = cols.sort()
    rows = rows[cols_index]
    r_sim_tg = torch.gather(sim.t, -1, rows.view(-1, 1)).squeeze(1)

    if pad_len > 0:
        r_sim_sr = r_sim_sr[:-pad_len]
    if reverse:
        r_sim_sr, r_sim_tg = r_sim_tg, r_sim_sr
    return r_sim_sr, r_sim_tg


def build_transe_tv_matrix(triples, num_entity):
    pass


class CrossAdjacencyMatrix(nn.Module):
    def __init__(self, embedding_dim, cgc):
        '''
        '''
        super(CrossAdjacencyMatrix, self).__init__()
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.embedding_dim = embedding_dim
        self.entity_embedding_sr = nn.Embedding(
            len(cgc.id2entity_sr), embedding_dim)
        self.entity_embedding_tg = nn.Embedding(
            len(cgc.id2entity_tg), embedding_dim)
        self.relation_embedding_sr = nn.Embedding(
            len(cgc.id2relation_sr), embedding_dim)
        self.relation_embedding_tg = nn.Embedding(
            len(cgc.id2relation_tg), embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding_sr.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding_tg.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_sr.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_tg.weight.data)

    def init_constant_part(self):
        cgc = self.cgc

        # for the transe
        self.head_sr, self.tail_sr, self.relation_sr = np.asarray(
            list(zip(*cgc.triples_sr)))
        self.head_tg, self.tail_tg, self.relation_tg = np.asarray(
            list(zip(*cgc.triples_tg)))

        # part of the matrix
        self.ad_constant_matrix_sr = build_adms_rconf_imp_pca(cgc.triples_sr, cgc.new_triple_confs_sr, len(
            cgc.id2entity_sr), cgc.relation2conf_sr, cgc.relation2imp_sr)
        self.ad_constant_matrix_tg = build_adms_rconf_imp_pca(cgc.triples_tg, cgc.new_triple_confs_tg, len(
            cgc.id2entity_tg), cgc.relation2conf_tg, cgc.relation2imp_tg)

        # torch
        self.head_sr = torch.from_numpy(self.head_sr)
        self.head_tg = torch.from_numpy(self.head_tg)
        self.tail_sr = torch.from_numpy(self.tail_sr)
        self.tail_tg = torch.from_numpy(self.tail_tg)
        self.relation_sr = torch.from_numpy(self.relation_sr)
        self.relation_tg = torch.from_numpy(self.relation_tg)
        self.ad_constant_matrix_sr = torch.from_numpy(
            self.ad_constant_matrix_sr)
        self.ad_constant_matrix_tg = torch.from_numpy(
            self.ad_constant_matrix_tg)

    def forward(self):
        relation_w_sr, relation_w_tg = relation_weighting(
            self.relation_embedding_sr, self.relation_embedding_tg)
        tv_sr, tv_tg = self._forward_transe_tv()


    def _forward_transe_tv(self):
        cgc = self.cgc

        def score_func(h, t, r):
            score = 1 - torch.norm(h + r - t, dim=-1) / \
                3 / math.sqrt(self.embedding_dim)
            coordinates = torch.cat([h, t], dim=0)
            return score, coordinates
        h_sr = self.entity_embedding_sr(self.head_sr)
        h_tg = self.entity_embedding_sr(self.head_tg)
        t_sr = self.entity_embedding_sr(self.tail_sr)
        t_tg = self.entity_embedding_sr(self.tail_tg)
        r_sr = self.relation_embedding_sr(self.relation_sr)
        r_tg = self.relation_embedding_sr(self.relation_tg)
        score_sr, coordinates_sr = score_func(h_sr, t_sr, r_sr)
        score_tg, coordinates_tg = score_func(h_tg, t_tg, r_tg)
        score_m_sr = torch.sparse.FloatTensor(coordinates_sr, score_sr, torch.Size([
                                              len(cgc.id2entity_sr)] * 2)).todense()
        score_m_tg = torch.sparse.FloatTensor(coordinates_tg, score_tg, torch.Size([
                                              len(cgc.id2entity_tg)] * 2)).todense()
        return score_m_sr, score_m_tg


if __name__ == '__main__':
    pass
