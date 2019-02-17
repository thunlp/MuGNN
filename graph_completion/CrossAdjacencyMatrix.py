import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from tools.print_time_info import print_time_info
from pprint import pprint


def build_adms_rconf_imp_pca(triples, new_triple_confs, num_entity, relation2conf, relation2imp):
    # the last dimension: (relation_conf, rel_imp, pca_conf)
    # print(num_entity)
    matrix = np.zeros([num_entity, num_entity, 3])
    for triple in triples:
        head, tail, relation = triple
        head, tail = int(head), int(tail)
        matrix[head, tail][1] = relation2imp[relation]
        if triple in new_triple_confs:
            matrix[head, tail][0] = max(relation2conf[relation], matrix[head, tail][0])
            matrix[head, tail][2] = max(new_triple_confs[triple], matrix[head, tail][2])
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
    r_sim_tg = torch.gather(sim.transpose(0, 1), -1, rows.view(-1, 1)).squeeze(1)
    
    if pad_len > 0:
        r_sim_sr = r_sim_sr[:-pad_len]
    if reverse:
        r_sim_sr, r_sim_tg = r_sim_tg, r_sim_sr
    return r_sim_sr, r_sim_tg


class CrossAdjacencyMatrix(nn.Module):
    def __init__(self, embedding_dim, cgc):
        '''
        '''
        super(CrossAdjacencyMatrix, self).__init__()
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.entity_embedding_sr = nn.Embedding(len(cgc.id2entity_sr), embedding_dim)
        self.relation_embedding_sr = nn.Embedding(len(cgc.id2relation_sr), embedding_dim)
        self.entity_embedding_tg = nn.Embedding(len(cgc.id2entity_tg), embedding_dim)
        self.relation_embedding_tg = nn.Embedding(len(cgc.id2relation_tg), embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding_sr.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_sr.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding_tg.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_tg.weight.data)

    def builf_cross_adjacency_matrix(self):
        cgc = self.cgc
        ad_constant_matrix_sr = build_adms_rconf_imp_pca(cgc.triples_sr, cgc.new_triple_confs_sr, len(cgc.id2entity_sr), cgc.relation2conf_sr, cgc.relation2imp_sr)
        ad_constant_matrix_tg = build_adms_rconf_imp_pca(cgc.triples_tg, cgc.new_triple_confs_tg, len(cgc.id2entity_tg), cgc.relation2conf_tg, cgc.relation2imp_tg)
        relation_w_sr, relation_w_tg = relation_weighting(self.relation_embedding_sr, self.relation_embedding_tg)


    def forward(self, triple2conf_sr, triple2conf_tg, relation2idf_sr, relation2idf_tg):
        pass


if __name__ == '__main__':
    pass
