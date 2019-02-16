import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

def builf_cross_adjacency_matrix(cgc):
    assert isinstance(cgc, CrossGraphCompletion)
    ad_constant_matrix_sr = build_adms_rconf_imp_pca(cgc.triples_sr, cgc.new_triple_confs_sr, len(cgc.id2entity_sr), cgc.relation2conf_sr, cgc.relation2imp_sr)
    ad_constant_matrix_tg = build_adms_rconf_imp_pca(cgc.triples_tg, cgc.new_triple_confs_tg, len(cgc.id2entity_tg), cgc.relation2conf_tg, cgc.relation2imp_tg)
    

    


class CrossAdjacencyMatrix(nn.Module):
    def __init__(self, entity_num_sr, relation_num_sr, entity_num_tg, relation_num_tg, embedding_dim):
        '''
        '''
        super(CrossAdjacencyMatrix, self).__init__()
        self.entity_embedding_sr = nn.Embedding(entity_num_sr, embedding_dim)
        self.relation_embedding_sr = nn.Embedding(relation_num_sr, embedding_dim)
        self.entity_embedding_tg = nn.Embedding(entity_num_tg, embedding_dim)
        self.relation_embedding_tg = nn.Embedding(relation_num_tg, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding_sr)
        nn.init.xavier_uniform_(self.relation_embedding_sr)
        nn.init.xavier_uniform_(self.entity_embedding_tg)
        nn.init.xavier_uniform_(self.relation_embedding_tg)

    def forward(self, triple2conf_sr, triple2conf_tg, relation2idf_sr, relation2idf_tg):
        pass


if __name__ == '__main__':
    pass
