import torch
import torch.nn as nn
import torch.nn.functional as F


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
