import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_completion.GraphConvolution import GraphConvolution
from graph_completion.CrossAdjacencyMatrix import CrossAdjacencyMatrix
from graph_completion.CrossGraphCompletion import CrossGraphCompletion


class GCN(nn.Module):
    def __init__(self, cuda, cgc, num_layer, embedding_dim, dropout_rate=0.5, act_func=F.relu, bias=False):
        super(GCN, self).__init__()
        assert isinstance(cgc, CrossGraphCompletion)
        self.cam = CrossAdjacencyMatrix(embedding_dim, cgc, cuda)
        self.gcn_list = nn.ModuleList(
            [GraphConvolution(embedding_dim, embedding_dim, dropout_rate, act_func, bias) for _ in range(num_layer)])

    def forward(self, sr_data, tg_data, sr_rel_data, tg_rel_data):
        adjacency_matrix_sr, adjacency_matrix_tg = self.cam()
        rel_embedding_sr, rel_embedding_tg = self.cam.relation_embedding_sr, self.cam.relation_embedding_tg
        graph_embedding_sr, graph_embedding_tg = self.cam.entity_embedding_sr.weight, self.cam.entity_embedding_tg.weight

        for gcn in self.gcn_list:
            graph_embedding_sr = gcn(graph_embedding_sr, adjacency_matrix_sr)
            graph_embedding_tg = gcn(graph_embedding_tg, adjacency_matrix_tg)

        repre_e_sr = F.embedding(sr_data, graph_embedding_sr)
        repre_e_tg = F.embedding(tg_data, graph_embedding_tg)
        repre_r_sr = rel_embedding_sr(sr_rel_data)
        repre_r_tg = rel_embedding_tg(tg_rel_data)
        return repre_e_sr, repre_e_tg, repre_r_sr, repre_r_tg

    def loss(self, pos_score, nega_score):
        y = torch.cuda.tensor([-1])
        return self.criterion(pos_score, nega_score, y)

    def _calc(self, repre_sr, repre_tg):
        '''
        repre shape: [batch_size, 1+nega_sample_num, embedding_dim]
        '''
        score = torch.sum(torch.abs(repre_sr - repre_tg), dim=-1)
        pos_score = score[:, :1]
        nega_score = score[:, 1:]
        return pos_score, nega_score