import torch
import torch.nn as nn
import torch.nn.functional as F
from .functions import str2int4triples
from graph_completion.models import GCN, GAT
from graph_completion.layers import DoubleEmbedding
from graph_completion.adjacency_matrix import CrossAdjacencyMatrix, SpTwinAdj
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from graph_completion.torch_functions import cosine_similarity_nbyn, torch_l2distance
from scipy.optimize import linear_sum_assignment

__all__ = ['GATNet']


class AlignGraphNet(nn.Module):
    def __init__(self, dropout_rate=0.5, non_acylic=False, cuda=True):
        super(AlignGraphNet, self).__init__()
        self.is_cuda = cuda
        self.non_acylic = non_acylic
        self.dropout_rate = dropout_rate

    def predict(self, sr_data, tg_data):
        raise NotImplementedError

    def bootstrap(self, *args):
        raise NotImplementedError


class GATNet(AlignGraphNet):
    def __init__(self, cgc, num_layer, dim, nheads, sp, alpha=0.2, *args, **kwargs):
        super(GATNet, self).__init__(*args, **kwargs)
        assert isinstance(cgc, CrossGraphCompletion)
        num_entity_sr = len(cgc.id2entity_sr)
        num_entity_tg = len(cgc.id2entity_tg)
        sp_twin_adj = SpTwinAdj(num_entity_sr, num_entity_tg, str2int4triples(cgc.triples_sr),
                                str2int4triples(cgc.triples_tg), self.non_acylic)
        self.adj_sr = sp_twin_adj.sp_adj_sr
        self.adj_tg = sp_twin_adj.sp_adj_tg
        self.sp_gat = GAT(dim, dim, nheads, num_layer, self.dropout_rate, alpha, sp, self.is_cuda)
        self.entity_embedding = DoubleEmbedding(num_entity_sr, num_entity_tg, dim)
        self.relation_embedding = DoubleEmbedding(len(cgc.id2relation_sr), len(cgc.id2entity_tg), dim)
        if self.is_cuda:
            self.adj_sr = self.adj_sr.cuda()
            self.adj_tg = self.adj_tg.cuda()

    def trans_e(self, ent_embedding, rel_embedding, h_list, t_list, r_list):
        h = ent_embedding[h_list]
        t = ent_embedding[t_list]
        r = rel_embedding[r_list]
        # shape [num, 2*nega + 1, dim]
        return h + r - t

    def entity_align(self, sr_embedding, tg_embedding):
        return sr_embedding - tg_embedding

    def forward(self, sr_data, tg_data, h_list_sr, h_list_tg, t_list_sr, t_list_tg, r_list_sr, r_list_tg):
        graph_embedding_sr, graph_embedding_tg = self.entity_embedding.weight
        rel_embedding_sr, rel_embedding_tg = self.relation_embedding.weight
        output_sr = self.sp_gat(graph_embedding_sr, self.adj_sr)
        output_tg = self.sp_gat(graph_embedding_tg, self.adj_tg)
        align_score = self.entity_align(output_sr[sr_data], output_tg[tg_data])
        sr_transe_score = self.trans_e(output_sr, rel_embedding_sr, h_list_sr, t_list_sr, r_list_sr)
        tg_transe_score = self.trans_e(output_tg, rel_embedding_tg, h_list_tg, t_list_tg, r_list_tg)
        return align_score, sr_transe_score, tg_transe_score

    def predict(self, sr_data, tg_data):
        graph_embedding_sr, graph_embedding_tg = self.entity_embedding.weight
        graph_embedding_sr = self.sp_gat(graph_embedding_sr, self.adj_sr)
        graph_embedding_tg = self.sp_gat(graph_embedding_tg, self.adj_tg)
        repre_e_sr, repre_e_tg = graph_embedding_sr[sr_data], graph_embedding_tg[tg_data]
        sim = torch_l2distance(repre_e_sr.detach(), repre_e_tg.detach()).cpu().numpy()
        return sim


    def bootstrap(self, entity_seeds, relation_seeds):
        adjacency_matrix_sr, adjacency_matrix_tg = self.cam()
        graph_embedding_sr, graph_embedding_tg = self.cam.entity_embedding_sr.weight, self.cam.entity_embedding_tg.weight
        for gcn in self.gcn_list:
            graph_embedding_sr = gcn(graph_embedding_sr, adjacency_matrix_sr)
            graph_embedding_tg = gcn(graph_embedding_tg, adjacency_matrix_tg)
        rel_embedding_sr, rel_embedding_tg = self.cam.relation_embedding_sr.weight, self.cam.relation_embedding_tg.weight
        sim_graph_embedding = cosine_similarity_nbyn(graph_embedding_sr, graph_embedding_tg).cpu().detatch().numpy()
        sim_rel_embedding = cosine_similarity_nbyn(rel_embedding_sr, rel_embedding_tg).cpu().detatch().numpy()
        e_rows, e_cols = linear_sum_assignment(sim_graph_embedding)
        r_rows, r_cols = linear_sum_assignment(sim_rel_embedding)
