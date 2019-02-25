import torch
import torch.nn as nn
import torch.nn.functional as F
from .functions import str2int4triples, multi_process_get_nearest_neighbor
from .models import GAT
from .layers import DoubleEmbedding
from .adjacency_matrix import SpTwinAdj
from .CrossGraphCompletion import CrossGraphCompletion
from .torch_functions import cosine_similarity_nbyn, torch_l2distance
from tools.timeit import timeit
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
        self.entity_embedding = DoubleEmbedding(num_entity_sr, num_entity_tg, dim, type='entity')
        self.relation_embedding = DoubleEmbedding(len(cgc.id2relation_sr), len(cgc.id2entity_tg), dim, type='relation')
        if self.is_cuda:
            self.adj_sr = self.adj_sr.cuda()
            self.adj_tg = self.adj_tg.cuda()

    def trans_e(self, ent_embedding, rel_embedding, h_list, t_list, r_list):
        h = ent_embedding[h_list]
        t = ent_embedding[t_list]
        r = rel_embedding[r_list]
        # shape [num, 2*nega + 1, dim]
        return h + r - t


    def forward(self, sr_data, tg_data, h_list_sr, h_list_tg, t_list_sr, t_list_tg, r_list_sr, r_list_tg):
        graph_embedding_sr, graph_embedding_tg = self.entity_embedding.weight
        rel_embedding_sr, rel_embedding_tg = self.relation_embedding.weight
        output_sr = self.sp_gat(graph_embedding_sr, self.adj_sr)
        output_tg = self.sp_gat(graph_embedding_tg, self.adj_tg)
        sr_transe_score = self.trans_e(output_sr, rel_embedding_sr, h_list_sr, t_list_sr, r_list_sr)
        tg_transe_score = self.trans_e(output_tg, rel_embedding_tg, h_list_tg, t_list_tg, r_list_tg)
        transe_score = torch.cat((sr_transe_score, tg_transe_score), dim=0)
        # print(sr_transe_score.size())
        return output_sr[sr_data], output_tg[tg_data], transe_score

    @timeit
    def negative_sample(self, sr_data, tg_data):
        graph_embedding_sr, graph_embedding_tg = self.entity_embedding.weight
        output_sr = self.sp_gat(graph_embedding_sr, self.adj_sr)
        output_tg = self.sp_gat(graph_embedding_tg, self.adj_tg)
        sr_repre = output_sr[sr_data]
        tg_repre =  output_tg[tg_data]
        sim_sr = torch_l2distance(sr_repre.detach(), output_sr.detach()).cpu().numpy()
        sim_tg = torch_l2distance(tg_repre.detach(), output_tg.detach()).cpu().numpy()
        sr_nns = multi_process_get_nearest_neighbor(sim_sr, sr_data.cpu().numpy())
        tg_nns = multi_process_get_nearest_neighbor(sim_tg, tg_data.cpu().numpy())
        return sr_nns, tg_nns

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
