import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .functions import str2int4triples, multi_process_get_nearest_neighbor
from .models import GAT
from .layers import DoubleEmbedding
from .adjacency_matrix import SpTwinAdj, SpTwinCAW
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
    def __init__(self, rule_scale, cgc, num_layer, dim, nheads, sp, alpha=0.2, w_adj=False, *args, **kwargs):
        super(GATNet, self).__init__(*args, **kwargs)
        assert isinstance(cgc, CrossGraphCompletion)
        self.dim = dim
        num_entity_sr = len(cgc.id2entity_sr)
        num_entity_tg = len(cgc.id2entity_tg)
        if not w_adj:
            self.sp_twin_adj = SpTwinAdj(cgc, self.non_acylic, cuda=self.is_cuda)
        else:
            self.sp_twin_adj = SpTwinCAW(rule_scale, cgc, self.non_acylic, cuda=self.is_cuda)
        self.sp_gat = GAT(dim, dim, nheads, num_layer, self.dropout_rate, alpha, sp, w_adj, self.is_cuda)
        self.entity_embedding = DoubleEmbedding(num_entity_sr, num_entity_tg, dim, type='entity')
        self.relation_embedding = DoubleEmbedding(len(cgc.id2relation_sr), len(cgc.id2relation_tg), dim,
                                                  type='relation')

    def trans_e(self, ent_embedding, rel_embedding, triples_data):
        h_list, t_list, r_list = triples_data
        h = ent_embedding[h_list]
        t = ent_embedding[t_list]
        r = rel_embedding[r_list]
        score = h + r - t
        return score

    def truth_value(self, score):
        score = torch.norm(score, p=1, dim=-1, keepdim=False)
        return 1 - score / (3 * math.sqrt(self.dim))

    def rule(self, rules_data, transe_tv, ent_embedding, rel_embedding):
        # trans_e_score shape = [num, dim]
        # r_h shape = [num, 1], r_r shape = [num, 2], premises shape = [num, 2]
        r_h, r_t, r_r, premises = rules_data
        pad_value = torch.tensor([[1.0]])
        if self.is_cuda:
            pad_value = pad_value.cuda()
        transe_tv = torch.cat((transe_tv, pad_value), dim=0)  # for padding
        rule_score = self.trans_e(ent_embedding, rel_embedding, (r_h, r_t, r_r))
        rule_score = self.truth_value(rule_score)
        f1_score = transe_tv[premises[:, 0]]
        f2_score = transe_tv[premises[:, 1]]
        rule_score = 1 + f1_score * f2_score * (rule_score - 1)
        return rule_score

    def __forward_gat__(self, sr_data, tg_data):
        ent_embedding_sr, ent_embedding_tg = self.entity_embedding.weight
        rel_embedding_sr, rel_embedding_tg = self.relation_embedding.weight
        adj_sr, adj_tg = self.sp_twin_adj(rel_embedding_sr, rel_embedding_tg)
        graph_embedding_sr = self.sp_gat(ent_embedding_sr, adj_sr)
        graph_embedding_tg = self.sp_gat(ent_embedding_tg, adj_tg)
        normed_graph_embedding_sr = F.normalize(graph_embedding_sr, p=2, dim=-1)
        normed_graph_embedding_tg = F.normalize(graph_embedding_tg, p=2, dim=-1)
        rel_embedding_sr = F.normalize(rel_embedding_sr, dim=-1, p=2)
        rel_embedding_tg = F.normalize(rel_embedding_tg, dim=-1, p=2)
        return graph_embedding_sr[sr_data], graph_embedding_tg[
            tg_data], normed_graph_embedding_sr, normed_graph_embedding_tg, rel_embedding_sr, rel_embedding_tg

    def forward(self, sr_data, tg_data, triples_data_sr, triples_data_tg, rules_data_sr, rules_data_tg):
        sr_data_repre, tg_data_repre, graph_embedding_sr, graph_embedding_tg, rel_embedding_sr, rel_embedding_tg = self.__forward_gat__(
            sr_data, tg_data)
        sr_transe_tv = self.truth_value(self.trans_e(graph_embedding_sr, rel_embedding_sr, triples_data_sr))
        tg_transe_tv = self.truth_value(self.trans_e(graph_embedding_tg, rel_embedding_tg, triples_data_tg))
        sr_rule_tv = self.rule(rules_data_sr, sr_transe_tv[:, :1], graph_embedding_sr, rel_embedding_sr)
        tg_rule_tv = self.rule(rules_data_tg, tg_transe_tv[:, 1:], graph_embedding_tg, rel_embedding_tg)
        transe_tv = torch.cat((sr_transe_tv, tg_transe_tv), dim=0)
        rule_tv = torch.cat((sr_rule_tv, tg_rule_tv), dim=0)
        return sr_data_repre, tg_data_repre, transe_tv, rule_tv

    def negative_sample(self, sr_data, tg_data):
        sr_data_repre, tg_data_repre, _, _, _, _ = self.__forward_gat__(sr_data, tg_data)
        sr_data_repre, tg_data_repre = sr_data_repre.detach(), tg_data_repre.detach()
        sim_sr = torch_l2distance(sr_data_repre, sr_data_repre).cpu().numpy()
        sim_tg = torch_l2distance(tg_data_repre, tg_data_repre).cpu().numpy()
        sr_nns = multi_process_get_nearest_neighbor(sim_sr, sr_data.cpu().numpy())
        tg_nns = multi_process_get_nearest_neighbor(sim_tg, tg_data.cpu().numpy())
        if self.is_cuda:
            torch.cuda.empty_cache()
        return sr_nns, tg_nns

    def predict(self, sr_data, tg_data):
        sr_data_repre, tg_data_repre, _, _, _, _ = self.__forward_gat__(sr_data, tg_data)
        sr_data_repre, tg_data_repre = sr_data_repre.detach(), tg_data_repre.detach()
        sim = torch_l2distance(sr_data_repre, tg_data_repre).cpu().numpy()
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
