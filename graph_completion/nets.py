import torch, math
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import multi_process_get_nearest_neighbor, get_nearest_neighbor
from models.models import GATmGCN
from models.layers import DoubleEmbedding
from graph_completion.adjacency_matrix import SpTwinAdj, SpRelWeiADJ
from graph_completion.cross_graph_completion import CrossGraphCompletion

__all__ = ['GATNet']


class AlignGraphNet(nn.Module):
    def __init__(self, dropout_rate=0.5, non_acylic=False, cuda=True):
        super(AlignGraphNet, self).__init__()
        self.is_cuda = cuda
        self.non_acylic = non_acylic
        self.dropout_rate = dropout_rate

    def predict(self, ad_data):
        raise NotImplementedError

    def bootstrap(self, *args):
        raise NotImplementedError


class GATNet(AlignGraphNet):
    def __init__(self, rule_scale, cgc, num_layer, dim, nheads, alpha=0.2, rule_infer=False, w_adj='', *args,
                 **kwargs):
        super(GATNet, self).__init__(*args, **kwargs)
        assert isinstance(cgc, CrossGraphCompletion)
        self.dim = dim
        self.rule_infer = rule_infer
        num_entity_sr = len(cgc.id2entity_sr)
        num_entity_tg = len(cgc.id2entity_tg)
        if w_adj == 'adj':
            self.sp_twin_adj = SpTwinAdj(cgc, self.non_acylic, cuda=self.is_cuda)
        elif w_adj == 'rel_adj':
            self.sp_twin_adj = SpRelWeiADJ(cgc, self.non_acylic, cuda=self.is_cuda)
        else:
            raise NotImplementedError
        # self.sp_gat = GAT(dim, dim, nheads, num_layer, self.dropout_rate, alpha, sp, w_adj, self.is_cuda)
        self.sp_gat = GATmGCN(dim, dim, nheads, num_layer, self.dropout_rate, alpha, w_adj, self.is_cuda)
        self.entity_embedding = DoubleEmbedding(num_entity_sr, num_entity_tg, dim, type='entity')
        self.relation_embedding = DoubleEmbedding(len(cgc.id2relation_sr), len(cgc.id2relation_tg), dim,
                                                  type='relation')

    def normalize(self):
        self.entity_embedding.normalize()
        self.relation_embedding.normalize()

    def trans_e(self, ent_embedding, rel_embedding, triples_data):
        h_list, t_list, r_list = triples_data
        h = ent_embedding[h_list]
        t = ent_embedding[t_list]
        r = rel_embedding[r_list]
        score = h + r - t
        return score

    def truth_value(self, score):
        score = torch.norm(score, p=1, dim=-1)
        return 1 - score / 3 / math.sqrt(self.dim)  # (3 * math.sqrt(self.dim))

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

    def __forward_gat__(self, ad_data):
        sr_data, tg_data = ad_data
        ent_embedding_sr, ent_embedding_tg = self.entity_embedding.weight
        rel_embedding_sr, rel_embedding_tg = self.relation_embedding.weight
        adj_sr, adj_tg = self.sp_twin_adj(rel_embedding_sr, rel_embedding_tg)
        graph_embedding_sr = self.sp_gat(ent_embedding_sr, adj_sr)
        graph_embedding_tg = self.sp_gat(ent_embedding_tg, adj_tg)
        rel_embedding_sr = F.normalize(rel_embedding_sr, dim=-1, p=2)
        rel_embedding_tg = F.normalize(rel_embedding_tg, dim=-1, p=2)
        graph_embedding_sr = F.normalize(graph_embedding_sr, dim=-1, p=2)
        graph_embedding_tg = F.normalize(graph_embedding_tg, dim=-1, p=2)
        sr_data_repre = graph_embedding_sr[sr_data]
        tg_data_repre = graph_embedding_tg[tg_data]
        return sr_data_repre, tg_data_repre, graph_embedding_sr, graph_embedding_tg, rel_embedding_sr, rel_embedding_tg

    def forward(self, ad_data, ad_rel_data, triples_data_sr, triples_data_tg, rules_data_sr, rules_data_tg):
        sr_data_repre, tg_data_repre, graph_embedding_sr, graph_embedding_tg, rel_embedding_sr, rel_embedding_tg = self.__forward_gat__(
            ad_data)

        # for relation alignment
        sr_rel_data, tg_rel_data = ad_rel_data
        sr_rel_repre, tg_rel_repre = rel_embedding_sr[sr_rel_data], rel_embedding_tg[tg_rel_data]

        sr_transe_tv = self.truth_value(self.trans_e(graph_embedding_sr, rel_embedding_sr, triples_data_sr))
        tg_transe_tv = self.truth_value(self.trans_e(graph_embedding_tg, rel_embedding_tg, triples_data_tg))
        transe_tv = torch.cat((sr_transe_tv, tg_transe_tv), dim=0)
        if self.rule_infer:
            sr_rule_tv = self.rule(rules_data_sr, sr_transe_tv[:, :1], graph_embedding_sr, rel_embedding_sr)
            tg_rule_tv = self.rule(rules_data_tg, tg_transe_tv[:, 1:], graph_embedding_tg, rel_embedding_tg)
            rule_tv = torch.cat((sr_rule_tv, tg_rule_tv), dim=0)
            return sr_data_repre, tg_data_repre, sr_rel_repre, tg_rel_repre, transe_tv, rule_tv
        else:
            return sr_data_repre, tg_data_repre, sr_rel_repre, tg_rel_repre, transe_tv, []

    def negative_sample(self, ad_data, ad_rel_data, sample_relation=True):
        sr_data_repre, tg_data_repre, _, _, rel_embedding_sr, rel_embedding_tg = self.__forward_gat__(ad_data)

        # for entity alignment
        sr_data_repre, tg_data_repre = sr_data_repre.detach(), tg_data_repre.detach()
        # sim_sr = torch_l2distance(sr_data_repre, sr_data_repre).cpu().numpy()
        # sim_tg = torch_l2distance(tg_data_repre, tg_data_repre).cpu().numpy()
        sim_sr = - torch.mm(sr_data_repre, sr_data_repre.t()).cpu().numpy()
        sim_tg = - torch.mm(tg_data_repre, tg_data_repre.t()).cpu().numpy()
        sr_data, tg_data = ad_data
        # sr_nns = multi_process_get_nearest_neighbor(sim_sr, sr_data.cpu().numpy())
        sr_nns = get_nearest_neighbor(sim_sr, sr_data.cpu().numpy())
        # tg_nns = multi_process_get_nearest_neighbor(sim_tg, tg_data.cpu().numpy())
        tg_nns = get_nearest_neighbor(sim_tg, tg_data.cpu().numpy())

        # def check_different(nns1, nns2):
        #     assert len(nns1) == len(nns2)
        #     for i, j in nns1.items():
        #         assert i in nns2
        #         assert j == nns2[i]
        
        # check_different(sr_nns, sr_nns2)
        # check_different(tg_nns, tg_nns2)
        # print("passed")
        if not sample_relation:
            return sr_nns, tg_nns, None, None

        # for relation alignment
        sr_rel_data, tg_rel_data = ad_rel_data
        sr_rel_repre, tg_rel_repre = rel_embedding_sr[sr_rel_data].detach(), rel_embedding_tg[tg_rel_data].detach()
        # rel_sim_sr = torch_l2distance(sr_rel_repre, sr_rel_repre).cpu().numpy()
        # rel_sim_tg = torch_l2distance(tg_rel_repre, tg_rel_repre).cpu().numpy()
        rel_sim_sr = - torch.mm(sr_rel_repre, sr_rel_repre.t()).cpu().numpy()
        rel_sim_tg = - torch.mm(tg_rel_repre, tg_rel_repre.t()).cpu().numpy()
        # sr_rel_nns = multi_process_get_nearest_neighbor(rel_sim_sr, sr_rel_data.cpu().numpy())
        sr_rel_nns = get_nearest_neighbor(rel_sim_sr, sr_rel_data.cpu().numpy())
        # tg_rel_nns = multi_process_get_nearest_neighbor(rel_sim_tg, tg_rel_data.cpu().numpy())
        tg_rel_nns = get_nearest_neighbor(rel_sim_tg, tg_rel_data.cpu().numpy())


        # check_different(sr_rel_nns, sr_rel_nns2)
        # check_different(tg_rel_nns, tg_rel_nns2)
        # print("passed")
        return sr_nns, tg_nns, sr_rel_nns, tg_rel_nns

    def predict(self, ad_data):
        sr_data_repre, tg_data_repre, _, _, _, _ = self.__forward_gat__(ad_data)
        sr_data_repre, tg_data_repre = sr_data_repre.detach(), tg_data_repre.detach()
        # sim = torch_l2distance(sr_data_repre, tg_data_repre).cpu().numpy()
        sim = - torch.mm(sr_data_repre, tg_data_repre.t()).cpu().numpy()
        return sim

    def bootstrap(self, ad_data, ad_rel_data):
        sr_data_repre, tg_data_repre, _, _, rel_embedding_sr, rel_embedding_tg = self.__forward_gat__(ad_data)
        assert sr_data_repre.size()[0] == tg_data_repre.size()[0]

        sr_data_repre, tg_data_repre = sr_data_repre.detach(), tg_data_repre.detach()
        sim_entity = torch.mm(sr_data_repre, tg_data_repre.t()).cpu().numpy()
        sr_rel_data, tg_rel_data = ad_rel_data
        sr_rel_repre, tg_rel_repre = rel_embedding_sr[sr_rel_data].detach(), rel_embedding_tg[tg_rel_data].detach()
        assert sr_rel_repre.size()[0] == tg_rel_repre.size()[0]
        sim_rel = torch.mm(sr_rel_repre, tg_rel_repre.t()).cpu().numpy()
        return sim_entity, sim_rel