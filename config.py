import torch
import shutil
from graph_completion.nets import GATNet
from pathlib import Path
from tensorboardX import SummaryWriter
from utils.tools import print_time_info, timeit
from models.torch_functions import SpecialLossAlign, SpecialLossRule
from utils.Datasets import AliagnmentDataset, TripleDataset, RuleDataset
from utils.functions import set_random_seed
from utils.functions import get_hits
from graph_completion.cross_graph_completion import CrossGraphCompletion


class Config(object):

    def __init__(self):
        # boot strap
        self.train_bootstrap = False
        self.ent_seeds = list()
        self.aligned_entites = set()
        self.aligned_relations = set()

        # training
        self.patience = 10
        self.pre_train = 10
        self.split_num = 1  # split triple dataset into parts
        # self.min_epoch = 3000
        self.bad_result = 0
        self.now_epoch = 0
        self.best_hits_1 = (0, 0, 0)  # (epoch, sr, tg)
        self.num_epoch = 500
        self.update_cycle = 10
        self.rule_infer = True
        self.rule_transfer = True
        self.train_seeds_ratio = 0.3

        # model
        self.w_adj = ''
        self.net = None
        self.sparse = True
        self.optimizer = None
        self.nheads = 1
        self.num_layer = 2
        self.non_acylic = True
        self.embedding_dim = 300
        self.graph_completion = True

        # dataset
        self.shuffle = True
        self.batch_size = 64
        self.num_workers = 4  # for the data_loader
        self.nega_n_e = 25  # number of negative samples for each positive one
        self.nega_n_r = 2

        # hyper parameter
        self.lr = 1e-3
        self.beta = 1.0  # ratio of transe loss
        self.alpha = 0.2  # alpha for the leaky relu
        self.rule_scale = 0.9
        self.l2_penalty = 0.0001
        self.dropout_rate = 0.5
        self.align_gamma = 3.0  # margin for entity loss
        self.rel_align_gamma = 1.0
        self.rule_gamma = 0.12  # margin for relation loss
        # cuda
        self.is_cuda = True

    def init(self, directory, load=False):
        set_random_seed()
        directory = Path(directory)
        self.graph_pair = directory.name
        if load:
            try:
                self.cgc = CrossGraphCompletion.restore(directory / 'running_temp')
            except FileNotFoundError:
                print_time_info('CrossGraphCompletion cache file not found, start from the beginning.')
                self.cgc = CrossGraphCompletion(directory, self.train_seeds_ratio, self.rule_transfer,
                                                self.graph_completion)
                self.cgc.init()
                self.cgc.save(directory / 'running_temp')
        else:
            self.cgc = CrossGraphCompletion(directory, self.train_seeds_ratio, self.rule_transfer,
                                            self.graph_completion)
            self.cgc.init()
            self.cgc.save(directory / 'running_temp')
        self.cgc.check()

    def train(self):
        cgc = self.cgc
        with torch.no_grad():
            triples_sr = TripleDataset(cgc.triples_sr, self.nega_n_r)
            triples_tg = TripleDataset(cgc.triples_tg, self.nega_n_r)
            triples_data_sr = triples_sr.get_all()
            triples_data_tg = triples_tg.get_all()
            rules_sr = RuleDataset(cgc, 'new_triple_premises_sr', cgc.triples_sr, list(cgc.id2relation_sr.keys()),
                                   self.nega_n_r)
            rules_tg = RuleDataset(cgc, 'new_triple_premises_tg', cgc.triples_tg, list(cgc.id2relation_tg.keys()),
                                   self.nega_n_r)
            rules_data_sr = rules_sr.get_all()
            rules_data_tg = rules_tg.get_all()
            ad = AliagnmentDataset(cgc, 'entity_seeds', self.nega_n_e, len(cgc.id2entity_sr), len(cgc.id2entity_tg),
                                   self.is_cuda)
            ad_data = ad.get_all()
            ad_rel = AliagnmentDataset(cgc, 'relation_seeds', self.nega_n_r, len(cgc.id2relation_sr),
                                       len(cgc.id2relation_tg), self.is_cuda)
            ad_rel_data = ad_rel.get_all()

        if self.is_cuda:
            self.net.cuda()
            ad_data = [data.cuda() for data in ad_data]
            ad_rel_data = [data.cuda() for data in ad_rel_data]
            triples_data_sr = [data.cuda() for data in triples_data_sr]
            triples_data_tg = [data.cuda() for data in triples_data_tg]
            rules_data_sr = [data.cuda() for data in rules_data_sr]
            rules_data_tg = [data.cuda() for data in rules_data_tg]

        optimizer = self.optimizer(self.net.parameters(), lr=self.lr, weight_decay=self.l2_penalty)
        criterion_align = SpecialLossAlign(self.align_gamma, cuda=self.is_cuda)
        criterion_rel = SpecialLossAlign(self.rel_align_gamma, cuda=self.is_cuda)
        criterion_transe = SpecialLossRule(self.rule_gamma, cuda=self.is_cuda)
        criterion_rule = SpecialLossRule(self.rule_gamma, cuda=self.is_cuda)

        for epoch in range(self.num_epoch):
            self.net.train()
            optimizer.zero_grad()
            repre_sr, repre_tg, sr_rel_repre, tg_rel_repre, transe_tv, rule_tv = self.net(ad_data, ad_rel_data,
                                                                                          triples_data_sr,
                                                                                          triples_data_tg,
                                                                                          rules_data_sr, rules_data_tg)

            align_loss = criterion_align(repre_sr, repre_tg)
            rel_align_loss = criterion_rel(sr_rel_repre, tg_rel_repre)
            transe_loss = criterion_transe(transe_tv)
            if self.rule_infer:
                rule_loss = criterion_rule(rule_tv)
                loss = sum([align_loss, transe_loss, rel_align_loss, rule_loss])
            else:
                rule_loss = 0.0
                loss = sum([align_loss, rel_align_loss, transe_loss])
            loss.backward()
            optimizer.step()
            print_time_info(
                'Epoch: %d; align loss = %.4f; relation align loss = %.4f; transe loss = %.4f; rule loss = %.4f.' % (
                    epoch + 1, float(align_loss), float(rel_align_loss), float(transe_loss), float(rule_loss)))
            self.writer.add_scalars('data/Loss',
                                    {'Align Loss': float(align_loss), 'TransE Loss': float(transe_loss),
                                     'Rule Loss': float(rule_loss), 'Relation Align Loss': float(rel_align_loss)},
                                    epoch)
            self.now_epoch += 1
            if (epoch + 1) % self.update_cycle == 0:
                self.evaluate()
                ad_data, ad_rel_data, triples_data_sr, triples_data_tg, rules_data_sr, rules_data_tg = self.negative_sampling(
                    ad, ad_rel, triples_sr, triples_tg, rules_sr, rules_tg)
                if self.is_cuda:
                    torch.cuda.empty_cache()
                    ad_data = [data.cuda() for data in ad_data]
                    ad_rel_data = [data.cuda() for data in ad_rel_data]
                    triples_data_sr = [data.cuda() for data in triples_data_sr]
                    triples_data_tg = [data.cuda() for data in triples_data_tg]
                    rules_data_sr = [data.cuda() for data in rules_data_sr]
                    rules_data_tg = [data.cuda() for data in rules_data_tg]

    @timeit
    def negative_sampling(self, ad, ad_rel, triples_sr, triples_tg, rules_sr, rules_tg):
        self.net.eval()
        with torch.no_grad():
            ad_seeds = ad.get_seeds()
            ad_rel_seeds = ad_rel.get_seeds()
            if self.is_cuda:
                ad_seeds = [seeds.cuda() for seeds in ad_seeds]
                ad_rel_seeds = [seeds.cuda() for seeds in ad_rel_seeds]

            sample_relation = True
            if self.graph_pair == 'dbp_yg':
                sample_relation = False

            # For Alignment
            sr_nns, tg_nns, sr_rel_nns, tg_rel_nns = self.net.negative_sample(ad_seeds, ad_rel_seeds, sample_relation)
            ad.update_negative_sample(sr_nns, tg_nns)
            if sample_relation:
                ad_rel.update_negative_sample(sr_rel_nns, tg_rel_nns)
            else:
                ad_rel.init()

            ad_data = ad.get_all()
            ad_rel_data = ad_rel.get_all()
            triples_sr.init(), triples_tg.init(), rules_sr.init(), rules_tg.init()

            # For TransE
            triples_data_sr = triples_sr.get_all()
            triples_data_tg = triples_tg.get_all()
            # For rules
            rules_data_sr = rules_sr.get_all()
            rules_data_tg = rules_tg.get_all()
            return ad_data, ad_rel_data, triples_data_sr, triples_data_tg, rules_data_sr, rules_data_tg

    @timeit
    @torch.no_grad()
    def evaluate(self):
        self.net.eval()
        sr_data, tg_data = list(zip(*self.cgc.test_entity_seeds))
        sr_data = torch.tensor(sr_data, dtype=torch.int64)
        tg_data = torch.tensor(tg_data, dtype=torch.int64)
        if self.is_cuda:
            sr_data = sr_data.cuda()
            tg_data = tg_data.cuda()
        sim = self.net.predict((sr_data, tg_data))
        for x, y in self.aligned_entites:
            sim[x, y] -= 1.0
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(sim)
        self.writer.add_scalars('data/Hits@N', {'Hits@1 sr': top_lr[0],
                                                'Hits@10 sr': top_lr[1],
                                                'Hits@1 tg': top_rl[0],
                                                'Hits@10 tg': top_rl[1]},
                                self.now_epoch)
        self.writer.add_scalars('data/Rank', {'MR sr': mr_lr,
                                              'MRR sr': mrr_lr,
                                              'MR tg': mr_rl,
                                              'MRR tg': mrr_rl},
                                self.now_epoch)

        if top_lr[0] + top_rl[0] > self.best_hits_1[1] + self.best_hits_1[2]:
            self.best_hits_1 = (self.now_epoch, top_lr[0], top_rl[0])
            self.bad_result = 0
        else:
            self.bad_result += 1
        print_time_info('Current best Hits@1 at the %dth epoch: (%.2f, %.2f)' % (self.best_hits_1))

    def print_parameter(self, file=None):
        parameters = self.__dict__
        print_time_info('Parameter setttings:', dash_top=True, file=file)
        print('\tNet: ', type(self.net).__name__, file=file)
        for key, value in parameters.items():
            if type(value) in {int, float, str, bool}:
                print('\t%s:' % key, value, file=file)
        print('---------------------------------------', file=file)

    def init_log(self, log_dir):
        log_dir = Path(log_dir)
        if log_dir.exists():
            print('Warning: we will remove %s' % (str(log_dir)))
            shutil.rmtree(str(log_dir))
        log_dir.mkdir()
        comment = log_dir.name
        self.writer = SummaryWriter(str(log_dir))
        with open(log_dir / 'parameters.txt', 'w') as f:
            print_time_info(comment, file=f)
            self.print_parameter(f)
        print_time_info('Successfully initialized log in "%s" directory!' % log_dir)

    def set_cuda(self, is_cuda):
        self.is_cuda = is_cuda

    def set_net(self):
        self.net = GATNet(self.rule_scale, self.cgc, self.num_layer, self.embedding_dim, self.nheads, self.alpha,
                          self.rule_infer, self.w_adj, self.dropout_rate, self.non_acylic, self.is_cuda)

    def set_graph_completion(self, graph_completion):
        self.graph_completion = graph_completion

    def set_learning_rate(self, learning_rate):
        self.lr = learning_rate

    def set_dropout(self, dropout):
        self.dropout_rate = dropout

    def set_align_gamma(self, align_gamma):
        self.align_gamma = align_gamma

    def set_rule_gamma(self, rule_gamma):
        self.rule_gamma = rule_gamma

    def set_dim(self, dim):
        self.embedding_dim = dim

    def set_nheads(self, nheads):
        self.nheads = nheads

    def set_l2_penalty(self, l2_penalty):
        self.l2_penalty = l2_penalty

    def set_num_layer(self, num_layer):
        self.num_layer = num_layer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_beta(self, beta):
        self.beta = beta

    def set_update_cycle(self, update_cycle):
        self.update_cycle = update_cycle

    def set_w_adj(self, w_adj):
        self.w_adj = w_adj

    def set_rule_infer(self, rule_infer):
        self.rule_infer = rule_infer

    def set_bootstrap(self, bootstrap):
        self.train_bootstrap = bootstrap

    def set_train_seed_ratio(self, seed_ratio):
        self.train_seeds_ratio = seed_ratio

    def set_rule_transfer(self, rule_transfer):
        self.rule_transfer = rule_transfer

    def set_rel_align_gamma(self, rel_align_gamma):
        self.rel_align_gamma = rel_align_gamma