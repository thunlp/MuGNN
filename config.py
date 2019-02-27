import torch
from torch import optim
from tools.timeit import timeit
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tools.print_time_info import print_time_info
from graph_completion.nets import GATNet
from graph_completion.functions import str2int4triples
from graph_completion.torch_functions import SpecialLossTransE, SpecialLossAlign, SpecialLossRule
from graph_completion.Datasets import AliagnmentDataset, TripleDataset, EpochDataset, RuleDataset
from graph_completion.torch_functions import set_random_seed
from graph_completion.functions import get_hits
from graph_completion.CrossGraphCompletion import CrossGraphCompletion


class Config(object):

    def __init__(self, directory):
        # training
        self.patience = 10
        self.pre_train = 10
        self.split_num = 1  # split triple dataset into parts
        self.min_epoch = 3000
        self.bad_result = 0
        self.now_epoch = 0
        self.best_hits_1 = (0, 0, 0)  # (epoch, sr, tg)
        self.num_epoch = 10000
        self.update_cycle = 10
        self.directory = directory
        self.train_seeds_ratio = 0.3

        # model
        self.net = None
        self.w_adj = False
        self.sparse = True
        self.optimizer = None
        self.nheads = 4
        self.num_layer = 2
        self.non_acylic = True
        self.embedding_dim = 300
        self.graph_completion = False

        # dataset
        self.shuffle = True
        self.batch_size = 64
        self.num_workers = 4  # for the data_loader
        self.nega_n_e = 25  # number of negative samples for each positive one
        self.nega_n_r = 2
        self.corrupt = False

        # hyper parameter
        self.lr = 1e-3
        self.beta = 1.0  # ratio of transe loss
        self.alpha = 0.2  # alpha for the leaky relu
        self.rule_scale = 0.9
        self.l2_penalty = 0.0001
        self.dropout_rate = 0.5
        self.align_gamma = 3.0  # margin for entity loss
        self.rule_gamma = 3.0  # margin for relation loss
        # cuda
        self.is_cuda = True

    def init(self, load=True):
        set_random_seed()
        language_pair_dirs = list(self.directory.glob('*_en'))
        directory = language_pair_dirs[0]
        if load:
            self.cgc = CrossGraphCompletion.restore(directory / 'running_temp')
        else:
            self.cgc = CrossGraphCompletion(directory, self.train_seeds_ratio, self.graph_completion)
            self.cgc.init()
            self.cgc.save(directory / 'running_temp')
        self.cgc.check()

    def train(self):
        cgc = self.cgc
        with torch.no_grad():
            triples_sr = TripleDataset(cgc.triples_sr, self.nega_n_r, corruput=self.corrupt)
            triples_tg = TripleDataset(cgc.triples_tg, self.nega_n_r, corruput=self.corrupt)
            triples_data_sr = triples_sr.get_all()
            triples_data_tg = triples_tg.get_all()
            rules_sr = RuleDataset(cgc.new_triple_premises_sr, cgc.triples_sr, list(cgc.id2relation_sr.keys()),
                                   self.nega_n_r)
            rules_tg = RuleDataset(cgc.new_triple_premises_tg, cgc.triples_tg, list(cgc.id2relation_tg.keys()),
                                   self.nega_n_r)
            rules_data_sr = rules_sr.get_all()
            rules_data_tg = rules_tg.get_all()
            ad = AliagnmentDataset(cgc.train_entity_seeds, self.nega_n_e, len(cgc.id2entity_sr), len(cgc.id2entity_tg),
                                   self.is_cuda, corruput=self.corrupt)
            sr_data, tg_data = ad.get_all()

        if self.is_cuda:
            self.net.cuda()
            sr_data, tg_data = sr_data.cuda(), tg_data.cuda()
            triples_data_sr = [data.cuda() for data in triples_data_sr]
            triples_data_tg = [data.cuda() for data in triples_data_tg]
            rules_data_sr = [data.cuda() for data in rules_data_sr]
            rules_data_tg = [data.cuda() for data in rules_data_tg]

        optimizer = self.optimizer(self.net.parameters(), lr=self.lr, weight_decay=self.l2_penalty)
        criterion_align = SpecialLossAlign(self.align_gamma, cuda=self.is_cuda)
        # criterion_transe = SpecialLossTransE(self.transe_gamma, p=2, re_scale=self.beta, cuda=self.is_cuda)
        criterion_trase = SpecialLossRule(self.rule_gamma, cuda=self.is_cuda)
        criterion_rule = SpecialLossRule(self.rule_gamma, re_scale=self.beta, cuda=self.is_cuda)
        for epoch in range(self.num_epoch):
            self.net.train()
            optimizer.zero_grad()
            repre_sr, repre_tg, transe_tv, rule_tv = self.net(sr_data, tg_data, triples_data_sr, triples_data_tg,
                                                              rules_data_sr, rules_data_tg)
            align_loss = criterion_align(repre_sr, repre_tg)
            transe_loss = criterion_trase(transe_tv)
            rule_loss = criterion_rule(rule_tv)
            if epoch < self.pre_train:
                loss = sum([transe_loss, rule_loss])
            else:
                loss = sum([align_loss, transe_loss, rule_loss])
            loss.backward()
            optimizer.step()
            print_time_info(
                'Epoch: %d; align loss = %.4f; transe loss = %.4f; rule loss = %.4f.' % (
                    epoch + 1, float(align_loss), float(transe_loss), float(rule_loss)))
            self.writer.add_scalars('data/Loss',
                                    {'Align Loss': float(align_loss), 'TransE Loss': float(transe_loss),
                                     'Rule Loss': float(rule_loss)}, epoch)
            self.now_epoch += 1
            if epoch > self.pre_train:
                self.evaluate()
            if (epoch + 1) % self.update_cycle == 0:
                sr_data, tg_data, triples_data_sr, triples_data_tg, rules_data_sr, rules_data_tg = self.negative_sampling(
                    ad, triples_sr, triples_tg, rules_sr, rules_tg)
                if self.is_cuda:
                    torch.cuda.empty_cache()
                    sr_data, tg_data = sr_data.cuda(), tg_data.cuda()
                    triples_data_sr = [data.cuda() for data in triples_data_sr]
                    triples_data_tg = [data.cuda() for data in triples_data_tg]
                    rules_data_sr = [data.cuda() for data in rules_data_sr]
                    rules_data_tg = [data.cuda() for data in rules_data_tg]
    @timeit
    def negative_sampling(self, ad, triples_sr, triples_tg, rules_sr, rules_tg):
        with torch.no_grad():
            sr_seeds, tg_seeds = ad.get_seeds()
            if self.is_cuda:
                sr_seeds = sr_seeds.cuda()
                tg_seeds = tg_seeds.cuda()
            if self.now_epoch > self.pre_train:
                # For Alignment
                sr_nns, tg_nns = self.net.negative_sample(sr_seeds, tg_seeds)
                ad.update_negative_sample(sr_nns, tg_nns)
            sr_data, tg_data = ad.get_all()
            # For TransE
            triples_data_sr = triples_sr.init().get_all()
            triples_data_tg = triples_tg.init().get_all()
            # For rules
            rules_data_sr = rules_sr.init().get_all()
            rules_data_tg = rules_tg.init().get_all()
            return sr_data, tg_data, triples_data_sr, triples_data_tg, rules_data_sr, rules_data_tg

    @timeit
    def evaluate(self):
        self.net.eval()
        sr_data, tg_data = list(zip(*self.cgc.test_entity_seeds))
        sr_data = [int(ele) for ele in sr_data]
        tg_data = [int(ele) for ele in tg_data]
        sr_data = torch.tensor(sr_data, dtype=torch.int64)
        tg_data = torch.tensor(tg_data, dtype=torch.int64)
        if self.is_cuda:
            sr_data = sr_data.cuda()
            tg_data = tg_data.cuda()
        sim = self.net.predict(sr_data, tg_data)
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

        if self.now_epoch < self.min_epoch:
            return
        if self.bad_result >= self.patience:
            print_time_info('My patience is limited. It is time to stop!', dash_bot=True)
            exit()

    def print_parameter(self, file=None):
        parameters = self.__dict__
        print_time_info('Parameter setttings:', dash_top=True, file=file)
        print('\tNet: ', type(self.net).__name__, file=file)
        for key, value in parameters.items():
            if type(value) in {int, float, str, bool}:
                print('\t%s:' % key, value, file=file)
        print('---------------------------------------', file=file)

    def init_log(self, comment):
        from project_path import project_dir
        log_dir = project_dir / 'log'  #
        if not log_dir.exists():
            log_dir.mkdir()
        log_dir = log_dir / comment
        if log_dir.exists():
            raise FileExistsError('The directory already exists!')
        log_dir.mkdir()
        self.writer = SummaryWriter(str(log_dir))
        with open(log_dir / 'parameters.txt', 'w') as f:
            print_time_info(comment, file=f)
            self.print_parameter(f)
        print_time_info('Successfully initialized log in "%s" directory!' % log_dir)

    def set_cuda(self, is_cuda):
        self.is_cuda = is_cuda

    def set_net(self):
        self.net = GATNet(self.rule_scale, self.cgc, self.num_layer, self.embedding_dim, self.nheads, self.sparse,
                          self.alpha, self.w_adj, self.dropout_rate, self.non_acylic, self.is_cuda)

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

    def set_sparse(self, sparse):
        self.sparse = sparse

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_beta(self, beta):
        self.beta = beta

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def set_corrupt(self, corrupt):
        self.corrupt = corrupt

    def set_update_cycle(self, update_cycle):
        self.update_cycle = update_cycle

    def set_w_adj(self, w_adj):
        self.w_adj = w_adj

    def set_rule_scale(self, rule_scale):
        self.rule_scale = rule_scale

    def set_pre_train(self, pre_train):
        self.pre_train = pre_train

    def loop(self, bin_dir):
        # todo: finish it
        train_seeds_ratio = 0.3
        directory = bin_dir / 'dbp15k'
        language_pair_dirs = list(directory.glob('*_en'))
        for local_directory in language_pair_dirs:
            cgc = CrossGraphCompletion(local_directory, train_seeds_ratio)
            cgc.init()
            cgc.save(local_directory / 'running_temp')
