import torch
from torch import optim
from tools.timeit import timeit
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tools.print_time_info import print_time_info
from graph_completion.functions import str2int4triples
from graph_completion.torch_functions import SpecialLoss
from graph_completion.Datasets import AliagnmentDataset, TripleDataset, EpochDataset
from graph_completion.torch_functions import set_random_seed
from graph_completion.functions import get_hits
from graph_completion.CrossGraphCompletion import CrossGraphCompletion


class Config(object):

    def __init__(self, directory):
        # training
        self.patience = 10
        self.min_epoch = 3000
        self.bad_result = 0
        self.now_epoch = 0
        self.best_hits_1 = (0, 0, 0)  # (epoch, sr, tg)
        self.num_epoch = 10000
        self.directory = directory
        self.train_seeds_ratio = 0.3

        # model
        self.net = None
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
        self.l2_penalty = 0.0001
        self.dropout_rate = 0.5
        self.entity_gamma = 3.0  # margin for entity loss
        self.transe_gamma = 3.0  # margin for relation loss
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
        if load:
            self.triples_sr = TripleDataset.restore(directory / 'running_temp' / 'td_sr.pkl')
            self.triples_tg = TripleDataset.restore(directory / 'running_temp' / 'td_tg.pkl')
        else:
            self.triples_sr = TripleDataset(str2int4triples(self.cgc.triples_sr), self.nega_n_r, corruput=self.corrupt)
            self.triples_tg = TripleDataset(str2int4triples(self.cgc.triples_tg), self.nega_n_r, corruput=self.corrupt)
            self.triples_sr.save(directory / 'running_temp' / 'td_sr.pkl')
            self.triples_tg.save(directory / 'running_temp' / 'td_tg.pkl')

    def train(self):
        cgc = self.cgc
        h_sr, t_sr, r_sr = self.triples_sr.get_all()
        h_tg, t_tg, r_tg = self.triples_tg.get_all()
        if self.is_cuda:
            self.net.cuda()
            h_sr, t_sr, r_sr = h_sr.cuda(), t_sr.cuda(), r_sr.cuda()
            h_tg, t_tg, r_tg = h_tg.cuda(), t_tg.cuda(), r_tg.cuda()

        # for the log
        writer = self.writer

        ad = AliagnmentDataset(cgc.train_entity_seeds, self.nega_n_e, len(cgc.id2entity_sr), len(cgc.id2entity_tg),
                               self.is_cuda, corruput=self.corrupt)
        sr_data, tg_data = self.negative_sampling(ad)

        optimizer = self.optimizer(self.net.parameters(), lr=self.lr, weight_decay=self.l2_penalty)
        criterion_align = SpecialLoss(self.entity_gamma, cuda=self.is_cuda)
        criterion_transe = SpecialLoss(self.transe_gamma, p=1, re_scale=self.beta, cuda=self.is_cuda)

        loss_acc = 0
        for epoch in range(self.num_epoch):
            self.net.train()
            if (epoch + 1) % 10 == 0:
                print_time_info('Epoch: %d started!' % (epoch + 1))
            optimizer.zero_grad()
            align_score, sr_transe_score, tg_transe_score = self.net(sr_data, tg_data, h_sr, h_tg, t_sr, t_tg, r_sr,
                                                                     r_tg)
            # print('align score', torch.max(align_score), torch.min(align_score))
            align_loss = criterion_align(align_score)
            sr_transe_loss = criterion_transe(sr_transe_score)
            tg_transe_loss = criterion_transe(tg_transe_score)
            loss = sum([align_loss, sr_transe_loss, tg_transe_loss])
            loss.backward()
            optimizer.step()
            loss_acc += float(loss)
            self.now_epoch += 1
            # if (epoch + 1) % 1 == 0:
            print('\rEpoch: %d; align loss = %f, sr_transe_loss: %f, tg_transe_loss %f.' % (
                epoch + 1, float(align_loss), float(sr_transe_loss), float(tg_transe_loss)))
            writer.add_scalars('data/train_loss', {'Align Loss': align_loss.item(),
                                                   'SR TransE Loss': sr_transe_loss.item(),
                                                   'TG TransE Loss': tg_transe_loss.item()},
                               epoch)
            self.evaluate()
            if (epoch + 1) % 5 == 0:
                sr_data, tg_data = self.negative_sampling(ad)

    def negative_sampling(self, ad):
        sr_seeds, tg_seeds = ad.get_seeds()
        if self.is_cuda:
            sr_seeds = sr_seeds.cuda()
            tg_seeds = tg_seeds.cuda()
        sr_nns, tg_nns = self.net.negative_sample(sr_seeds, tg_seeds)
        ad.update_negative_sample(sr_nns, tg_nns)
        sr_data, tg_data = EpochDataset(ad, self.num_epoch).get_data()
        if self.is_cuda:
            sr_data, tg_data = sr_data.cuda(), tg_data.cuda()
        return sr_data, tg_data

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
        top_lr, top_rl = get_hits(sim)
        self.writer.add_scalars('data/performance', {'Hits@1 sr': top_lr[0],
                                                     'Hits@10 sr': top_lr[1],
                                                     'Hits@1 tg': top_rl[0],
                                                     'Hits@10 tg': top_rl[1]},
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
            self.print_parameter(f)

    def set_cuda(self, is_cuda):
        self.is_cuda = is_cuda

    def set_net(self, net):
        self.net = net(self.cgc, self.num_layer, self.embedding_dim, self.nheads, self.sparse, self.alpha,
                       self.dropout_rate, self.non_acylic, self.is_cuda)

    def set_graph_completion(self, graph_completion):
        self.graph_completion = graph_completion

    def set_learning_rate(self, learning_rate):
        self.lr = learning_rate

    def set_dropout(self, dropout):
        self.dropout_rate = dropout

    def set_gamma(self, gamma):
        self.entity_gamma = gamma

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

    def loop(self, bin_dir):
        # todo: finish it
        train_seeds_ratio = 0.3
        directory = bin_dir / 'dbp15k'
        language_pair_dirs = list(directory.glob('*_en'))
        for local_directory in language_pair_dirs:
            cgc = CrossGraphCompletion(local_directory, train_seeds_ratio)
            cgc.init()
            cgc.save(local_directory / 'running_temp')
