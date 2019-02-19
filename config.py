import torch, os
from torch import optim
from project_path import bin_dir
from graph_completion.GCN import GCN
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from graph_completion.AlignmentDataset import AliagnmentDataset
from torch.utils.data import DataLoader
from graph_completion.functions import GCNAlignLoss
from tools.print_time_info import print_time_info
from graph_completion.functions import get_hits
from tools.timeit import timeit

class Config(object):

    def __init__(self, directory):
        # training
        self.num_epoch = 1000
        self.directory = directory
        self.train_seeds_ratio = 0.3

        # model
        self.num_layer = 2
        self.embedding_dim = 300

        # dataset
        self.shuffle = True
        self.batch_size = 64
        self.num_workers = 4  # for the data_loader
        self.nega_sample_num = 24  # number of negative samples for each positive one

        # hyper parameter
        self.beta = 0.01  # ratio of relation loss
        self.dropout_rate = 0.0
        self.entity_gamma = 3.0  # margin for entity loss
        self.relation_gamma = 3.0  # margin for relation loss

        # cuda
        self.is_cuda = True

    def init(self, load=True):
        language_pair_dirs = list(directory.glob('*_en'))
        if load:
            self.cgc = CrossGraphCompletion.restore(language_pair_dirs[0] / 'running_temp')
        else:
            self.cgc = CrossGraphCompletion(language_pair_dirs[0], self.train_seeds_ratio)
            self.cgc.init()
            self.cgc.save(language_pair_dirs[0] / 'running_temp')

    def train(self):
        cgc = self.cgc
        entity_seeds = AliagnmentDataset(cgc.train_entity_seeds, self.nega_sample_num, len(cgc.id2entity_sr),
                                         len(cgc.id2entity_tg))
        entity_seeds = DataLoader(entity_seeds, batch_size=self.batch_size, shuffle=self.shuffle,
                                  num_workers=self.num_workers)
        relation_seeds = AliagnmentDataset(cgc.relation_seeds, self.nega_sample_num, len(cgc.id2relation_sr),
                                           len(cgc.id2relation_tg))
        relation_seeds = DataLoader(relation_seeds, batch_size=self.batch_size, shuffle=self.shuffle,
                                    num_workers=self.num_workers)

        self.gcn = GCN(self.is_cuda, cgc, self.num_layer, self.embedding_dim, self.dropout_rate)
        if self.is_cuda:
            self.gcn.cuda()
        # for name, param in gcn.named_parameters():
        #     print(name, param.requires_grad)
        optimizer = optim.Adam(self.gcn.parameters(), lr=0.01)
        criterion_entity = GCNAlignLoss(self.entity_gamma, cuda=self.is_cuda)
        criterion_relation = GCNAlignLoss(self.relation_gamma, re_scale=self.beta, cuda=self.is_cuda)

        if self.is_cuda:
            criterion_entity.cuda()
            criterion_relation.cuda()

        batch_num = len(entity_seeds)
        for epoch in range(self.num_epoch):
            relation_seeds_iter = iter(relation_seeds)
            print_time_info('Epoch: %d started!' % (epoch + 1))
            self.gcn.train()
            loss_acc = 0
            for i_batch, batch in enumerate(entity_seeds):
                optimizer.zero_grad()
                sr_data, tg_data = batch
                if self.is_cuda:
                    sr_data, tg_data = sr_data.cuda(), tg_data.cuda()
                try:
                    sr_rel_data, tg_rel_data = next(relation_seeds_iter)
                except StopIteration:
                    relation_seeds_iter = iter(relation_seeds)
                    sr_rel_data, tg_rel_data = next(relation_seeds_iter)
                if self.is_cuda:
                    sr_rel_data, tg_rel_data = sr_rel_data.cuda(), tg_rel_data.cuda()
                repre_e_sr, repre_e_tg, repre_r_sr, repre_r_tg = self.gcn(sr_data, tg_data, sr_rel_data, tg_rel_data)
                entity_loss = criterion_entity(repre_e_sr, repre_e_tg)
                # relation_loss = criterion_relation(repre_r_sr, repre_r_tg)
                # loss = sum([entity_loss, relation_loss])
                loss = entity_loss
                loss.backward()
                loss_acc += float(loss)
                optimizer.step()
                if (i_batch) % 10 == 0:
                    print('\rBatch: %d/%d; loss = %f' % (i_batch + 1, batch_num, loss_acc/ (i_batch + 1)), end='')
            print('')
            self.evaluate()

    @timeit
    def evaluate(self):
        self.gcn.eval()
        print('Training: ' + str(self.gcn.training))
        sr_data, tg_data = list(zip(*self.cgc.test_entity_seeds))
        sr_data = [int(ele) for ele in sr_data]
        tg_data = [int(ele) for ele in tg_data]
        sr_data = torch.tensor(sr_data, dtype=torch.int64)
        tg_data = torch.tensor(tg_data, dtype=torch.int64)
        if self.is_cuda:
            sr_data = sr_data.cuda()
            tg_data = tg_data.cuda()
        repre_sr, repre_tg = self.gcn.predict(sr_data, tg_data)
        get_hits(repre_sr, repre_tg)

    def loop(self):
        # todo: finish it
        train_seeds_ratio = 0.3
        directory = bin_dir / 'dbp15k'
        language_pair_dirs = list(directory.glob('*_en'))
        for local_directory in language_pair_dirs:
            cgc = CrossGraphCompletion(local_directory, train_seeds_ratio)
            cgc.init()
            cgc.save(local_directory / 'running_temp')

    def set_cuda(self, is_cuda):
        self.is_cuda = is_cuda

if __name__ == '__main__':
    # CUDA_LAUNCH_BLOCKING=1
    import sys
    directory = bin_dir / 'dbp15k'
    config = Config(directory)
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    except IndexError:
        config.set_cuda(False)
    config.init(True)
    config.train()
    # config.init(True)
