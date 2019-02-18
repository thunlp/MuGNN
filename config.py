import torch, os
from project_path import bin_dir
from graph_completion.GCN import GCN
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from graph_completion.AlignmentDataset import AliagnmentDataset
from torch.utils.data import DataLoader
from graph_completion.functions import GCNAlignLoss
from tools.print_time_info import print_time_info

class Config(object):

    def __init__(self, directory):
        # training
        self.num_epoch = 1000
        self.directory = directory
        self.train_seeds_ratio = 0.3

        # model
        self.num_layer = 3
        self.embedding_dim = 100

        # dataset
        self.shuffle = True
        self.batch_size = 64
        self.num_workers = 2  # for the data_loader
        self.nega_sample_num = 24  # number of negative samples for each positive one

        # hyper parameter
        self.beta = 0.3  # ratio of relation loss
        self.dropout_rate = 0.5
        self.entity_gamma = 3.0  # margin for entity loss
        self.relation_gamma = 3.0  # margin for relation loss

        # cuda
        self.cuda = False

    def init(self):
        language_pair_dirs = list(directory.glob('*_en'))
        if False:
            self.cgc = CrossGraphCompletion(language_pair_dirs[0], train_seeds_ratio)
            self.cgc.init()
            # cgc.save(language_pair_dirs[0] / 'running_temp')
        else:
            self.cgc = CrossGraphCompletion.restore(language_pair_dirs[0] / 'running_temp')

    def train(self):
        cgc = self.cgc
        entity_seeds = AliagnmentDataset(cgc.entity_seeds, self.nega_sample_num, len(cgc.id2entity_sr), len(cgc.id2entity_tg))
        entity_seeds = DataLoader(entity_seeds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        relation_seeds = AliagnmentDataset(cgc.relation_seeds, self.nega_sample_num, len(cgc.id2relation_sr),
                                           len(cgc.id2relation_tg))
        relation_seeds = DataLoader(relation_seeds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        # relation_seeds_iter = iter(relation_seeds)
        # for sr_rel_data, tg_rel_data in relation_seeds_iter:
        #     print(sr_rel_data)
        #     print(tg_rel_data)
        #     exit()

        gcn = GCN(self.cuda, cgc, self.num_layer, self.embedding_dim, self.dropout_rate)
        if self.cuda:
            gcn.cuda()
        # for name, param in gcn.named_parameters():
        #     print(name, param.requires_grad)
        optimizer = torch.optim.SGD(gcn.parameters(), lr=0.01)
        criterion_entity = GCNAlignLoss(self.entity_gamma)
        criterion_relation = GCNAlignLoss(self.relation_gamma, re_scale=self.beta)
        if self.cuda:
            criterion_entity.cuda()
            criterion_relation.cuda()
        for epoch in range(self.num_epoch):
            relation_seeds_iter = iter(relation_seeds)
            for i_batch, batch in enumerate(entity_seeds):
                optimizer.zero_grad()
                sr_data, tg_data = batch
                if self.cuda:
                    sr_data, tg_data = sr_data.cuda(), tg_data.cuda()
                try:
                    sr_rel_data, tg_rel_data = next(relation_seeds_iter)
                except StopIteration:
                    relation_seeds_iter = iter(relation_seeds)
                    sr_rel_data, tg_rel_data = next(relation_seeds_iter)
                if self.cuda:
                    sr_rel_data, tg_rel_data = sr_rel_data.cuda(), tg_rel_data.cuda()
                repre_e_sr, repre_e_tg, repre_r_sr, repre_r_tg = gcn(sr_data, tg_data, sr_rel_data, tg_rel_data)

                entity_loss = criterion_entity(repre_e_sr, repre_e_tg)
                relation_loss = criterion_relation(repre_r_sr, repre_r_tg)

                loss = sum([entity_loss, relation_loss])

                print('loss: ', loss)
                loss.backward()
                optimizer.step()
                print_time_info('Epoch: %d; Batch: %d; loss = %.2f'%(epoch, i_batch, float(loss)))

        def loop():
            # todo: finish it
            train_seeds_ratio = 0.3
            directory = bin_dir / 'dbp15k'
            language_pair_dirs = list(directory.glob('*_en'))
            for local_directory in language_pair_dirs:
                cgc = CrossGraphCompletion(local_directory, train_seeds_ratio)
                cgc.init()
                cgc.save(local_directory / 'running_temp')


if __name__ == '__main__':
    # CUDA_LAUNCH_BLOCKING=1
    import sys
    directory = bin_dir / 'dbp15k'
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    config = Config(directory)
    config.init()
    config.train()