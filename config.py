import torch, os
from project_path import bin_dir
from graph_completion.GCN import GCN
from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from graph_completion.AlignmentDataset import AliagnmentDataset
from torch.utils.data import DataLoader
from graph_completion.functions import gcn_align_loss
from tools.print_time_info import print_time_info

class Config(object):

    def __init__(self, directory):
        self.train_seeds_ratio = 0.3
        self.directory = directory

        self.entity_gamma = 3.0  # margin for entity loss
        self.relation_gamma = 3.0  # margin for relation loss
        self.beta = 0.3  # ratio of relation loss
        self.num_layer = 3
        self.num_epoch = 1000
        self.embedding_dim = 100
        self.dropout_rate = 0.5
        self.nega_sample_num = 24  # number of negative samples for each positive one
        self.batch_size = 64
        self.shuffle = True
        self.num_workers = 2  # for the data_loader

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

        gcn = GCN(cgc, self.num_layer, self.embedding_dim, self.dropout_rate)
        gcn.cuda()
        # for name, param in gcn.named_parameters():
        #     print(name, param.requires_grad)
        optimizer = torch.optim.SGD(gcn.parameters(), lr=0.01)
        for epoch in range(self.num_epoch):
            relation_seeds_iter = iter(relation_seeds)
            for i_batch, batch in enumerate(entity_seeds):
                optimizer.zero_grad()
                sr_data, tg_data = batch
                try:
                    sr_rel_data, tg_rel_data = next(relation_seeds_iter)
                except StopIteration:
                    relation_seeds_iter = iter(relation_seeds)
                    sr_rel_data, tg_rel_data = next(relation_seeds_iter)
                repre_e_sr, repre_e_tg, repre_r_sr, repre_r_tg = gcn(sr_data, tg_data, sr_rel_data, tg_rel_data)
                entity_loss = gcn_align_loss(repre_e_sr, repre_e_tg, self.entity_gamma)
                relation_loss = gcn_align_loss(repre_r_sr, repre_r_tg, self.relation_gamma)
                loss = entity_loss + self.beta * relation_loss
                loss.backward()
                optimizer.step()
                print_time_info('Epoch: %d; Batch: %d; loss = %.2f'%(epoch, i_batch, float(loss)))

def main():
    from project_path import bin_dir
    train_seeds_ratio = 0.3
    directory = bin_dir / 'dbp15k'
    language_pair_dirs = list(directory.glob('*_en'))
    for local_directory in language_pair_dirs:
        cgc = CrossGraphCompletion(local_directory, train_seeds_ratio)
        cgc.init()
        cgc.save(local_directory / 'running_temp')
        # train()


if __name__ == '__main__':
    directory = bin_dir / 'dbp15k'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = Config(directory)
    config.init()
    config.train()