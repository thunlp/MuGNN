from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from graph_completion.AlignmentDataset import AliagnmentDataset
from torch.utils.data import DataLoader
from graph_completion.functions import gcn_align_loss
from tools.print_time_info import print_time_info


def train():
    import torch
    from project_path import bin_dir
    from graph_completion.GCN import GCN
    train_seeds_ratio = 0.3
    directory = bin_dir / 'dbp15k'
    language_pair_dirs = list(directory.glob('*_en'))

    gpu_num = 1
    entity_gamma = 3.0 # margin for entity loss
    relation_gamma = 3.0 # margin for relation loss
    beta = 0.3 # ratio of relation loss
    num_layer = 3
    num_epoch = 1000
    embedding_dim = 100
    dropout_rate = 0.5
    nega_sample_num = 24  # number of negative samples for each positive one
    batch_size = 64
    shuffle = True
    num_workers = 2  # for the data_loader

    if False:
        cgc = CrossGraphCompletion(language_pair_dirs[0], train_seeds_ratio)
        cgc.init()
        # cgc.save(language_pair_dirs[0] / 'running_temp')
    else:
        cgc = CrossGraphCompletion.restore(language_pair_dirs[0] / 'running_temp')

    entity_seeds = AliagnmentDataset(cgc.entity_seeds, nega_sample_num, len(cgc.id2entity_sr), len(cgc.id2entity_tg))
    entity_seeds = DataLoader(entity_seeds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    relation_seeds = AliagnmentDataset(cgc.relation_seeds, nega_sample_num, len(cgc.id2relation_sr),
                                       len(cgc.id2relation_tg))
    relation_seeds = DataLoader(relation_seeds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    gcn = GCN(cgc, num_layer, embedding_dim, dropout_rate)
    gcn.to('cuda: %d' % gpu_num)
    optimizer = torch.optim.SGD(gcn.parameters(), lr=0.01)
    for epoch in range(num_epoch):
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
            entity_loss = gcn_align_loss(repre_e_sr, repre_e_tg, entity_gamma)
            relation_loss = gcn_align_loss(repre_r_sr, repre_r_tg, relation_gamma)
            loss = entity_loss + beta * relation_loss
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
    train()
