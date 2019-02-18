from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from graph_completion.AlignmentDataset import AliagnmentDataset
from torch.utils.data import DataLoader
from tools.print_time_info import print_time_info

def train():
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    from project_path import bin_dir
    from graph_completion.GCN import GCN
    from graph_completion.CrossAdjacencyMatrix import CrossAdjacencyMatrix
    train_seeds_ratio = 0.3
    directory = bin_dir / 'dbp15k'
    language_pair_dirs = list(directory.glob('*_en'))

    gpu_num = 1
    num_layer = 3
    num_epoch = 1000
    embedding_dim = 100
    dropout_rate = 0.5
    nega_sample_num = 24 # number of negative samples for each positive one
    batch_size = 64
    shuffle = True
    num_workers = 2 # for the data_loader
    if False:
        cgc = CrossGraphCompletion(language_pair_dirs[0], train_seeds_ratio)
        cgc.init()
        # cgc.save(language_pair_dirs[0] / 'running_temp')
    else:
        cgc = CrossGraphCompletion.restore(language_pair_dirs[0] / 'running_temp')

    entity_seeds = AliagnmentDataset(cgc.entity_seeds, nega_sample_num, len(cgc.id2entity_sr), len(cgc.id2entity_tg))
    entity_seeds = DataLoader(entity_seeds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    relation_seeds = AliagnmentDataset(cgc.relation_seeds, nega_sample_num, len(cgc.id2relation_sr), len(cgc.id2relation_tg))
    relation_seeds = DataLoader(relation_seeds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    cam = CrossAdjacencyMatrix(embedding_dim, cgc)
    cam = cam.to('cuda:%d' % gpu_num)
    id2gcn = {}
    for i in range(num_layer):
        id2gcn[i] = GCN(embedding_dim, embedding_dim, dropout_rate)
        id2gcn[i].to('cuda:%d' % gpu_num)

    for epoch in range(num_epoch):
        adjacency_matrix_sr, adjacency_matrix_tg = cam()
        rel_embedding_sr, rel_embedding_tg = cam.relation_embedding_sr, cam.relation_embedding_tg
        graph_embedding_sr, graph_embedding_tg = cam.entity_embedding_sr.weight, cam.entity_embedding_tg.weight
        for i in range(num_layer):
            graph_embedding_sr = id2gcn[i](graph_embedding_sr, adjacency_matrix_sr)
            graph_embedding_tg = id2gcn[i](graph_embedding_tg, adjacency_matrix_tg)
        
        relation_seeds_iter = iter(relation_seeds)
        for i_batch, batch in enumerate(entity_seeds):
            entity_sr, entity_tg, entity_labels = batch
            try:
                relaiton_sr, relation_tg, rel_labels = next(relation_seeds_iter)
            except StopIteration:
                relation_seeds_iter = iter(relation_seeds)
                relaiton_sr, relation_tg, rel_labels = next(relation_seeds_iter)

            repre_e_sr = F.embedding(entity_sr, graph_embedding_sr)
            repre_e_tg = F.embedding(entity_tg, graph_embedding_tg)
            repre_r_sr = rel_embedding_sr(relaiton_sr)
            repre_r_tg = rel_embedding_tg(relation_tg)



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
    # main()
    train()