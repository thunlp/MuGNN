from graph_completion.CrossGraphCompletion import CrossGraphCompletion
from tools.print_time_info import print_time_info

def train():
    '''
    先直接写，根据写出来的样子在进行抽象。
    '''
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    from graph_completion.GCN import GCN
    from graph_completion.CrossAdjacencyMatrix import CrossAdjacencyMatrix

    from project_path import bin_dir
    train_seeds_ratio = 0.3
    directory = bin_dir / 'dbp15k'
    language_pair_dirs = list(directory.glob('*_en'))

    gpu_num = 1
    num_layer = 3
    num_epoch = 1000
    embedding_dim = 100
    dropout_rate = 0.5

    if False:
        cgc = CrossGraphCompletion(language_pair_dirs[0], train_seeds_ratio)
        cgc.init()
        # cgc.save(language_pair_dirs[0] / 'running_temp')
    else:
        cgc = CrossGraphCompletion.restore(language_pair_dirs[0] / 'running_temp')

    cam = CrossAdjacencyMatrix(embedding_dim, cgc)
    cam = cam.to('cuda:%d' % gpu_num)
    adjacency_matrix_sr, adjacency_matrix_tg = cam()

    id2gcn = {}
    for i in range(num_layer):
        id2gcn[i] = GCN(embedding_dim, embedding_dim, dropout_rate)
        id2gcn[i].to('cuda:%d' % gpu_num)

    for epoch in range(num_epoch):
        graph_embeding_sr, graph_embeding_tg = cam.entity_embedding_sr.weight, cam.entity_embedding_tg.weight
        for i in range(num_layer):
            graph_embeding_sr = id2gcn[i](graph_embeding_sr, adjacency_matrix_sr)
            graph_embeding_tg = id2gcn[i](graph_embeding_tg, adjacency_matrix_tg)
            print_time_info('graph_embedding_sr size:', graph_embeding_sr.size())
            print_time_info('graph_embedding_tg size:', graph_embeding_tg.size())

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