from graph_completion.CrossGraphCompletion import CrossGraphCompletion


def train():
    '''
    先直接写，根据写出来的样子在进行抽象。
    '''
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from graph_completion.GCN import GCN
    
    from project_path import bin_dir
    train_seeds_ratio = 0.3
    directory = bin_dir / 'dbp15k'
    language_pair_dirs = list(directory.glob('*_en'))
    CrossGraphCompletion(language_pair_dirs[0], train_seeds_ratio)

    id2entity_sr = {}
    id2relation_sr = {}

    id2entity_tg = {}
    id2relation_tg = {}


    triples_sr = [('triples')]
    triples_tg = [('triples')]

    id2gcn = {}
    num_layer = 3
    num_epoch = 3
    embedding_dim = 50

    entity_embedding_sr = nn.Embedding(len(id2entity_sr), embedding_dim)
    relation_embedding_sr = nn.Embedding(len(id2relation_sr), embedding_dim)
    entity_embedding_tg = nn.Embedding(len(id2entity_tg), embedding_dim)
    relation_embedding_tg = nn.Embedding(len(id2relation_tg), embedding_dim)

    nn.init.xavier_uniform_(entity_embedding_sr)
    nn.init.xavier_uniform_(relation_embedding_sr)
    nn.init.xavier_uniform_(entity_embedding_tg)
    nn.init.xavier_uniform_(relation_embedding_tg)

    for i in range(num_layer):
        id2gcn[i] = GCN(10, 10, 0.5)

    for epoch in range(num_epoch):
        for i in range(num_layer):
            layer = id2gcn[i]



def main():
    from project_path import bin_dir
    train_seeds_ratio = 0.3
    directory = bin_dir / 'dbp15k'
    language_pair_dirs = list(directory.glob('*_en'))
    for local_directory in language_pair_dirs:
        cgc = CrossGraphCompletion(local_directory, train_seeds_ratio)
        cgc.init()
        # train()

if __name__ == '__main__':
    main()
    exit()
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    cost = np.array([[3, 2, 1, 0], [1, 3, 2, 0], [2, 1, 3, 0]])
    row_ind, col_ind = linear_sum_assignment(cost)
    print(row_ind, col_ind)
