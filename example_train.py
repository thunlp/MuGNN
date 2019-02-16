import math
from graph_completion.CrossGraphCompletion import CrossGraphCompletion


def get_relation2idf(triples, relations):
    triples_num = len(triples)
    relation2num = {relation: 0 for relation in relations}
    for triple in triples:
        relation2num[triple[2]] += 1
    for relation, num in relation2num.items():
        assert num > 0
        relation2num[relation] = math.log10(triples_num/num)
    return relation2num

def get_triple2conf(triples, new_triple2conf):
    triple2conf = {}
    for triple in triples:
        if triple in new_triple2conf:
            triple2conf[triple] = new_triple2conf[triple]
        else:
            triple2conf[triple] = 1.0
    return triple2conf

def get_entity_seeds2relation_matrix(entity_seeds, triples_sr, triples_tg):
    '''
    目前只考虑entity作为head
    output shape = [entity_seeds_num, max_relation_num] * 2
    '''
    
    entities_sr, entities_tg = list(zip(*entity_seeds))
    
    def _get(entities, triples):
        entity2id = {entity: i for i, entity in enumerate(entities)}
        id2relation = [set() for _ in range(len(entities))]
        for head, tail, relation in triples:
            if head in entity2id:
                id2relation[entity2id[head]].add(relation)
        id2relation = [list(relations) for relations in id2relation]
        id2length = [len(relations) for relations in id2relation]
        return id2relation, id2length
    
    id2relation_sr, id2length_sr = _get(entities_sr, triples_sr)
    id2relation_tg, id2length_tg = _get(entities_tg, triples_tg)
    max_len = max(*(id2length_sr + id2length_tg))
    id2mask_sr = [[False if i < length else True for i in range(max_len)] for length in id2length_sr]
    id2mask_tg = [[False if i < length else True for i in range(max_len)] for length in id2length_tg]
    id2relation_sr = [relations + [0] * (max_len - len(relations)) for relations in id2relation_sr]
    id2relation_tg = [relations + [0] * (max_len - len(relations)) for relations in id2relation_tg]
    return id2relation_sr, id2relation_tg, id2mask_sr, id2mask_tg

def get_adjacency_matrix(triple2conf, relation_idf, entity_embedding, relation_embedding):
    # for triple in triple2conf.items():
    pass


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
    embedding_dim = 100


    cgc = CrossGraphCompletion(language_pair_dirs[0], train_seeds_ratio)
    cgc.init()

    triples_sr = cgc.triples_sr
    triples_tg = cgc.triples_tg

    relation2idf_sr = get_relation2idf(triples_sr, cgc.id2relation_sr.keys())
    relation2idf_tg = get_relation2idf(triples_tg, cgc.id2relation_tg.keys())
    triple2conf_sr = get_triple2conf(triples_sr, cgc.new_triple_confs_sr)
    triple2conf_tg = get_triple2conf(triples_tg, cgc.new_triple_confs_tg)


    entity_seeds_sr, entity_seeds_tg = list(zip(*cgc.entity_seeds))

    
    

    entity_embedding_sr = nn.Embedding(len(cgc.id2entity_sr), embedding_dim)
    relation_embedding_sr = nn.Embedding(len(cgc.id2relation_sr), embedding_dim)
    entity_embedding_tg = nn.Embedding(len(cgc.id2entity_tg), embedding_dim)
    relation_embedding_tg = nn.Embedding(len(cgc.id2relation_tg), embedding_dim)
    nn.init.xavier_uniform_(entity_embedding_sr)
    nn.init.xavier_uniform_(relation_embedding_sr)
    nn.init.xavier_uniform_(entity_embedding_tg)
    nn.init.xavier_uniform_(relation_embedding_tg)


    # shape = [num_entity_seeds, max_relation_num]
    seed2relation_sr, seed2relation_tg, seed2mask_sr, seed2mask_tg = get_entity_seeds2relation_matrix(cgc.entity_seeds, triples_sr, triples_tg)

    seed2relation_sr = torch.tensor(seed2relation_sr, requires_grad=False)
    seed2relation_tg = torch.tensor(seed2relation_tg, requires_grad=False)
    seed2mask_sr = torch.tensor(seed2mask_sr, requires_grad=False)
    seed2mask_tg = torch.tensor(seed2mask_tg, requires_grad=False)

    
    # shape = [num_entity_seeds, max_num_relation, embedding_dim]
    seed2relation_embedding_sr = relation_embedding_sr(seed2relation_sr)
    seed2relation_embedding_tg = relation_embedding_tg(seed2relation_tg)
    seed2relation_embedding_sr[seed2mask_sr] = 0
    seed2relation_embedding_tg[seed2mask_tg] = 0
    assert seed2relation_embedding_sr.size()[-1] == embedding_dim, print(seed2relation_embedding_sr.size())

    # [num_entity, num_relation, num_relation]
    # [num_entity, num_relation]
    # score = [num_entity]

    # ... = [num_entity, num_max_relation]  mask = [num_entity]
    # ... = [num_entity, num_max_relation, embedding_dim]
    # ... = [num_entity, num_max_relation, num_max_relation]

    # embedding lookup




    id2gcn = {}
    num_layer = 3
    num_epoch = 1000
    embedding_dim = 100


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
