import numpy as np
from scipy import spatial as spatial

from tools.print_time_info import print_time_info


def str2int4triples(triples):
    return [(int(head), int(tail), int(relation)) for head, tail, relation in triples]


def get_hits(sim, top_k=(1, 10, 50, 100)):
    test_num = sim.shape[0]
    # sim = spatial.distance.cdist(Lvec, Rvec, metric='minkowski', p=2)
    # sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    def top_get(sim):
        top_x = [0] * len(top_k)
        for i in range(sim.shape[0]):
            rank = sim[i, :].argsort()
            rank_index = np.where(rank == i)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_x[j] += 1
        return top_x
    top_lr = top_get(sim)
    top_rl = top_get(sim.T)
    print_time_info('For each source:')
    for i in range(len(top_lr)):
        print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / test_num * 100))
    print('')
    print_time_info('For each target:')
    for i in range(len(top_rl)):
        print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / test_num * 100))
    # return Hits@10
    return (top_lr[1] / test_num * 100, top_rl[1] / test_num * 100)
