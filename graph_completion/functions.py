import numpy as np
import multiprocessing
from scipy import spatial as spatial

from tools.print_time_info import print_time_info


def str2int4triples(triples):
    return [(int(head), int(tail), int(relation)) for head, tail, relation in triples]


def get_hits(sim, top_k=(1, 10, 50, 100)):
    test_num = sim.shape[0]

    # sim = spatial.distance.cdist(Lvec, Rvec, metric='minkowski', p=2)
    # sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    def top_get(sim, top_k):
        top_x = [0] * len(top_k)
        for i in range(sim.shape[0]):
            rank = sim[i, :].argsort()
            rank_index = np.where(rank == i)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_x[j] += 1
        return top_x

    top_lr, mr_lr, mrr_lr = multiprocess_topk(sim, top_k)
    top_rl, mr_rl, mrr_rl = multiprocess_topk(sim.T, top_k)

    print_time_info('For each source:')
    print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_lr, mrr_lr))
    for i in range(len(top_lr)):
        print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_lr[i]))
    print('')
    print_time_info('For each target:')
    print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_rl, mrr_rl))
    for i in range(len(top_rl)):
        print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_rl[i]))
    # return Hits@10
    return top_lr, top_rl


def multiprocess_topk(sim, top_k=(1, 10, 50, 100)):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    n_p = 4

    # top_x = [0] * len(top_k)
    def top_get(sim, top_k, id, return_dict, chunk_size):
        top_x = [0] * len(top_k)
        s = chunk_size * id
        rank_sum = 0.0
        r_rank_sum = 0.0
        for i in range(sim.shape[0]):
            rank = sim[i, :].argsort()
            rank_index = np.where(rank == i + s)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_x[j] += 1
            rank_sum += rank_index + 1
            r_rank_sum += 1.0 / (rank_index + 1)
        return_dict[id] = (top_x, rank_sum, r_rank_sum)

    test_num = sim.shape[0]
    chunk = test_num // n_p + 1
    sim_chunked = [sim[i:chunk + i, :] for i in range(0, test_num, chunk)]
    assert len(sim_chunked) == n_p
    pool = []
    for i, sim_chunk in enumerate(sim_chunked):
        p = multiprocessing.Process(target=top_get, args=(sim_chunk, top_k, i, return_dict, chunk))
        pool.append(p)
        p.start()
    for p in pool:
        p.join()
    top_x = [0] * len(top_k)
    rank_sum = 0.0
    r_rank_sum = 0.0
    for i, eval_result in return_dict.items():
        top_kk, rank_sum_local, r_rank_sum_local = eval_result
        rank_sum += rank_sum_local
        r_rank_sum += r_rank_sum_local
        assert len(top_x) == len(top_kk)
        for j in range(len(top_kk)):
            top_x[j] += top_kk[j]
    for i in range(len(top_x)):
        top_x[i] = top_x[i] / test_num * 100
    mr = rank_sum / test_num
    mrr = r_rank_sum / test_num
    return top_x, mr, mrr


def multi_process_get_nearest_neighbor(sim, ranks, nega_sample_num=25):
    assert len(sim) == len(ranks)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    n_p = 4

    # top_x = [0] * len(top_k)
    def nega_get(sim, c_ranks, ranks, nega_sample_num, return_dict):
        for i in range(len(sim)):
            rank = sim[i, :].argsort()
            nega_sample = rank[1:nega_sample_num + 1]
            nega_sample = [ranks[sample] for sample in nega_sample]
            assert len(nega_sample) == nega_sample_num
            return_dict[c_ranks[i]] = nega_sample

    test_num = len(sim)
    chunk = test_num // n_p + 1
    sim_chunked = [sim[i:chunk + i, :] for i in range(0, test_num, chunk)]
    rank_chunked = [ranks[i:chunk + i] for i in range(0, test_num, chunk)]
    assert len(sim_chunked) == n_p
    pool = []
    for i, sim_chunk in enumerate(sim_chunked):
        p = multiprocessing.Process(target=nega_get,
                                    args=(sim_chunk, rank_chunked[i], ranks, nega_sample_num, return_dict))
        pool.append(p)
        p.start()
    for p in pool:
        p.join()
    return dict(return_dict)
