import torch, random
import numpy as np
import multiprocessing

from utils.tools import print_time_info


def set_random_seed(seed_value=999):
    torch.manual_seed(seed_value)  # cpu  vars
    np.random.seed(seed_value)  # cpu vars
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)


def str2int4triples(triples):
    return [(int(head), int(tail), int(relation)) for head, tail, relation in triples]


def get_hits(sim, top_k=(1, 10, 50, 100)):
    if isinstance(sim, np.ndarray):
        sim = torch.from_numpy(sim)
    top_lr, mr_lr, mrr_lr = topk(sim, top_k)
    top_rl, mr_rl, mrr_rl = topk(sim.t(), top_k)
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
    return top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl


def topk(sim, top_k=(1, 10, 50, 100)):
    # Sim shape = [num_ent, num_ent]
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert sim.shape[0] == sim.shape[1]
    test_num = sim.shape[0]
    batched = True
    if sim.shape[0] * sim.shape[1] < 20000 * 128:
        batched = False
        sim = sim.to(device)

    def _opti_topk(sim):
        sorted_arg = torch.argsort(sim)
        true_pos = torch.arange(test_num, device=device).reshape((-1, 1))
        locate = sorted_arg - true_pos
        del sorted_arg, true_pos
        locate = torch.nonzero(locate == 0)
        cols = locate[:, 1]  # Cols are ranks
        cols = cols.float()
        top_x = [0.0] * len(top_k)
        for i, k in enumerate(top_k):
            top_x[i] = float(torch.sum(cols < k)) / test_num * 100
        mr = float(torch.sum(cols + 1)) / test_num
        mrr = float(torch.sum(1.0 / (cols + 1))) / test_num * 100
        return top_x, mr, mrr

    def _opti_topk_batched(sim):
        mr = 0.0
        mrr = 0.0
        top_x = [0.0] * len(top_k)
        batch_size = 1024
        for i in range(0, test_num, batch_size):
            batch_sim = sim[i:i + batch_size].to(device)
            sorted_arg = torch.argsort(batch_sim)
            true_pos = torch.arange(
                batch_sim.shape[0]).reshape((-1, 1)).to(device) + i
            locate = sorted_arg - true_pos
            del sorted_arg, true_pos
            locate = torch.nonzero(locate == 0,)
            cols = locate[:, 1]  # Cols are ranks
            cols = cols.float()
            mr += float(torch.sum(cols + 1))
            mrr += float(torch.sum(1.0 / (cols + 1)))
            for i, k in enumerate(top_k):
                top_x[i] += float(torch.sum(cols < k))
        mr = mr / test_num
        mrr = mrr / test_num * 100
        for i in range(len(top_x)):
            top_x[i] = top_x[i] / test_num * 100
        return top_x, mr, mrr

    with torch.no_grad():
        if not batched:
            return _opti_topk(sim)
        return _opti_topk_batched(sim)


def get_nearest_neighbor(sim, indices, nega_sample_num=25):
    # Sim do not have to be a square matrix
    # Let us assume sim is a numpy array
    if isinstance(sim, np.ndarray):
        sim = torch.from_numpy(sim)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sim = sim.to(device)
    assert indices.dtype == 'int'
    ranks = torch.argsort(sim, dim=1)
    ranks = ranks[:, 1:nega_sample_num + 1].cpu().numpy()
    nega_samples = {}
    for i, row in enumerate(ranks):
        i = indices[i]
        row = [indices[j] for j in row]
        nega_samples[i] = row
    return nega_samples

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
