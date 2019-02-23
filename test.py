import torch
import numpy as np
import multiprocessing
from graph_completion.torch_functions import torch_l2distance

a = torch.randn((120, 2), dtype=torch.float)
b = torch.randn((120, 2), dtype=torch.float)
sim = torch_l2distance(a, b)
print(sim)

def multiprocess_topk(sim, top_k=(1, 10, 50, 100)):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    n_p = 4

    # top_x = [0] * len(top_k)
    def top_get(sim, top_k, id, return_dict, chunk_size):
        top_x = [0] * len(top_k)
        s = chunk_size * id
        for i in range(sim.shape[0]):
            rank = sim[i, :].argsort()
            rank_index = np.where(rank == i + s)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_x[j] += 1
        return_dict[id] = top_x

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
    for i, top_kk in return_dict.items():
        assert len(top_x) == len(top_kk)
        for j in range(len(top_kk)):
            top_x[j] += top_kk[j]
    return top_x


def top_get(sim, top_k=(1, 10, 50, 100)):
    top_x = [0] * len(top_k)
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_x[j] += 1
    return top_x




top_x = multiprocess_topk(sim.numpy())
top_x2 = top_get(sim.numpy())
print(top_x)
print(top_x2)
