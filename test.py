import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn.functional as F


def cosine_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.mm(a, b.transpose(0, 1))


def relaton_weighting(a, b):
    reverse = False
    
    if a.size()[0] > b.size()[0]:
        a, b = b, a
        reverse = True
    
    pad_len = b.size()[0] > a.size()[0]
    if pad_len > 0:
        a = F.pad(a, (0, 0, 0, pad_len))

    sim = cosine_similarity_nbyn(a, b)
    rows, cols = linear_sum_assignment(-sim.numpy())
    # print(rows[cols.argsort()], 'cds')
    rows = torch.from_numpy(rows)
    cols = torch.from_numpy(cols)
    r_sim_sr = torch.gather(sim, -1, cols.view(-1, 1)).squeeze(1)
    cols, cols_index = cols.sort()
    rows = rows[cols_index]
    r_sim_tg = torch.gather(sim.t, -1, rows.view(-1, 1)).squeeze(1)
    
    if pad_len > 0:
        r_sim_sr = r_sim_sr[:-pad_len]
    if reverse:
        r_sim_sr, r_sim_tg = r_sim_tg, r_sim_sr
    return r_sim_sr, r_sim_tg

a = torch.tensor([[1.0, 2.0], [10, 1]])
b = torch.tensor([[3.0, 4.0], [2.0, 3.0], [1.0, 2.0]])

r_sim_sr, r_sim_tg = relaton_weighting(a, b)

print(r_sim_sr)
print(r_sim_tg)