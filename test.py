import torch
from scipy.optimize import linear_sum_assignment
import numpy as np


# def cosine_similarity(a, b):
#     '''
#     a shape: [num_item_1, embedding_dim]
#     b shape: [num_item_2, embedding_dim]
#     return sim_matrix: [num_item_1, num_item_2]
#     '''
#     a = a / a.norm(dim=-1, keepdim=True)
#     b = b / b.norm(dim=-1, keepdim=True)
#     return  torch.mm(a, b.transpose(0, 1))

# # print(a.mm(b.transpose(0,1)))

# a = torch.tensor([[1.0, 2.0], [10, 1]])
# a[1,1] = 10 # torch.tensor([1])
# print(a)
# exit()
# b = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])



# sim = cosine_similarity(a ,b)
# print(sim)


# rows, cols = linear_sum_assignment(-sim.numpy())
# rows =  torch.from_numpy(rows)
# cols = torch.from_numpy(cols).view(-1, 1)
# print(rows)
# print(cols)
# sim = torch.index_select(sim, 0, rows)
# sim = torch.gather(sim, -1, cols)


# print(torch.sum(sim))
# # print(a.size())
# # print(b.size())

# # cos = torch.nn.CosineSimilarity(dim=0)
# # sim = cos(a, b)
# # print(sim)