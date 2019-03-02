import torch
from torch import sparse


a = torch.tensor([[1,2,3], [0, 1, 2]])
c = torch.tensor([0,1,0])

t = torch.sparse_coo_tensor(a, c).coalesce()

one = torch.ones_like(t)
print(one + t)