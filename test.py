import torch

a = torch.Tensor([[1, 2, 0], [0,1,3]])
print(a.nonzero().t())