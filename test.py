import torch


b = [torch.tensor([[i+j+k for k in range(4)] for i in range(j, j+2)], dtype=torch.float) for j in range(3)]
print(b)
print(b[0].size())
b = [bb.unsqueeze(-1) for bb in b]
print(b[0].size())
c = torch.cat(b, dim=-1)
print(c.size())
d = torch.mean(c, dim=-1)
print(d.size())
print(d)