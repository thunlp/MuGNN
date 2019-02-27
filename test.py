import torch


a = torch.tensor([i for i in range(10)], dtype=torch.float, requires_grad=True)

print(a.requires_grad)
a = a.detach()
print(a.requires_grad)