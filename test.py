import torch
import torch.nn.functional as F
import numpy as np

a = [[[i + k - j + 3 for k in range(3)] for j in range(2)] for i in range(4)]
b = [[[i * k + j for k in range(3)] for j in range(2)] for i in range(4)]
print(a)
print(b)
a = torch.tensor(a)
b = torch.tensor(b)

print(a-b)
