import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn.functional as F


a = torch.tensor([[0,1,4], [1,1,0]], dtype=torch.float)
b = torch.tensor([1, 2, 3], dtype=torch.float)
print(a + b)
# c = torch.sparse.FloatTensor(a, b)
# c = c * c * c + c*0.5
# print(c)