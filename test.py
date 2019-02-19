import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch import sparse


pos = torch.tensor([[0, 1, 0], [1, 0, 1]])
value  = torch.tensor([3, 1, 2])
scp = torch.sparse_coo_tensor(pos, value)
print(scp.to_dense())
exit()
di_degree = np.sum(a, axis=1)
di_degree = np.power(di_degree, -0.5)
degree = np.diag(di_degree)
# a = np.asarray([[j+1 if j == i else 0 for j in range(3)] for i in range(3)])
# degree = np.power(degree, -0.5)
degree[degree==np.inf] = 0
print(degree)
result = np.matmul(np.matmul(degree, a), degree)
print(result)
result = np.multiply(np.multiply(di_degree, a).transpose(), di_degree).transpose()
print(result)
exit()
def normalize_adj_torch(adj):
    row_sum = torch.sparse.sum(adj, 1)
    d_inv_sqrt = torch.pow(row_sum, -1)
    d_inv_sqrt = d_inv_sqrt.to_dense().view(-1, 1)
    adj = adj.to_dense()
    result = (adj * d_inv_sqrt).t() #* d_inv_sqrt) # the result of gcn code
    # del d_inv_sqrt, adj
    return result.t() #.to_sparse()


a = np.asarray([[1, 1 ,0], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
aa = sp.coo_matrix(a)
row = np.asarray(aa.row, dtype=np.int64)
col = np.asarray(aa.col, dtype=np.int64)
row = torch.from_numpy(row).view(1, -1)
col = torch.from_numpy(col).view(1, -1)
pos = torch.cat((row, col), dim=0)
value = aa.data
value = torch.from_numpy(value)
sparse_a = torch.sparse_coo_tensor(pos, value) #.to_dense()
a = normalize_adj_torch(sparse_a)
print(a)