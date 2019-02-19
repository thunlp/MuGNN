import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch import sparse

# a = torch.tensor([[1,2,3], [2, 3, 4], [3, 4, 5]])
# b = torch.tensor([1, 2, 3])
# print(a*b)
# exit()


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     # format transform
#     # adj = sp.coo_matrix(adj)
#
#     # norm
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt).todense()
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).transpose()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # format transform
    adj = sp.coo_matrix(adj)

    # norm
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


    # return (rel.t() * d_mat_inv_sqrt).t()
    # print(d_inv_sqrt)

a = np.asarray([[1, 2 ,3], [0, 2, 0], [3, 2, 1]], dtype=np.float)
aa = sp.coo_matrix(a)
row = np.asarray(aa.row, dtype=np.float)
col = np.asarray(aa.col, dtype=np.float)
row = torch.from_numpy(row).view(1, -1)
col = torch.from_numpy(col).view(1, -1)
pos = torch.cat((row, col), dim=0)
value = aa.data
value = torch.from_numpy(value)
sparse_a = torch.sparse_coo_tensor(pos, value)
# print(sparse_a)
aa = normalize_adj_torch(sparse_a)
print(aa.to_dense())
# print(aa)
a = normalize_adj(a)
print(a.todense())
print(type(a))
# print(a)
# print(a)