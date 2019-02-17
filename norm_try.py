import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp

def normalize_adj():
    """Symmetrically normalize adjacency matrix."""
    # format transform
    adj = ([1, 2, 3, 1], ([0, 1, 3, 0], [1, 1, 0, 0]))
    adj = sp.coo_matrix(adj, shape=(4,4))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # print(type(d_inv_sqrt))
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    print(d_mat_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return adj.dot(d_mat_inv_sqrt).dot(d_mat_inv_sqrt).tocoo()


a = torch.tensor([[0, 1, 3, 0], [1, 1, 0, 0]])
b = torch.tensor([1, 2, 3, 1], dtype=torch.float)

c = torch.sparse.FloatTensor(a, b, size=torch.Size([4,4]))


def normailize(adj):
    rowsum = torch.sparse.sum(adj, 1)
    print(adj)
    d_inv_sqrt = torch.pow(rowsum, -0.5).to_dense()
    # d_inv_sqrt =
    print(d_inv_sqrt)
    x =  torch.dot(d_inv_sqrt, adj)
    print(x)
    return
    # adj = adj * d_inv_sqrt
    # print(adj)
normailize(c)
# result = normalize_adj().toarray()
# print(result)