import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # format transform
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



a = torch.tensor([[0,1,4], [1,1,0]])
b = torch.tensor([1, 2, 3], dtype=torch.float)

c = torch.sparse.FloatTensor(a, b)


def normailize(adj):
    rowsum = torch.sum(adj, 1)
    print(rowsum)
    d_inv_sqrt = torch.pow(rowsum, 0.5)
    print(d_inv_sqrt)
normailize((c))