from graph_completion import graph_completion


if __name__ == '__main__':
    # graph_completion.main()
    import torch
    a = torch.Tensor([15, 28])
    a = a.expand([1, 2]) #.repeat([3,1])
    print(a)
    exit()
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    cost = np.array([[3, 2, 1, 0], [1, 3, 2, 0], [2, 1, 3, 0]])
    row_ind, col_ind = linear_sum_assignment(cost)
    print(row_ind, col_ind)
