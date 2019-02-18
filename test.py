import torch
import scipy.sparse as sp


a = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=torch.float)
b = torch.tensor([[2, 2, 0], [3, 1, 2], [12, 7, 8]], dtype=torch.float)
c = torch.tensor([2, -1, -1], dtype=torch.float)

def cal_loss(repre_sr, repre_tg, labels):
    print(torch.abs(repre_sr - repre_tg))
    print(torch.sum(torch.abs(repre_sr - repre_tg), dim=1))
    loss = torch.sum(torch.abs(repre_sr - repre_tg), dim=1) * labels
    print(loss)
    return loss

loss = cal_loss(a, b, c)