import torch
import torch.nn.functional as F

a = torch.tensor([[1, 2, 3], [2, 10, 17], [100, 100000, -1]], dtype=torch.float)
a = a.expand(1, 1, 3, 3)
print(a.size())

print(col_pool)
print(row_pool)
exit()

import torch
import torch.nn.functional as F

from graph_completion.functions import GCNAlignLoss

gal = GCNAlignLoss(2)
# repre_sr shape: [batch_size, 1 + nega_sample_num, embedding_dim]

a = torch.tensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], ], dtype=torch.float)
b = torch.tensor([[[2, 2, 0], [3, 1, 2], [10, 2, 12]], ], dtype=torch.float)

loss = gal(a, b)

print(loss)
exit()
def cal_loss(repre_sr, repre_tg, gamma):
    '''
    repre shape: [batch_size, 1+nega_sample_num, embedding_dim]
    '''
    score = torch.sum(torch.abs(repre_sr - repre_tg), dim=-1)
    pos_score = score[:, :1]
    nega_score = score[:, 1:]
    losses = F.relu(pos_score - nega_score + gamma)
    return torch.mean(losses, dim=1)
    # print(pos_score)
    # print(nega_score)
    # print(score)
    # print(loss)
    # return loss

loss = cal_loss(a, b, 3)


def cal_loss(repre_sr, repre_tg, gamma):
    '''
    repre shape: [batch_size, 1+nega_sample_num, embedding_dim]
    '''
    score = torch.sum(torch.abs(repre_sr - repre_tg), dim=-1)
    pos_score = score[:, :1]
    nega_score = score[:, 1:]
    loss = pos_score - nega_score + gamma
    loss = F.relu(loss)
    # print(loss)
    return torch.mean(loss, dim=1)
