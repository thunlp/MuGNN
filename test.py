import torch
import torch.nn.functional as F

def ss(score):
    # distance = F.normalize(score, p=2, dim=-1)
    # pos_score = distance[:,0,:].sum(-1, keepdim=True)
    # nega_score = distance[:,1,:].sum(-1, keepdim=True)

    pos_score = torch.tensor([1 for _ in range(3)], dtype=torch.float).view(-1,1)
    neg_score = torch.tensor([2 for _ in range(3)], dtype=torch.float).view(-1,1)
    y = torch.FloatTensor([-1.0])
    loss = F.margin_ranking_loss(pos_score, neg_score, y, margin=3)
    print(loss)

ss(1)