import torch
import torch.nn.functional as F

def gcn_align_loss(repre_sr, repre_tg, gamma):
    '''
    repre shape: [batch_size, 1+nega_sample_num, embedding_dim]
    '''
    score = torch.sum(torch.abs(repre_sr - repre_tg), dim=-1)
    pos_score = score[:, :1]
    nega_score = score[:, 1:]
    losses = F.relu(pos_score - nega_score + gamma)
    return torch.mean(losses)
