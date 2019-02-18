import torch


def auction_lap(X, eps=None, compute_score=False):
    """
        X: n-by-n matrix w/ integer entries
        eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """
    # --
    # Init
    is_cuda = X.is_cuda
    dim = X.shape[0]
    eps = 1 / dim if eps is None else eps
    cost = torch.zeros((1, X.shape[1]))
    col_idx = torch.zeros(dim).long() - 1
    bids = torch.zeros(X.shape)

    if is_cuda:
        cost, col_idx, bids = cost.cuda(), col_idx.cuda(), bids.cuda()

    counter = 0
    while (col_idx == -1).any():
        counter += 1
        # --
        # Bidding

        unassigned = (col_idx == -1).nonzero().squeeze()
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)

        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]

        bid_increments = first_value - second_value + eps
        # print('size:', bid_increments.size())
        bids_ = bids[unassigned]
        bids_.zero_()
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1)
        )
        # --
        # Assignment

        have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()
        high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]
        cost[:, have_bidder] += high_bids
        col_idx[(col_idx.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        col_idx[high_bidders] = have_bidder.squeeze()

    row_idx = torch.tensor([i for i in range(dim)], dtype=torch.int64)
    if is_cuda:
        row_idx = row_idx.cuda()
    if not compute_score:
        return row_idx, col_idx
    score = X.gather(dim=1, index=col_idx.view(-1, 1)).sum()
    return score, row_idx, col_idx


if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3], [3, 1, 2], [2, 3, 1]], dtype=torch.float)
    print(a.gather(dim=1, index=torch.tensor([1, 1, 1]).view(-1, 1)))
    print(auction_lap(a))
