import torch


class SpreadLoss(torch.nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, a, label):
        mask = torch.zeros_like(a, dtype=torch.bool).cuda()
        mask.scatter_(-1, label.view(-1, 1), True)

        at = a[mask][:, None]
        ai = a[~mask].reshape(a.size(0), -1)

        assert not torch.any(torch.isnan(a))

        assert 0 <= label.min() <= 39

        print((at - ai).max(), (at - ai).min())
        return torch.mean(torch.sum(torch.max(torch.zeros_like(ai).cuda(), self.margin - (at - ai)) ** 2, -1))


if __name__ == "__main__":
    l = SpreadLoss()
    res = l(torch.zeros((2, 40)).cuda(), torch.zeros((2,), dtype=torch.long).cuda())
    print(res.item())