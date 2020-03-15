import torch
from qenet import QENet

class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qenet1 = QENet([int(i) for i in cfg.first_transform_units.split(',')], 1, cfg.immediate_nchannels)
        self.qenet2 = QENet([int(i) for i in cfg.second_transform_units.split(',')], cfg.immediate_nchannels, cfg.nclasses)

    def forward(self, x, lrfs):
        B, N, K, _ = x.size()
        a = torch.ones(B, N, K, 1).cuda()
        cap, a = self.qenet1(x, lrfs[..., None, :], a)
        cap, a = self.qenet2(x[:, None, :, 0], cap[:, None], a[:, None])
        cap = cap[:, 0]
        a = a[:, 0]
        return cap, a

if __name__ == "__main__":
    pass