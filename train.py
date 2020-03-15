import torch
import hydra
from dataset import ModelNetDataset
from model import Model
from loss import SpreadLoss
from tqdm import tqdm
import numpy as np


@hydra.main(config_path='config/config.yaml')
def train(cfg):
    print(cfg.pretty())

    ds = ModelNetDataset(cfg)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    model = Model(cfg).cuda()
    model.train()

    crit = SpreadLoss(cfg.min_margin)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(cfg.max_epoch):
        td = tqdm(dataloader)
        for i, batch in enumerate(td):
            pts, lrfs,  labels = batch

            pts = pts.cuda()
            lrfs = lrfs.cuda()
            labels = labels.cuda()

            opt.zero_grad()

            # with torch.autograd.detect_anomaly():
            output_cap, output_a = model(pts, lrfs)
            loss = crit(output_a, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            for p in model.parameters():
                assert not torch.any(torch.isnan(p.grad))
            opt.step()

            td.set_description('iter {}/{}'.format(i, len(td)))
            td.set_postfix({'loss': loss.item()})

            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (e + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
            

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    train()