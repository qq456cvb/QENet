import torch
import h5py
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R


class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        fns = [line.rstrip() for line in open(cfg.data_path)]
        pts = []
        labels = []
        for fn in fns:
            f = h5py.File(os.path.join(os.path.dirname(cfg.data_path), fn), 'r')
            data = f['data'][:]
            label = f['label'][:]
            pts.append(data)
            labels.append(label)
        self.pts = np.concatenate(pts, 0)
        self.labels = np.concatenate(labels, 0)[:, 0]

    def __len__(self):
        return self.pts.shape[0]

    def __getitem__(self, index):
        pt = self.pts[index]
        label = self.labels[index]

        nbrs = NearestNeighbors(n_neighbors=self.cfg.lrf_nn, algorithm='auto').fit(pt)
        distances, indices = nbrs.kneighbors(pt)

        knns = pt[indices]
        _, lrfs = np.linalg.eigh(np.transpose(pt[indices], axes=(0, 2, 1)) @ knns)
        r = R.from_matrix(lrfs).as_quat()
        
        x = knns[:, :self.cfg.k]
        lrfs = r[indices[:, :self.cfg.k]]

        subset = np.random.choice(x.shape[0], self.cfg.rand_sample, replace=False)
        return x[subset].astype(np.float32), lrfs[subset].astype(np.float32), label.astype(np.int64)


if __name__ == "__main__":
    ds = ModelNetDataset('D:/data/modelnet40_ply_hdf5_2048/train_files.txt')
    for d in ds:
        print(d[0].shape, d[1].shape)
        break