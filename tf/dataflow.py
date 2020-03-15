from tensorpack import *
import numpy as np


class ModelNetDataFlow(RNGDataFlow):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        yield [np.random.randn(64, 9, 3), np.random.randn(64, 9, 4)]

    def __len__(self):
        return 10000
