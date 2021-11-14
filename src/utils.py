import random

import numpy as np
import torch


def set_seed(seed: int,
             use_cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_y(x, roots=[0.0, 0.0]):
    _x = 1
    for root in roots:
        _x *= (x - root)
    return _x
