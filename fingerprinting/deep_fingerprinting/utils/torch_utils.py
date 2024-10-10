import random

import numpy as np
import torch

from utils import utils

def device():
    """
    Check if cuda is avaliable else choose the cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu') # For testing purposes
    print(f"pyTorch is using {device}")
    return device


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_GPU():
    return torch.cuda.empty_cache()

class Collate_Fn():
    def __init__(self, rng: np.random.Generator, p:float = 0.33):
        self.rng = rng
        self.prob = p

    def __call__(self, batch):
        if self.rng.random() <= self.prob:
            mask = torch.from_numpy(utils.cutout_spec_augment_mask(self.rng))
            x_orgs = [mask * sample[0] for sample in batch]
            x_augs = [mask * sample[1] for sample in batch]
            return torch.stack(x_orgs), torch.stack(x_augs)
        else:
            x_orgs, x_augs = list(zip(*batch))
            return torch.stack(x_orgs), torch.stack(x_augs)
