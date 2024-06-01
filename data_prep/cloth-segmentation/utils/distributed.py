import os
import math
import numpy as np
import random
import pickle
import torch
from torch import distributed as dist
from torch.utils.data.sampler import Sampler


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    dist.barrier()


def cleanup(distributed):
    if distributed:
        dist.destroy_process_group()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()
