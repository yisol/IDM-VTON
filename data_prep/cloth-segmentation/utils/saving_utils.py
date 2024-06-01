import os
import copy
import cv2
import numpy as np
from collections import OrderedDict

import torch


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def load_checkpoint_mgpu(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def save_checkpoint(model, save_path):
    print(save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)


def save_checkpoints(opt, itr, net):
    save_checkpoint(
        net,
        os.path.join(opt.save_dir, "checkpoints", "itr_{:08d}_u2net.pth".format(itr)),
    )