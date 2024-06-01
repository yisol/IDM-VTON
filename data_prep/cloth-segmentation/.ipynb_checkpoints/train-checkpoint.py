import os
import sys
import time
import yaml
import cv2
import pprint
import traceback
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from data.custom_dataset_data_loader import CustomDatasetDataLoader, sample_data


from options.base_options import parser
from utils.tensorboard_utils import board_add_images
from utils.saving_utils import save_checkpoints
from utils.saving_utils import load_checkpoint, load_checkpoint_mgpu
from utils.distributed import get_world_size, set_seed, synchronize, cleanup

from networks import U2NET


def options_printing_saving(opt):
    os.makedirs(opt.logs_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "checkpoints"), exist_ok=True)

    # Saving options in yml file
    option_dict = vars(opt)
    with open(os.path.join(opt.save_dir, "training_options.yml"), "w") as outfile:
        yaml.dump(option_dict, outfile)

    for key, value in option_dict.items():
        print(key, value)


def training_loop(opt):

    if opt.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK"))
        # Unique only on individual node.
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
        local_rank = 0

    u_net = U2NET(in_ch=3, out_ch=4)
    if opt.continue_train:
        u_net = load_checkpoint(u_net, opt.unet_checkpoint)
    u_net = u_net.to(device)
    u_net.train()

    if local_rank == 0:
        with open(os.path.join(opt.save_dir, "networks.txt"), "w") as outfile:
            print("<----U-2-Net---->", file=outfile)
            print(u_net, file=outfile)

    if opt.distributed:
        u_net = nn.parallel.DistributedDataParallel(
            u_net,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        print("Going super fast with DistributedDataParallel")

    # initialize optimizer
    optimizer = optim.Adam(
        u_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    custom_dataloader = CustomDatasetDataLoader()
    custom_dataloader.initialize(opt)
    loader = custom_dataloader.get_loader()

    if local_rank == 0:
        dataset_size = len(custom_dataloader)
        print("Total number of images avaliable for training: %d" % dataset_size)
        writer = SummaryWriter(opt.logs_dir)
        print("Entering training loop!")

    # loss function
    weights = np.array([1, 1.5, 1.5, 1.5], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)

    pbar = range(opt.iter)
    get_data = sample_data(loader)

    start_time = time.time()
    # Main training loop
    for itr in pbar:
        data_batch = next(get_data)
        image, label = data_batch
        image = Variable(image.to(device))
        label = label.type(torch.long)
        label = Variable(label.to(device))

        d0, d1, d2, d3, d4, d5, d6 = u_net(image)

        loss0 = loss_CE(d0, label)
        loss1 = loss_CE(d1, label)
        loss2 = loss_CE(d2, label)
        loss3 = loss_CE(d3, label)
        loss4 = loss_CE(d4, label)
        loss5 = loss_CE(d5, label)
        loss6 = loss_CE(d6, label)
        del d1, d2, d3, d4, d5, d6

        total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        for param in u_net.parameters():
            param.grad = None

        total_loss.backward()
        if opt.clip_grad != 0:
            nn.utils.clip_grad_norm_(u_net.parameters(), opt.clip_grad)
        optimizer.step()

        if local_rank == 0:
            # printing and saving work
            if itr % opt.print_freq == 0:
                pprint.pprint(
                    "[step-{:08d}] [time-{:.3f}] [total_loss-{:.6f}]  [loss0-{:.6f}]".format(
                        itr, time.time() - start_time, total_loss, loss0
                    )
                )

            if itr % opt.image_log_freq == 0:
                d0 = F.log_softmax(d0, dim=1)
                d0 = torch.max(d0, dim=1, keepdim=True)[1]
                visuals = [[image, torch.unsqueeze(label, dim=1) * 85, d0 * 85]]
                board_add_images(writer, "grid", visuals, itr)

            writer.add_scalar("total_loss", total_loss, itr)
            writer.add_scalar("loss0", loss0, itr)

            if itr % opt.save_freq == 0:
                save_checkpoints(opt, itr, u_net)

    print("Training done!")
    if local_rank == 0:
        itr += 1
        save_checkpoints(opt, itr, u_net)


if __name__ == "__main__":

    opt = parser()

    if opt.distributed:
        if int(os.environ.get("LOCAL_RANK")) == 0:
            options_printing_saving(opt)
    else:
        options_printing_saving(opt)

    try:
        if opt.distributed:
            print("Initialize Process Group...")
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

        set_seed(1000)
        training_loop(opt)
        cleanup(opt.distributed)
        print("Exiting..............")

    except KeyboardInterrupt:
        cleanup(opt.distributed)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        cleanup(opt.distributed)
