import os
import gdown
import torch

from networks import U2NET
from utils.saving_utils import save_checkpoint

os.makedirs("prev_checkpoints", exist_ok=True)
gdown.download(
    "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
    "./prev_checkpoints/u2net.pth",
    quiet=False,
)

u_net = U2NET(in_ch=3, out_ch=4)
save_checkpoint(u_net, os.path.join("prev_checkpoints", "u2net_random.pth"))

# u2net.pth contains trained weights
trained_net_pth = os.path.join("prev_checkpoints", "u2net.pth")
# u2net_random.pth contains random weights
custom_net_pth = os.path.join("prev_checkpoints", "u2net_random.pth")

net_state_dict = torch.load(trained_net_pth)
count = 0
for k, v in net_state_dict.items():
    count += 1
print("Total number of layers in trained model are: {}".format(count))

custom_state_dict = torch.load(custom_net_pth)
count = 0
for k, v in custom_state_dict.items():
    count += 1
print("Total number of layers in trained model are: {}".format(count))

total_count = 0
update_count = 0
for k, v in net_state_dict.items():
    total_count += 1
    if custom_state_dict[k].shape == v.shape:
        update_count += 1
        custom_state_dict[k] = v

print(
    "Out of {} layers in custom network, {} layers weights are recovered from trained model".format(
        total_count, update_count
    )
)
torch.save(
    custom_state_dict, os.path.join("prev_checkpoints", "cloth_segm_unet_surgery.pth")
)
print("cloth_segm_unet_surgery.pth is generated in prev_checkpoints directory!")