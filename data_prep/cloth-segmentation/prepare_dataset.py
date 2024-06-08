import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET

device = "cuda"

image_dir = "/notebooks/ayna/working_repo/IDM-VTON/deepfashion_dataset/images"
result_dir = "output_images"
mask_dir = "cloth_mask"
grey_mask_dir = "masked_image"
segmentation_mask_dir = "segmentation_masks"
checkpoint_path = "/notebooks/ayna/working_repo/IDM-VTON/data_prep/cloth-segmentation/cloth_segm_u2net_latest.pth"
do_palette = True
batch_size = 16  # Set an appropriate batch size

if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

if not os.path.exists(grey_mask_dir):
    os.makedirs(grey_mask_dir)

if not os.path.exists(segmentation_mask_dir):
    os.makedirs(segmentation_mask_dir)

def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.image_list[idx]

transforms_list = [transforms.ToTensor(), Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

dataset = ImageDataset(image_dir, transform_rgb)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

palette = get_palette(4)

pbar = tqdm(total=len(dataset))

for batch in dataloader:
    imgs, img_names = batch
    imgs = imgs.to(device)

    with torch.no_grad():
        output_tensors = net(imgs)
        output_tensors = F.log_softmax(output_tensors[0], dim=1)
        output_tensors = torch.max(output_tensors, dim=1, keepdim=True)[1]
        output_tensors = torch.squeeze(output_tensors, dim=1)

    output_arrays = output_tensors.cpu().numpy()

    for i in range(len(img_names)):
        img_name = img_names[i]
        output_arr = output_arrays[i]

        # Save the segmentation mask with only the upper body (ID = 1)
        upper_body_mask = (output_arr == 1).astype(np.uint8) * 255
        upper_body_img = Image.fromarray(upper_body_mask, mode="L")

        # Find bounding box of the upper body
        bbox = upper_body_img.getbbox()
        if bbox:
            upper_body_cropped = upper_body_img.crop(bbox)
            cropped_width, cropped_height = upper_body_cropped.size

            # Create a new white image with the same size as the original image
            centered_img = Image.new("RGB", (imgs.size(3), imgs.size(2)), (255, 255, 255))

            # Calculate position to paste the cropped image
            paste_x = (centered_img.width - cropped_width) // 2
            paste_y = (centered_img.height - cropped_height) // 2

            # Convert image tensor to uint8 numpy array
            img_array = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)

            # Paste the cropped upper body image onto the centered white background
            centered_img.paste(img_pil.crop(bbox), (paste_x, paste_y), upper_body_cropped)

            # Create a black and white segmented mask based on the centered image
            centered_mask_img = Image.new("L", centered_img.size, 0)
            centered_mask_img.paste(upper_body_cropped, (paste_x, paste_y))

            centered_mask_img.save(os.path.join(segmentation_mask_dir, img_name[:-3] + "png"))

            centered_img.save(os.path.join(mask_dir, img_name[:-3] + "png"))

        # Save the output image
        output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
        if do_palette:
            output_img.putpalette(palette)
        output_img.save(os.path.join(result_dir, img_name[:-3] + "png"))

        # Save the image with grey mask for upper body
        grey_color = (128, 128, 128)
        grey_mask_img = Image.fromarray((imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).convert("RGB")
        grey_draw = ImageDraw.Draw(grey_mask_img)
        for y in range(output_arr.shape[0]):
            for x in range(output_arr.shape[1]):
                if output_arr[y, x] == 1:
                    grey_draw.point((x, y), fill=grey_color)
        grey_mask_img.save(os.path.join(grey_mask_dir, img_name[:-3] + "png"))

    pbar.update(len(img_names))

pbar.close()
