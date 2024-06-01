import os
from PIL import Image
import cv2
import numpy as np
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return "BaseDataset"

    def initialize(self, opt):
        pass


class Rescale_fixed(object):
    """Rescale the input image into given size.

    Args:
        (w,h) (tuple): output size or x (int) then resized will be done in (x,x).
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        return image.resize(self.output_size, Image.BICUBIC)


class Rescale_custom(object):
    """Rescale the input image and target image into randomly selected size with lower bound of min_size arg.

    Args:
        min_size (int): Minimum desired output size.
    """

    def __init__(self, min_size, max_size):
        assert isinstance(min_size, (int, float))
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, sample):

        input_image, target_image = sample["input_image"], sample["target_image"]

        assert input_image.size == target_image.size
        w, h = input_image.size

        # Randomly select size to resize
        if min(self.max_size, h, w) > self.min_size:
            self.output_size = np.random.randint(
                self.min_size, min(self.max_size, h, w)
            )
        else:
            self.output_size = self.min_size

        # calculate new size by keeping aspect ratio same
        if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size, self.output_size * w / h

        new_w, new_h = int(new_w), int(new_h)
        input_image = input_image.resize((new_w, new_h), Image.BICUBIC)
        target_image = target_image.resize((new_w, new_h), Image.BICUBIC)
        return {"input_image": input_image, "target_image": target_image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, sample):
        input_image, target_image = sample["input_image"], sample["target_image"]

        return {
            "input_image": self.totensor(input_image),
            "target_image": self.totensor(target_image),
        }


class RandomCrop_custom(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.randomcrop = transforms.RandomCrop(self.output_size)

    def __call__(self, sample):
        input_image, target_image = sample["input_image"], sample["target_image"]
        cropped_imgs = self.randomcrop(torch.cat((input_image, target_image)))

        return {
            "input_image": cropped_imgs[
                :3,
                :,
            ],
            "target_image": cropped_imgs[
                3:,
                :,
            ],
        }


class Normalize_custom(object):
    """Normalize given dict into given mean and standard dev

    Args:
        mean (tuple or int): Desired mean to substract from dict's tensors
        std (tuple or int): Desired std to divide from dict's tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float, tuple))
        if isinstance(mean, float):
            self.mean = (mean, mean, mean)
        else:
            assert len(mean) == 3
            self.mean = mean

        if isinstance(std, float):
            self.std = (std, std, std)
        else:
            assert len(std) == 3
            self.std = std

        self.normalize = transforms.Normalize(self.mean, self.std)

    def __call__(self, sample):
        input_image, target_image = sample["input_image"], sample["target_image"]

        return {
            "input_image": self.normalize(input_image),
            "target_image": self.normalize(target_image),
        }


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"
